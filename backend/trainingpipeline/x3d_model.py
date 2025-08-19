import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# --- Attention Fusion Module ---
class AttentionFusion(nn.Module):
    """
    A cross-attention module to fuse X3D features and motion features.
    X3D features act as keys/values, motion features as query.
    """
    def __init__(self, x3d_dim: int, motion_dim: int):
        super().__init__()
        self.query_transform = nn.Linear(motion_dim, x3d_dim)
        self.key_transform = nn.Linear(x3d_dim, x3d_dim)
        self.value_transform = nn.Linear(x3d_dim, x3d_dim)
        self.scale = x3d_dim ** -0.5

    def forward(self, x3d_features: torch.Tensor, motion_features: torch.Tensor) -> torch.Tensor:
        # X3D features as keys/values, motion features as query
        query = self.query_transform(motion_features)  # [B, x3d_dim]
        key = self.key_transform(x3d_features)        # [B, x3d_dim]
        value = self.value_transform(x3d_features)      # [B, x3d_dim]

        # Calculate attention scores
        scores = torch.bmm(query.unsqueeze(1), key.unsqueeze(2)).squeeze(2) * self.scale # [B, 1]
        attention_weights = F.softmax(scores, dim=1) # [B, 1]

        # Create weighted X3D features and combine with motion features
        fused_x3d_features = attention_weights * value
        
        # Concatenate the original X3D features, motion features, and the new fused features
        # This provides both explicit and attention-weighted information.
        combined_features = torch.cat([x3d_features, motion_features, fused_x3d_features], dim=1)
        
        return combined_features


class MotionEnhancementModule(nn.Module):
    """
    Module to process optical flow information and enhance motion features.
    Lightweight design for real-time inference.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        
        # Reduced complexity to prevent overfitting
        self.flow_conv = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=1, stride=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1, stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1, stride=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        # Simplified motion feature processing
        self.motion_fc = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # Reduced dropout
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Proper weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, optical_flow: torch.Tensor) -> torch.Tensor:
        """
        Args:
            optical_flow: [B, C, T, H, W] optical flow tensor
        Returns:
            motion_features: [B, output_dim] motion features
        """
        # Process optical flow through 3D CNN
        flow_features = self.flow_conv(optical_flow)  # [B, 128, 1, 1, 1]
        flow_features = flow_features.view(flow_features.size(0), -1)  # [B, 128]
        
        # Generate motion features
        motion_features = self.motion_fc(flow_features)  # [B, output_dim]
        
        return motion_features


class X3DViolenceDetector(nn.Module):
    """
    FIXED X3D-based violence detection model with proper mixed precision support.
    Model stays in float32, mixed precision is handled by autocast() in training.
    """
    
    def __init__(
        self,
        x3d_model_name: str = "x3d_m",
        num_classes: int = 2,
        dropout_rate: float = 0.2,
        use_motion_enhancement: bool = True,
        motion_weight: float = 0.3,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.use_motion_enhancement = use_motion_enhancement
        self.motion_weight = motion_weight
        self.num_classes = num_classes
        self.device = device
        
        # Load pre-trained X3D model
        print(f"Loading {x3d_model_name} model...")
        self.x3d_backbone = self._load_x3d_model(x3d_model_name)
        
        # Move model to device before dummy forward pass
        self.x3d_backbone.to(self.device)
        
        # Get feature dimension by running a forward pass
        self.feature_dim = self._get_feature_dim()
        
        # Motion enhancement module
        if self.use_motion_enhancement:
            self.motion_module = MotionEnhancementModule(
                input_dim=3,
                output_dim=128
            )
            self.motion_module.to(self.device)
            
            self.attention_fusion = AttentionFusion(
                x3d_dim=self.feature_dim,
                motion_dim=128
            )
            self.attention_fusion.to(self.device)
            
            # The total features are now X3D features + Motion features + Fused features
            total_features = self.feature_dim + 128 + self.feature_dim
        else:
            total_features = self.feature_dim
        
        # Create classifier with proper initialization
        self.classifier = self._create_classifier(total_features, dropout_rate)
        self.classifier.to(self.device)
        
        print(f"Model initialized with {total_features} input features")
        print(f"X3D features: {self.feature_dim}, Motion features: {128 if use_motion_enhancement else 0}")
        print(f"Model dtype: {next(self.parameters()).dtype}")  # Should be float32
    
    def _load_x3d_model(self, model_name: str):
        """Load pre-trained X3D model from torch hub"""
        try:
            model = torch.hub.load(
                'facebookresearch/pytorchvideo', 
                model_name, 
                pretrained=True
            )
            return model
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            print("Falling back to x3d_s model...")
            model = torch.hub.load(
                'facebookresearch/pytorchvideo', 
                'x3d_s', 
                pretrained=True
            )
            return model
    
    def _get_feature_dim(self):
        """Determine feature dimension by running a test forward pass"""
        dummy_input = torch.zeros((1, 3, 16, 224, 224), device=self.device, dtype=torch.float32)
        
        with torch.no_grad():
            features = self._extract_x3d_features(dummy_input)
            
        return features.shape[1]
    
    def _extract_x3d_features(self, rgb_frames: torch.Tensor) -> torch.Tensor:
        """Extract features from X3D backbone"""
        x = rgb_frames
        
        # Forward through X3D backbone (excluding final classification head)
        if hasattr(self.x3d_backbone, 'blocks'):
            for i, block in enumerate(self.x3d_backbone.blocks):
                # Skip the final classification head block
                if hasattr(block, 'proj') and i == len(self.x3d_backbone.blocks) - 1:
                    continue
                x = block(x)
        else:
            # Fallback: extract features before final layer
            features = self.x3d_backbone.features(rgb_frames)
            x = features
        
        # Global average pooling
        if len(x.shape) == 5:  # [B, C, T, H, W]
            x = F.adaptive_avg_pool3d(x, (1, 1, 1))  # [B, C, 1, 1, 1]
            x = x.view(x.size(0), -1)  # [B, C]
        elif len(x.shape) == 2:  # [B, features]
            pass  # Already flattened
        else:
            x = x.view(x.size(0), -1)  # Flatten
        
        return x
    
    def _create_classifier(self, input_dim: int, dropout_rate: float):
        """Create classifier with proper initialization"""
        classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),  # Smaller hidden layer
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, self.num_classes)
        )
        
        # CRITICAL: Proper weight initialization to prevent gradient explosion
        for m in classifier.modules():
            if isinstance(m, nn.Linear):
                # Use Xavier initialization for better gradient flow
                nn.init.xavier_normal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0)
        
        return classifier
    
    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass with mixed precision compatibility
        Note: autocast() in trainer will handle precision conversion automatically
        """
        rgb_frames = data['rgb']  # [B, C, T, H, W]
        
        # Extract X3D features
        x3d_features = self._extract_x3d_features(rgb_frames)
        
        # Motion enhancement
        if self.use_motion_enhancement and 'flow' in data:
            optical_flow = data['flow']
            motion_features = self.motion_module(optical_flow)
            
            # Use attention to combine features
            combined_features = self.attention_fusion(x3d_features, motion_features)
        else:
            combined_features = x3d_features
        
        # Classification
        logits = self.classifier(combined_features)
        
        return logits


class StableCrossEntropyLoss(nn.Module):
    """
    Stable cross-entropy loss with optional label smoothing.
    Much more stable than Focal Loss for this application.
    """
    
    def __init__(self, label_smoothing: float = 0.05, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.weight = weight
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [B, num_classes] model predictions (logits)
            targets: [B] ground truth labels
        """
        return F.cross_entropy(
            predictions, 
            targets, 
            label_smoothing=self.label_smoothing,
            weight=self.weight
        )


def create_model(
    model_name: str = "x3d_m",
    num_classes: int = 2,
    use_motion_enhancement: bool = True,
    device: str = "cuda"
) -> X3DViolenceDetector:
    """
    Create and initialize the FIXED X3D violence detection model
    Model will remain in float32 for mixed precision compatibility
    """
    model = X3DViolenceDetector(
        x3d_model_name=model_name,
        num_classes=num_classes,
        use_motion_enhancement=use_motion_enhancement,
        dropout_rate=0.2,  # Reduced dropout
        motion_weight=0.3,
        device=device
    )
    
    # Verify model is in correct dtype
    model_dtype = next(model.parameters()).dtype
    print(f"Model created with dtype: {model_dtype}")
    
    if model_dtype != torch.float32:
        print("WARNING: Model not in float32, converting...")
        model = model.float()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: {model_name}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Motion enhancement: {use_motion_enhancement}")
    
    return model


if __name__ == "__main__":
    # Test the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing mixed precision compatible model on device: {device}")
    
    # Create model
    model = create_model(
        model_name="x3d_m",
        use_motion_enhancement=True,
        device=device
    )
    
    # Test forward pass
    batch_size, channels, frames, height, width = 2, 3, 16, 224, 224
    
    dummy_data = {
        'rgb': torch.randn(batch_size, channels, frames, height, width, device=device, dtype=torch.float32),
        'flow': torch.randn(batch_size, channels, frames, height, width, device=device, dtype=torch.float32)
    }
    
    print("\nTesting forward pass...")
    with torch.no_grad():
        output = model(dummy_data)
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        
        # Check if logits are reasonable
        if abs(output.max().item()) < 10:
            print("✅ Logits are in reasonable range")
        else:
            print("❌ Logits are still extreme")
    
    print("\nFixed model test completed!")
    print("Model is ready for mixed precision training with autocast()")