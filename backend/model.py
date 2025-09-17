import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class MotionEnhancementModule(nn.Module):
    """
    Clean motion enhancement module for processing optical flow features.
    Uses 3D CNN to extract motion patterns from optical flow data.
    """
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 128, output_dim: int = 128):
        super().__init__()
        
        # 3D CNN for optical flow processing with smaller temporal kernels
        self.flow_conv = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=1, stride=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1, stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            nn.Conv3d(64, hidden_dim, kernel_size=(3, 3, 3), padding=1, stride=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        # Motion feature processing
        self.motion_fc = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming normal"""
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
        # Process optical flow through 3D CNN
        flow_features = self.flow_conv(optical_flow)  # [B, hidden_dim, 1, 1, 1]
        flow_features = flow_features.view(flow_features.size(0), -1)  # [B, hidden_dim]
        
        # Generate motion features
        motion_features = self.motion_fc(flow_features)  # [B, output_dim]
        
        return motion_features


class X3DViolenceDetector(nn.Module):
    """
    Clean X3D model for violence detection.
    Removed dead code while maintaining functionality.
    """
    
    def __init__(
        self,
        x3d_model_name: str = "x3d_m",
        num_classes: int = 2,
        dropout_rate: float = 0.15,
        use_motion_enhancement: bool = True,
        device: str = "auto"
    ):
        super().__init__()
        
        # Auto-detect device with CUDA compatibility check
        if device == "auto":
            if torch.cuda.is_available():
                try:
                    # Test CUDA compatibility with a small tensor operation
                    test_tensor = torch.zeros(1, device='cuda')
                    test_tensor = test_tensor + 1
                    device = "cuda"
                    print("CUDA is available and compatible")
                except Exception as e:
                    print(f"CUDA available but incompatible: {e}")
                    print("Falling back to CPU")
                    device = "cpu"
            else:
                device = "cpu"
                print("CUDA not available, using CPU")
        
        self.use_motion_enhancement = use_motion_enhancement
        self.num_classes = num_classes
        self.device = device
        
        # Load and optimize X3D backbone
        print(f"Loading {x3d_model_name} model...")
        self.x3d_backbone = self._load_and_optimize_x3d(x3d_model_name)
        self.x3d_backbone.to(self.device)
        
        # Get feature dimension
        self.feature_dim = self._get_feature_dim()
        
        # Motion enhancement module
        if self.use_motion_enhancement:
            self.motion_module = MotionEnhancementModule(
                input_dim=3,
                hidden_dim=128,
                output_dim=128
            )
            self.motion_module.to(self.device)
            total_features = self.feature_dim + 128
        else:
            total_features = self.feature_dim
        
        # Classifier
        self.classifier = self._create_classifier(total_features, dropout_rate)
        self.classifier.to(self.device)
        
        print(f"Model initialized with {total_features} input features")
        print(f"X3D features: {self.feature_dim}, Motion features: {128 if use_motion_enhancement else 0}")
    
    def _load_and_optimize_x3d(self, model_name: str):
        """Load X3D and apply temporal kernel optimizations"""
        try:
            model = torch.hub.load(
                'facebookresearch/pytorchvideo', 
                model_name, 
                pretrained=True
            )
            
            # Apply temporal kernel optimizations
            self._optimize_temporal_kernels(model)
            
            return model
            
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            print("Falling back to x3d_s model...")
            model = torch.hub.load(
                'facebookresearch/pytorchvideo', 
                'x3d_s', 
                pretrained=True
            )
            self._optimize_temporal_kernels(model)
            return model
    
    def _optimize_temporal_kernels(self, model):
        """
        Reduce large temporal kernels to improve motion capture.
        Changes kernels >8 to size 3 for better temporal resolution.
        """
        def modify_conv3d(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Conv3d):
                    # Reduce large temporal kernels
                    if child.kernel_size[0] > 8:
                        # Create new conv with smaller temporal kernel
                        new_kernel = (3, child.kernel_size[1], child.kernel_size[2])
                        new_padding = (1, child.padding[1], child.padding[2])
                        
                        new_conv = nn.Conv3d(
                            child.in_channels,
                            child.out_channels,
                            kernel_size=new_kernel,
                            stride=child.stride,
                            padding=new_padding,
                            bias=child.bias is not None
                        )
                        
                        # Copy weights (center crop for temporal dimension)
                        with torch.no_grad():
                            old_t = child.weight.size(2)
                            new_t = 3
                            start_t = (old_t - new_t) // 2
                            
                            new_conv.weight.data = child.weight.data[:, :, start_t:start_t+new_t, :, :]
                            if child.bias is not None:
                                new_conv.bias.data = child.bias.data
                        
                        setattr(module, name, new_conv)
                        print(f"Reduced temporal kernel: {child.kernel_size} -> {new_kernel}")
                else:
                    modify_conv3d(child)
        
        modify_conv3d(model)
    
    def _get_feature_dim(self):
        """Determine X3D feature dimension"""
        dummy_input = torch.zeros((1, 3, 16, 336, 336), device=self.device, dtype=torch.float32)
        
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
        """Create classifier optimized for small datasets"""
        classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, self.num_classes)
        )
        
        # Initialize weights
        for m in classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0)
        
        return classifier
    
    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass"""
        rgb_frames = data['rgb']  # [B, C, T, H, W]
        
        # Extract X3D features
        x3d_features = self._extract_x3d_features(rgb_frames)
        
        # Motion enhancement
        if self.use_motion_enhancement and 'flow' in data:
            optical_flow = data['flow']
            motion_features = self.motion_module(optical_flow)
            
            # Simple concatenation
            combined_features = torch.cat([x3d_features, motion_features], dim=1)
        else:
            combined_features = x3d_features
        
        # Get classification logits
        logits = self.classifier(combined_features)
        
        return logits


# Backward compatibility - alias for any imports using the old name
OptimizedX3DViolenceDetector = X3DViolenceDetector


class StableCrossEntropyLoss(nn.Module):
    """Standard cross-entropy loss with label smoothing option"""
    
    def __init__(self, label_smoothing: float = 0.05, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.weight = weight
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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
    device: str = "auto"
) -> X3DViolenceDetector:
    """Create clean X3D violence detection model"""
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = X3DViolenceDetector(
        x3d_model_name=model_name,
        num_classes=num_classes,
        use_motion_enhancement=use_motion_enhancement,
        dropout_rate=0.15,
        device=device
    )
    
    # Ensure model is in correct dtype
    model_dtype = next(model.parameters()).dtype
    if model_dtype != torch.float32:
        model = model.float()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Clean Model: {model_name}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Motion enhancement: {use_motion_enhancement}")
    
    return model


if __name__ == "__main__":
    # Test the clean model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing clean model on device: {device}")
    
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
    
    print("\nTesting clean forward pass...")
    with torch.no_grad():
        output = model(dummy_data)
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    print("\nClean model ready for training!")