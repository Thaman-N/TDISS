import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class MotionEnhancementModule(nn.Module):
    """
    Module to process optical flow information and enhance motion features.
    Lightweight design for real-time inference.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()
        
        # Lightweight 3D CNN for optical flow processing
        self.flow_conv = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=1, stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1, stride=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1, stride=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        # Motion feature fusion
        self.motion_fc = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Attention mechanism for motion importance
        self.motion_attention = nn.Sequential(
            nn.Linear(output_dim, output_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim // 4, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, optical_flow: torch.Tensor) -> torch.Tensor:
        """
        Args:
            optical_flow: [B, C, T, H, W] optical flow tensor
        Returns:
            motion_features: [B, output_dim] motion features
        """
        # Ensure input is float32
        optical_flow = optical_flow.float()
        
        # Process optical flow through 3D CNN
        flow_features = self.flow_conv(optical_flow)  # [B, 256, 1, 1, 1]
        flow_features = flow_features.view(flow_features.size(0), -1)  # [B, 256]
        
        # Generate motion features
        motion_features = self.motion_fc(flow_features)  # [B, output_dim]
        
        # Apply attention to emphasize important motion patterns
        attention_weights = self.motion_attention(motion_features)
        enhanced_motion = motion_features * attention_weights
        
        return enhanced_motion


class X3DViolenceDetector(nn.Module):
    """
    X3D-based violence detection model with motion enhancement.
    Optimized for real-time inference and high accuracy.
    """
    
    def __init__(
        self,
        x3d_model_name: str = "x3d_s",
        num_classes: int = 2,
        dropout_rate: float = 0.3,
        use_motion_enhancement: bool = True,
        motion_weight: float = 0.3
    ):
        super().__init__()
        
        self.use_motion_enhancement = use_motion_enhancement
        self.motion_weight = motion_weight
        
        # Load pre-trained X3D model
        print(f"Loading {x3d_model_name} model...")
        self.x3d_backbone = self._load_x3d_model(x3d_model_name)
        
        # Motion enhancement module
        if self.use_motion_enhancement:
            self.motion_module = MotionEnhancementModule(
                input_dim=3,  # RGB optical flow
                output_dim=256
            )
        
        # We'll initialize the classifier after we know the feature dimensions
        self.classifier = None
        self.feature_dim_determined = False
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        
        print("Model structure initialized. Classifier will be created after first forward pass.")
    
    def _create_classifier(self, feature_dim: int):
        """Create classifier once we know the feature dimension"""
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, self.num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier_weights()
        
        # Move to same device as the backbone
        device = next(self.x3d_backbone.parameters()).device
        self.classifier = self.classifier.to(device)
        
        self.feature_dim_determined = True
        print(f"Classifier created with input dimension: {feature_dim}")
    
    def _load_x3d_model(self, model_name: str):
        """Load pre-trained X3D model from torch hub"""
        try:
            model = torch.hub.load(
                'facebookresearch/pytorchvideo', 
                model_name, 
                pretrained=True
            )
            # Don't remove blocks - we'll extract features properly in forward()
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
    
    def _init_classifier_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            data: Dictionary containing 'rgb' and optionally 'flow' tensors
                 'rgb': [B, C, T, H, W] RGB video tensor
                 'flow': [B, C, T, H, W] optical flow tensor
        
        Returns:
            logits: [B, num_classes] classification logits
        """
        rgb_frames = data['rgb']  # [B, C, T, H, W]
        batch_size = rgb_frames.size(0)
        
        # Ensure input is float32 (compatible with mixed precision)
        rgb_frames = rgb_frames.float()
        
        # Extract spatial-temporal features using X3D
        with torch.cuda.amp.autocast():
            # Forward through X3D backbone (excluding final classification head)
            x = rgb_frames
            
            # Go through X3D blocks except the final head
            if hasattr(self.x3d_backbone, 'blocks'):
                for i, block in enumerate(self.x3d_backbone.blocks):
                    # Skip the final classification head block
                    if hasattr(block, 'proj') and i == len(self.x3d_backbone.blocks) - 1:
                        # This is the final projection layer, skip it
                        continue
                    x = block(x)
            else:
                # Fallback: use the full model but extract features before final layer
                x = self.x3d_backbone(rgb_frames)
            
            # Global average pooling to get feature vector
            if len(x.shape) == 5:  # [B, C, T, H, W]
                x3d_features = F.adaptive_avg_pool3d(x, (1, 1, 1))  # [B, C, 1, 1, 1]
                x3d_features = x3d_features.view(batch_size, -1)  # [B, C]
            elif len(x.shape) == 2:  # [B, features] - already flattened
                x3d_features = x
            else:
                # Handle other shapes
                x3d_features = x.view(batch_size, -1)
        
        # Motion enhancement
        if self.use_motion_enhancement and 'flow' in data:
            optical_flow = data['flow']  # [B, C, T, H, W]
            optical_flow = optical_flow.float()  # Ensure float32
            motion_features = self.motion_module(optical_flow)  # [B, 256]
            
            # Combine X3D and motion features
            combined_features = torch.cat([x3d_features, motion_features], dim=1)
        else:
            combined_features = x3d_features
        
        # Create classifier on first forward pass
        if not self.feature_dim_determined:
            feature_dim = combined_features.shape[1]
            self._create_classifier(feature_dim)
        
        # Classification
        logits = self.classifier(combined_features)  # [B, num_classes]
        
        return logits
    
    def get_feature_maps(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate feature maps for analysis
        """
        rgb_frames = data['rgb']
        batch_size = rgb_frames.size(0)
        
        features = {}
        
        # Ensure input is float32
        rgb_frames = rgb_frames.float()
        
        # X3D features
        with torch.cuda.amp.autocast():
            # Forward through X3D backbone (excluding final classification head)
            x = rgb_frames
            
            if hasattr(self.x3d_backbone, 'blocks'):
                for i, block in enumerate(self.x3d_backbone.blocks):
                    # Skip the final classification head block
                    if hasattr(block, 'proj') and i == len(self.x3d_backbone.blocks) - 1:
                        continue
                    x = block(x)
            else:
                x = self.x3d_backbone(rgb_frames)
            
            # Global average pooling to get feature vector
            if len(x.shape) == 5:  # [B, C, T, H, W]
                x3d_features = F.adaptive_avg_pool3d(x, (1, 1, 1))
                x3d_features = x3d_features.view(batch_size, -1)
            elif len(x.shape) == 2:
                x3d_features = x
            else:
                x3d_features = x.view(batch_size, -1)
            
            features['x3d_features'] = x3d_features
        
        # Motion features
        if self.use_motion_enhancement and 'flow' in data:
            optical_flow = data['flow'].float()
            motion_features = self.motion_module(optical_flow)
            features['motion_features'] = motion_features
        
        return features


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in violence detection.
    Addresses the problem of easy negatives dominating the loss.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, label_smoothing: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [B, num_classes] model predictions (logits)
            targets: [B] ground truth labels
        """
        # Apply label smoothing
        num_classes = predictions.size(1)
        targets_one_hot = F.one_hot(targets, num_classes).float()
        
        if self.label_smoothing > 0:
            targets_one_hot = targets_one_hot * (1 - self.label_smoothing) + \
                             self.label_smoothing / num_classes
        
        # Compute cross entropy
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        
        # Compute probabilities
        p_t = torch.exp(-ce_loss)
        
        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal loss
        focal_loss = alpha_t * focal_weight * ce_loss
        
        return focal_loss.mean()


def create_model(
    model_name: str = "x3d_s",
    num_classes: int = 2,
    use_motion_enhancement: bool = True,
    device: str = "cuda"
) -> X3DViolenceDetector:
    """
    Create and initialize the X3D violence detection model
    
    Args:
        model_name: X3D model variant ('x3d_xs', 'x3d_s', 'x3d_m', 'x3d_l')
        num_classes: Number of output classes (2 for violence detection)
        use_motion_enhancement: Whether to use motion enhancement
        device: Device to load the model on
    
    Returns:
        model: Initialized model
    """
    model = X3DViolenceDetector(
        x3d_model_name=model_name,
        num_classes=num_classes,
        use_motion_enhancement=use_motion_enhancement,
        motion_weight=0.3
    )
    
    model = model.to(device)
    
    # Count parameters (excluding classifier which isn't created yet)
    backbone_params = sum(p.numel() for p in model.x3d_backbone.parameters())
    motion_params = sum(p.numel() for p in model.motion_module.parameters()) if use_motion_enhancement else 0
    
    print(f"Model: {model_name}")
    print(f"Backbone parameters: {backbone_params:,}")
    if use_motion_enhancement:
        print(f"Motion module parameters: {motion_params:,}")
    print(f"Motion enhancement: {use_motion_enhancement}")
    print("Note: Classifier parameters will be counted after first forward pass")
    
    return model