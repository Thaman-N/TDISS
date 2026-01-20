import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class TemporalScaleAdaptiveBlock(nn.Module):
    """
    NOVELTY MODULE: 'FPS-Aware' Feature Calibration.
    Dynamically re-weights channels based on motion velocity to handle
    variable frame rate artifacts (Slide Shows vs Sped-Up).
    """
    def __init__(self, in_channels, reduction=16):
        super(TemporalScaleAdaptiveBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        
        # The "Speedometer" Branch
        # Calculates importance weights based on temporal velocity
        self.motion_fc = nn.Sequential(
            nn.Linear(in_channels, max(4, in_channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(4, in_channels // reduction), in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [Batch, Channel, Time, Height, Width]
        b, c, t, h, w = x.size()
        
        # 1. Calculate Feature Velocity (Diff between T and T-1)
        # "Is the video changing fast or slow?"
        current = x[:, :, 1:, :, :]
        prev    = x[:, :, :-1, :, :]
        velocity = torch.abs(current - prev)
        
        # Pad back to original time length (Time 0 has 0 velocity)
        velocity = F.pad(velocity, (0,0, 0,0, 1,0), "constant", 0)
        
        # 2. Get Global Speed Score per channel
        global_speed = self.avg_pool(velocity).view(b, c)
        
        # 3. Calibrate features based on speed
        # "If speed is high, suppress static texture channels, boost motion channels"
        weights = self.motion_fc(global_speed).view(b, c, 1, 1, 1)
        
        return x * weights


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


class CleanX3DViolenceDetector(nn.Module):
    """
    Clean X3D model for violence detection.
    Supports Optional Motion Enhancement and New TSA (FPS-Aware) Block.
    """
    
    def __init__(
        self,
        x3d_model_name: str = "x3d_m",
        num_classes: int = 2,
        dropout_rate: float = 0.15,
        use_motion_enhancement: bool = True,
        use_temporal_kernel_optimization: bool = True, # Explicit flag
        use_tsa_block: bool = False, # NEW: Flag for Algo Novelty
        device: str = "cuda"
    ):
        super().__init__()
        
        self.use_motion_enhancement = use_motion_enhancement
        self.use_temporal_kernel_optimization = use_temporal_kernel_optimization
        self.use_tsa_block = use_tsa_block
        self.num_classes = num_classes
        self.device = device
        
        # Load and optimize X3D backbone
        # print(f"Loading {x3d_model_name} model...")
        self.x3d_backbone = self._load_and_optimize_x3d(x3d_model_name)
        self.x3d_backbone.to(self.device)

        # === ALGORITHMIC NOVELTY: FPS-Aware Block ===
        # We will initialize it lazily in forward to auto-detect channel dims,
        # or we could hardcode. Lazy initialization is safer for different X3D variants.
        self.tsa_block = None 
        
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
        
        # print(f"Model initialized with {total_features} input features")
        # if self.use_tsa_block:
        #    print("✅ FPS-Adaptive TSA Block ENABLED (Novelty Module)")
    
    def _load_and_optimize_x3d(self, model_name: str):
        """Load X3D and apply temporal kernel optimizations if requested"""
        try:
            model = torch.hub.load(
                'facebookresearch/pytorchvideo', 
                model_name, 
                pretrained=True
            )
            
            # Apply temporal kernel optimizations ONLY if flag is True
            if self.use_temporal_kernel_optimization:
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
            if self.use_temporal_kernel_optimization:
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
                        # print(f"Reduced temporal kernel: {child.kernel_size} -> {new_kernel}")
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
        """
        Extract features from X3D backbone.
        Injects TSA Block if enabled.
        """
        x = rgb_frames
        
        # Forward through X3D backbone (excluding final classification head)
        if hasattr(self.x3d_backbone, 'blocks'):
            for i, block in enumerate(self.x3d_backbone.blocks):
                # Skip the final classification head block
                if hasattr(block, 'proj') and i == len(self.x3d_backbone.blocks) - 1:
                    continue
                
                x = block(x)

                # === INJECT NOVELTY HERE ===
                # Apply TSA Block after Block 3 (mid-level features)
                # This ensures we calibrate features before the final high-level semantic abstraction
                if i == 3 and self.use_tsa_block:
                    if self.tsa_block is None:
                        # Initialize on first run to match channel dimensions
                        channels = x.size(1)
                        self.tsa_block = TemporalScaleAdaptiveBlock(in_channels=channels).to(self.device)
                    
                    x = self.tsa_block(x)
                # ===========================
        
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
        
        # Extract X3D features (includes TSA block if enabled)
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
    use_temporal_kernel_optimization: bool = True,
    use_tsa_block: bool = False,
    device: str = "cuda"
) -> CleanX3DViolenceDetector:
    """Create clean X3D violence detection model"""
    
    model = CleanX3DViolenceDetector(
        x3d_model_name=model_name,
        num_classes=num_classes,
        use_motion_enhancement=use_motion_enhancement,
        use_temporal_kernel_optimization=use_temporal_kernel_optimization,
        use_tsa_block=use_tsa_block,
        dropout_rate=0.15,
        device=device
    )
    
    # Ensure model is in correct dtype
    model_dtype = next(model.parameters()).dtype
    if model_dtype != torch.float32:
        model = model.float()
    
    return model


if __name__ == "__main__":
    # Test the clean model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing clean model on device: {device}")
    
    # Test Config: ENABLE novelty block to verify shape compatibility
    model = create_model(
        model_name="x3d_m",
        use_motion_enhancement=True,
        use_tsa_block=True, # Enable new block
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
    
    print("\nClean model ready for training!")