import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path 
sys.path.append(str(Path(__file__).parent.parent))

from model import (
    AttentionFusion, 
    MotionEnhancementModule, 
    X3DViolenceDetector,
    StableCrossEntropyLoss,
    create_model
)


class TestAttentionFusion:
    """Test the AttentionFusion module"""
    
    def test_initialization(self):
        """Test that AttentionFusion initializes correctly"""
        x3d_dim = 192
        motion_dim = 128
        
        fusion = AttentionFusion(x3d_dim, motion_dim)
        
        assert fusion.query_transform.in_features == motion_dim
        assert fusion.query_transform.out_features == x3d_dim
        assert fusion.key_transform.in_features == x3d_dim
        assert fusion.value_transform.in_features == x3d_dim
        assert fusion.scale == x3d_dim ** -0.5
    
    def test_forward_pass(self):
        """Test forward pass with dummy data"""
        x3d_dim = 192
        motion_dim = 128
        batch_size = 2
        
        fusion = AttentionFusion(x3d_dim, motion_dim)
        
        # Create dummy inputs
        x3d_features = torch.randn(batch_size, x3d_dim)
        motion_features = torch.randn(batch_size, motion_dim)
        
        # Forward pass
        output = fusion(x3d_features, motion_features)
        
        # Check output shape
        expected_output_dim = x3d_dim + motion_dim + x3d_dim  # 192 + 128 + 192 = 512
        assert output.shape == (batch_size, expected_output_dim)


class TestMotionEnhancementModule:
    """Test the MotionEnhancementModule"""
    
    def test_initialization(self):
        """Test module initialization"""
        input_dim = 3
        hidden_dim = 256
        output_dim = 128
        
        module = MotionEnhancementModule(input_dim, hidden_dim, output_dim)
        
        assert isinstance(module.flow_conv, nn.Sequential)
        assert isinstance(module.motion_fc, nn.Sequential)
    
    def test_forward_pass(self):
        """Test forward pass with dummy optical flow"""
        batch_size = 2
        channels = 3
        frames = 16
        height = 224
        width = 224
        
        module = MotionEnhancementModule(input_dim=channels)
        
        # Create dummy optical flow data
        optical_flow = torch.randn(batch_size, channels, frames, height, width)
        
        # Forward pass
        output = module(optical_flow)
        
        # Check output shape
        assert output.shape == (batch_size, 128)  # Default output_dim


class TestX3DViolenceDetector:
    """Test the main X3DViolenceDetector model"""
    
    @patch('torch.hub.load')
    def test_model_initialization(self, mock_torch_hub):
        """Test model initialization with mocked X3D backbone"""
        # Mock the X3D model
        mock_x3d = Mock()
        mock_x3d.blocks = [Mock(), Mock(), Mock()]  # Simulate X3D blocks
        mock_torch_hub.return_value = mock_x3d
        
        with patch.object(X3DViolenceDetector, '_get_feature_dim', return_value=192):
            model = X3DViolenceDetector(
                x3d_model_name="x3d_m",
                num_classes=2,
                use_motion_enhancement=True,
                device="cpu"
            )
            
            assert model.use_motion_enhancement == True
            assert model.num_classes == 2
            assert hasattr(model, 'motion_module')
            assert hasattr(model, 'attention_fusion')
            assert hasattr(model, 'classifier')
    
    @patch('torch.hub.load')
    def test_model_without_motion(self, mock_torch_hub):
        """Test model initialization without motion enhancement"""
        mock_x3d = Mock()
        mock_torch_hub.return_value = mock_x3d
        
        with patch.object(X3DViolenceDetector, '_get_feature_dim', return_value=192):
            model = X3DViolenceDetector(
                x3d_model_name="x3d_m",
                num_classes=2,
                use_motion_enhancement=False,
                device="cpu"
            )
            
            assert model.use_motion_enhancement == False
            assert not hasattr(model, 'motion_module')
            assert not hasattr(model, 'attention_fusion')
    
    @patch('torch.hub.load')
    def test_forward_pass_with_motion(self, mock_torch_hub):
        """Test forward pass with motion enhancement"""
        # Setup mocks
        mock_x3d = Mock()
        mock_torch_hub.return_value = mock_x3d
        
        with patch.object(X3DViolenceDetector, '_get_feature_dim', return_value=192):
            model = X3DViolenceDetector(
                x3d_model_name="x3d_m",
                use_motion_enhancement=True,
                device="cpu"
            )
            
            # Mock the feature extraction method
            model._extract_x3d_features = Mock(return_value=torch.randn(1, 192))
            
            # Create dummy input data
            data = {
                'rgb': torch.randn(1, 3, 16, 224, 224),
                'flow': torch.randn(1, 3, 16, 224, 224)
            }
            
            # Forward pass
            output = model(data)
            
            # Check output shape (should be [batch_size, num_classes])
            assert output.shape == (1, 2)
    
    @patch('torch.hub.load')
    def test_forward_pass_without_motion(self, mock_torch_hub):
        """Test forward pass without motion enhancement"""
        mock_x3d = Mock()
        mock_torch_hub.return_value = mock_x3d
        
        with patch.object(X3DViolenceDetector, '_get_feature_dim', return_value=192):
            model = X3DViolenceDetector(
                x3d_model_name="x3d_m",
                use_motion_enhancement=False,
                device="cpu"
            )
            
            # Mock the feature extraction method
            model._extract_x3d_features = Mock(return_value=torch.randn(1, 192))
            
            # Create dummy input data (only RGB)
            data = {'rgb': torch.randn(1, 3, 16, 224, 224)}
            
            # Forward pass
            output = model(data)
            
            # Check output shape
            assert output.shape == (1, 2)


class TestStableCrossEntropyLoss:
    """Test the StableCrossEntropyLoss"""
    
    def test_initialization(self):
        """Test loss function initialization"""
        loss_fn = StableCrossEntropyLoss(label_smoothing=0.1)
        assert loss_fn.label_smoothing == 0.1
    
    def test_forward_pass(self):
        """Test loss computation"""
        loss_fn = StableCrossEntropyLoss()
        
        # Create dummy predictions and targets
        predictions = torch.randn(4, 2)  # 4 samples, 2 classes
        targets = torch.tensor([0, 1, 0, 1])
        
        # Compute loss
        loss = loss_fn(predictions, targets)
        
        # Check that loss is a scalar tensor
        assert loss.dim() == 0
        assert loss.item() >= 0  # Loss should be non-negative


class TestCreateModel:
    """Test the create_model function"""
    
    @patch('torch.hub.load')
    def test_create_model_default(self, mock_torch_hub):
        """Test creating model with default parameters"""
        mock_x3d = Mock()
        mock_torch_hub.return_value = mock_x3d
        
        with patch.object(X3DViolenceDetector, '_get_feature_dim', return_value=192):
            model = create_model(device="cpu")
            
            assert isinstance(model, X3DViolenceDetector)
            assert model.use_motion_enhancement == True
            assert model.num_classes == 2
    
    @patch('torch.hub.load')
    def test_create_model_custom(self, mock_torch_hub):
        """Test creating model with custom parameters"""
        mock_x3d = Mock()
        mock_torch_hub.return_value = mock_x3d
        
        with patch.object(X3DViolenceDetector, '_get_feature_dim', return_value=192):
            model = create_model(
                model_name="x3d_s",
                num_classes=3,
                use_motion_enhancement=False,
                device="cpu"
            )
            
            assert isinstance(model, X3DViolenceDetector)
            assert model.use_motion_enhancement == False
            assert model.num_classes == 3


class TestModelComponents:
    """Test model component interactions"""
    
    def test_attention_fusion_with_different_dims(self):
        """Test attention fusion with various dimensions"""
        test_cases = [
            (64, 32),
            (128, 64),
            (256, 128)
        ]
        
        for x3d_dim, motion_dim in test_cases:
            fusion = AttentionFusion(x3d_dim, motion_dim)
            
            x3d_features = torch.randn(2, x3d_dim)
            motion_features = torch.randn(2, motion_dim)
            
            output = fusion(x3d_features, motion_features)
            expected_dim = x3d_dim + motion_dim + x3d_dim
            
            assert output.shape == (2, expected_dim)
    
    def test_motion_module_with_different_inputs(self):
        """Test motion module with different input sizes"""
        module = MotionEnhancementModule(input_dim=3)
        
        # Test with different batch sizes
        for batch_size in [1, 2, 4]:
            optical_flow = torch.randn(batch_size, 3, 16, 224, 224)
            output = module(optical_flow)
            assert output.shape == (batch_size, 128)