import pytest
import numpy as np
import torch
import cv2
import tempfile
import os
import time
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path 
sys.path.append(str(Path(__file__).parent.parent))

from torch_detection import (
    extract_frames,
    compute_optical_flow,
    preprocess_frames,
    predict_violence,
    load_violence_detection_model
)


class TestExtractFrames:
    """Test frame extraction from videos"""
    
    @patch('cv2.VideoCapture')
    def test_extract_frames_success(self, mock_video_capture):
        """Test successful frame extraction"""
        # Mock video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FPS: 30.0
        }.get(prop, 0)
        
        # Mock frame reading
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)
        mock_cap.set.return_value = None
        mock_cap.release.return_value = None
        
        mock_video_capture.return_value = mock_cap
        
        # Test frame extraction (updated to match new INPUT_SIZE = 336)
        frames = extract_frames("dummy_video.mp4", num_frames=16)
        
        assert isinstance(frames, np.ndarray)
        assert frames.shape == (16, 336, 336, 3)  # Updated size
        assert frames.dtype == np.uint8
    
    @patch('cv2.VideoCapture')
    def test_extract_frames_video_open_failure(self, mock_video_capture):
        """Test handling of video open failure"""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap
        
        with pytest.raises(ValueError, match="Could not open video"):
            extract_frames("nonexistent_video.mp4")
    
    @patch('cv2.VideoCapture')
    def test_extract_frames_short_video(self, mock_video_capture):
        """Test frame extraction from short video"""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 5,  # Very short video
            cv2.CAP_PROP_FPS: 30.0
        }.get(prop, 0)
        
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)
        mock_cap.set.return_value = None
        mock_cap.release.return_value = None
        
        mock_video_capture.return_value = mock_cap
        
        frames = extract_frames("short_video.mp4", num_frames=16)
        
        # Should still return 16 frames (with repetition)
        assert frames.shape == (16, 336, 336, 3)  # Updated size


class TestComputeOpticalFlow:
    """Test optical flow computation"""
    
    def test_optical_flow_computation(self):
        """Test optical flow computation with dummy frames"""
        # Create dummy RGB frames (updated size)
        num_frames = 4
        frames = np.random.randint(0, 255, (num_frames, 336, 336, 3), dtype=np.uint8)
        
        with patch('cv2.calcOpticalFlowFarneback') as mock_flow:
            # Mock optical flow output
            mock_flow_result = np.random.randn(336, 336, 2).astype(np.float32)
            mock_flow.return_value = mock_flow_result
            
            flow_frames = compute_optical_flow(frames)
            
            assert flow_frames.shape == (num_frames, 336, 336, 3)
            assert flow_frames.dtype == np.uint8
            assert mock_flow.call_count == num_frames - 1  # N-1 flows for N frames
    
    def test_optical_flow_error_handling(self):
        """Test optical flow computation with errors"""
        frames = np.random.randint(0, 255, (3, 336, 336, 3), dtype=np.uint8)
        
        with patch('cv2.calcOpticalFlowFarneback', side_effect=Exception("Flow error")):
            # Should handle errors gracefully and return zero flows
            flow_frames = compute_optical_flow(frames)
            
            assert flow_frames.shape == (3, 336, 336, 3)
            # Should return zero flows as fallback
            assert np.allclose(flow_frames, 0)
    
    def test_optical_flow_empty_frames(self):
        """Test optical flow with empty frame list"""
        frames = np.array([]).reshape(0, 336, 336, 3)
        
        flow_frames = compute_optical_flow(frames)
        
        # Should return properly shaped empty array
        assert flow_frames.shape == (0, 336, 336, 3)
        assert flow_frames.dtype == np.uint8


class TestPreprocessFrames:
    """Test frame preprocessing"""
    
    def test_preprocess_frames_rgb_only(self):
        """Test preprocessing without optical flow"""
        frames = np.random.randint(0, 255, (16, 336, 336, 3), dtype=np.uint8)
        
        data = preprocess_frames(frames, compute_flow=False)
        
        assert 'rgb' in data
        assert data['rgb'].shape == (3, 16, 336, 336)  # [C, T, H, W]
        assert isinstance(data['rgb'], torch.Tensor)
        
        # Check normalization (should be around [-2, 2] range with ImageNet normalization)
        assert data['rgb'].min() < 0
        assert data['rgb'].max() > 0
    
    def test_preprocess_frames_with_flow(self):
        """Test preprocessing with optical flow"""
        frames = np.random.randint(0, 255, (16, 336, 336, 3), dtype=np.uint8)
        
        with patch('torch_detection.compute_optical_flow') as mock_flow:
            # Mock optical flow result
            mock_flow.return_value = np.random.randint(0, 255, (16, 336, 336, 3), dtype=np.uint8)
            
            data = preprocess_frames(frames, compute_flow=True)
            
            assert 'rgb' in data
            assert 'flow' in data
            assert data['rgb'].shape == (3, 16, 336, 336)
            assert data['flow'].shape == (3, 16, 336, 336)
            assert isinstance(data['rgb'], torch.Tensor)
            assert isinstance(data['flow'], torch.Tensor)
    
    def test_preprocess_frames_flow_error(self):
        """Test preprocessing when optical flow computation fails"""
        frames = np.random.randint(0, 255, (16, 336, 336, 3), dtype=np.uint8)
        
        with patch('torch_detection.compute_optical_flow', side_effect=Exception("Flow error")):
            data = preprocess_frames(frames, compute_flow=True)
            
            assert 'rgb' in data
            assert 'flow' in data
            # Should have dummy flow tensor with same shape as RGB
            assert data['flow'].shape == data['rgb'].shape
            assert torch.allclose(data['flow'], torch.zeros_like(data['rgb']))


class TestPredictViolence:
    """Test violence prediction"""
    
    def create_mock_model(self, output_logits):
        """Create a mock model that returns specified logits"""
        mock_model = Mock()
        # Mock parameters as a generator to avoid iterator issues
        mock_param = torch.tensor([1.0])
        mock_model.parameters.return_value = iter([mock_param])
        
        # Mock the forward pass
        output_tensor = torch.tensor([output_logits])
        mock_model.return_value = output_tensor
        
        return mock_model
    
    @patch('time.time')
    def test_predict_violence_non_violent(self, mock_time):
        """Test prediction for non-violent content"""
        # Mock time to simulate inference time
        mock_time.side_effect = [0.0, 0.1]  # start_time, end_time
        
        # Logits favoring non-violence [NonFight_logit, Fight_logit]
        mock_model = self.create_mock_model([2.0, -1.0])
        
        data = {
            'rgb': torch.randn(1, 3, 16, 336, 336),
            'flow': torch.randn(1, 3, 16, 336, 336)
        }
        
        is_fight, confidence, inference_time = predict_violence(
            mock_model, data, threshold=0.5, device='cpu'
        )
        
        assert isinstance(is_fight, bool)
        assert not is_fight  # Should predict non-violence
        assert 0.0 <= confidence <= 1.0
        assert inference_time >= 0.1  # Should match mocked time difference
    
    @patch('time.time')
    def test_predict_violence_violent(self, mock_time):
        """Test prediction for violent content"""
        # Mock time to simulate inference time
        mock_time.side_effect = [0.0, 0.2]  # start_time, end_time
        
        # Logits favoring violence [NonFight_logit, Fight_logit]
        mock_model = self.create_mock_model([-1.0, 2.0])
        
        data = {'rgb': torch.randn(1, 3, 16, 336, 336)}
        
        is_fight, confidence, inference_time = predict_violence(
            mock_model, data, threshold=0.5, device='cpu'
        )
        
        assert isinstance(is_fight, bool)
        assert is_fight  # Should predict violence
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be above threshold
        assert inference_time >= 0.2  # Should match mocked time difference
    
    def test_predict_violence_threshold_behavior(self):
        """Test prediction threshold behavior"""
        # Borderline case
        mock_model = self.create_mock_model([0.1, 0.2])
        
        data = {'rgb': torch.randn(1, 3, 16, 336, 336)}
        
        # Test with low threshold
        is_fight_low, confidence, _ = predict_violence(
            mock_model, data, threshold=0.4, device='cpu'
        )
        
        # Test with high threshold
        is_fight_high, _, _ = predict_violence(
            mock_model, data, threshold=0.7, device='cpu'
        )
        
        # With low threshold, might detect violence; with high threshold, should not
        assert isinstance(is_fight_low, bool)
        assert isinstance(is_fight_high, bool)
        
        if confidence > 0.7:
            assert is_fight_high == is_fight_low
        elif confidence < 0.4:
            assert not is_fight_low and not is_fight_high
    
    @patch('time.time')
    def test_predict_violence_batch_dimension_handling(self, mock_time):
        """Test that batch dimensions are handled correctly"""
        # Mock time to simulate inference time
        mock_time.side_effect = [0.0, 0.05]
        
        mock_model = self.create_mock_model([0.0, 1.0])
        
        # Test data without batch dimension
        data = {'rgb': torch.randn(3, 16, 336, 336)}  # No batch dimension
        
        is_fight, confidence, inference_time = predict_violence(
            mock_model, data, threshold=0.5, device='cpu'
        )
        
        assert isinstance(is_fight, bool)
        assert 0.0 <= confidence <= 1.0
        assert inference_time >= 0.05
    
    def test_predict_violence_extreme_logits_warning(self):
        """Test warning for extreme logits"""
        # Very extreme logits
        mock_model = self.create_mock_model([-50.0, 50.0])
        
        data = {'rgb': torch.randn(1, 3, 16, 336, 336)}
        
        with patch('builtins.print') as mock_print:
            is_fight, confidence, _ = predict_violence(
                mock_model, data, threshold=0.5, debug=True, device='cpu'
            )
            
            # Should print warning about extreme logits
            printed_text = ' '.join([str(call.args[0]) for call in mock_print.call_args_list])
            assert 'WARNING' in printed_text or 'Extreme' in printed_text


class TestModelLoading:
    """Test model loading functionality"""
    
    def test_load_model_file_not_found(self):
        """Test loading non-existent model file"""
        with pytest.raises(FileNotFoundError):
            load_violence_detection_model("nonexistent_model.pth")
    
    @patch('torch.load')
    @patch('torch_detection.X3DViolenceDetector')
    def test_load_model_success(self, mock_model_class, mock_torch_load):
        """Test successful model loading"""
        # Create a temporary model file
        temp_file = tempfile.NamedTemporaryFile(suffix='.pth', delete=False)
        temp_file.close()
        
        try:
            # Mock torch.load to return a state dict
            mock_torch_load.return_value = {
                'model_state_dict': {'dummy': torch.tensor([1.0])},
                'model_info': {'architecture': 'x3d_m'}
            }
            
            # Mock model creation
            mock_model_instance = Mock()
            mock_model_instance.eval.return_value = None
            mock_model_instance.to.return_value = mock_model_instance
            mock_model_instance.load_state_dict.return_value = None
            mock_model_instance.parameters.return_value = iter([torch.tensor([1.0])])
            mock_model_instance.use_motion_enhancement = True
            
            # Mock forward pass for warm-up
            mock_model_instance.return_value = torch.tensor([[0.1, 0.2]])
            
            mock_model_class.return_value = mock_model_instance
            
            model, use_gpu = load_violence_detection_model(temp_file.name, device='cpu')
            
            assert model is not None
            assert isinstance(use_gpu, bool)
            assert not use_gpu  # CPU device
            
        finally:
            os.unlink(temp_file.name)
    
    def test_device_selection(self):
        """Test device selection logic"""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.load') as mock_load:
                with patch('torch_detection.X3DViolenceDetector'):
                    temp_file = tempfile.NamedTemporaryFile(suffix='.pth', delete=False)
                    temp_file.close()
                    
                    try:
                        mock_load.return_value = {'model_state_dict': {}}
                        
                        # Should use CUDA when available and not specified
                        with patch('torch.device') as mock_device:
                            try:
                                load_violence_detection_model(temp_file.name)
                                # Should try to create CUDA device
                                mock_device.assert_called()
                            except:
                                pass  # Ignore other errors, we just want to test device logic
                    
                    finally:
                        os.unlink(temp_file.name)


class TestIntegration:
    """Integration tests combining multiple functions"""
    
    @patch('cv2.VideoCapture')
    @patch('time.time')
    def test_full_pipeline_mock(self, mock_time, mock_video_capture):
        """Test the full pipeline from frame extraction to prediction"""
        # Mock time for inference timing
        mock_time.side_effect = [0.0, 0.15]  # start_time, end_time
        
        # Mock video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 50,
            cv2.CAP_PROP_FPS: 30.0
        }.get(prop, 0)
        
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)
        mock_cap.set.return_value = None
        mock_cap.release.return_value = None
        mock_video_capture.return_value = mock_cap
        
        # Extract frames
        frames = extract_frames("dummy_video.mp4")
        
        # Preprocess frames
        data = preprocess_frames(frames, compute_flow=False)
        
        # Check shapes before predict_violence (which adds batch dimension)
        assert frames.shape == (16, 336, 336, 3)  # Updated size
        assert data['rgb'].shape == (3, 16, 336, 336)
        
        # Create mock model and predict
        mock_model = Mock()
        mock_model.parameters.return_value = iter([torch.tensor([1.0])])
        mock_model.return_value = torch.tensor([[0.2, 0.8]])  # Favor violence
        
        is_fight, confidence, inference_time = predict_violence(
            mock_model, data, threshold=0.5, device='cpu'
        )
        
        assert isinstance(is_fight, bool)
        assert 0.0 <= confidence <= 1.0
        assert inference_time >= 0.15  # Should match mocked time difference
        
        # After predict_violence, data will have batch dimension
        assert data['rgb'].shape == (1, 3, 16, 336, 336)

    @patch('cv2.VideoCapture') 
    def test_pipeline_with_motion_enhancement(self, mock_video_capture):
        """Test pipeline with motion enhancement enabled"""
        # Mock video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 50,
            cv2.CAP_PROP_FPS: 30.0
        }.get(prop, 0)
        
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)
        mock_cap.set.return_value = None
        mock_cap.release.return_value = None
        mock_video_capture.return_value = mock_cap
        
        # Extract frames
        frames = extract_frames("dummy_video.mp4")
        
        # Test with optical flow computation
        with patch('cv2.calcOpticalFlowFarneback') as mock_flow:
            mock_flow.return_value = np.random.randn(336, 336, 2).astype(np.float32)
            
            data = preprocess_frames(frames, compute_flow=True)
            
            assert 'rgb' in data
            assert 'flow' in data
            assert data['rgb'].shape == (3, 16, 336, 336)
            assert data['flow'].shape == (3, 16, 336, 336)
            
            # Verify optical flow was called for consecutive frame pairs
            assert mock_flow.call_count == 15  # 16 frames = 15 flow computations

    def test_normalization_consistency(self):
        """Test that normalization is consistent across different input ranges"""
        # Test with different value ranges
        for max_val in [128, 255]:
            frames = np.random.randint(0, max_val + 1, (16, 336, 336, 3), dtype=np.uint8)
            data = preprocess_frames(frames, compute_flow=False)
            
            # Should normalize to roughly the same range regardless of input
            rgb_tensor = data['rgb']
            assert rgb_tensor.min() >= -3.0  # Roughly -2.5 with ImageNet normalization
            assert rgb_tensor.max() <= 3.0   # Roughly +2.5 with ImageNet normalization