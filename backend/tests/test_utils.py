import pytest
import tempfile
import os
import cv2
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path 
sys.path.append(str(Path(__file__).parent.parent))

from main import (
    secure_filename,
    allowed_file,
    get_video_metadata,
    generate_thumbnail,
    save_violence_clip,
    process_and_save_events
)


class TestSecureFilename:
    """Test the secure_filename function"""
    
    def test_normal_filename(self):
        """Test with normal filename"""
        result = secure_filename("video.mp4")
        assert result == "video.mp4"
    
    def test_filename_with_path(self):
        """Test filename with path components"""
        result = secure_filename("/path/to/video.mp4")
        assert result == "video.mp4"
        
        result = secure_filename("..\\..\\video.mp4")
        assert result == "video.mp4"
    
    def test_filename_with_dangerous_chars(self):
        """Test filename with dangerous characters"""
        result = secure_filename("video<script>.mp4")
        assert "<script>" not in result
        assert "video" in result
        assert ".mp4" in result
    
    def test_filename_with_spaces_and_special_chars(self):
        """Test filename with spaces and special characters"""
        result = secure_filename("my video file!@#$%^&*().mp4")
        assert " " not in result or "_" in result
        assert "my" in result
        assert ".mp4" in result
    
    def test_empty_filename(self):
        """Test with empty filename"""
        result = secure_filename("")
        assert result == "unnamed_file"
        
        result = secure_filename("...")
        assert result == "unnamed_file"


class TestAllowedFile:
    """Test the allowed_file function"""
    
    def test_allowed_extensions(self):
        """Test allowed video extensions"""
        assert allowed_file("video.mp4") == True
        assert allowed_file("movie.avi") == True
        assert allowed_file("clip.mov") == True
        assert allowed_file("film.mkv") == True
    
    def test_case_insensitive(self):
        """Test case insensitive extension checking"""
        assert allowed_file("video.MP4") == True
        assert allowed_file("movie.AVI") == True
        assert allowed_file("clip.Mov") == True
    
    def test_disallowed_extensions(self):
        """Test disallowed extensions"""
        assert allowed_file("document.txt") == False
        assert allowed_file("image.jpg") == False
        assert allowed_file("audio.mp3") == False
        assert allowed_file("archive.zip") == False
    
    def test_no_extension(self):
        """Test files without extension"""
        assert allowed_file("video") == False
        assert allowed_file("") == False
    
    def test_multiple_dots(self):
        """Test files with multiple dots"""
        assert allowed_file("my.video.file.mp4") == True
        assert allowed_file("my.video.file.txt") == False


class TestVideoMetadata:
    """Test video metadata extraction"""
    
    @patch('cv2.VideoCapture')
    def test_get_video_metadata_success(self, mock_video_capture):
        """Test successful metadata extraction"""
        # Mock successful video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 900
        }.get(prop, 0)
        mock_cap.release.return_value = None
        mock_video_capture.return_value = mock_cap
        
        metadata = get_video_metadata("test_video.mp4")
        
        assert metadata is not None
        assert metadata['width'] == 1920
        assert metadata['height'] == 1080
        assert metadata['fps'] == 30.0
        assert metadata['frame_count'] == 900
        assert metadata['duration'] == 30.0  # 900 frames / 30 fps
        assert metadata['duration_formatted'] == "0:30"
    
    @patch('cv2.VideoCapture')
    def test_get_video_metadata_failure(self, mock_video_capture):
        """Test metadata extraction failure"""
        # Mock failed video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap
        
        metadata = get_video_metadata("invalid_video.mp4")
        
        assert metadata is None
    
    @patch('cv2.VideoCapture')
    def test_get_video_metadata_zero_fps(self, mock_video_capture):
        """Test metadata with zero FPS"""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 0.0,  # Zero FPS
            cv2.CAP_PROP_FRAME_COUNT: 100
        }.get(prop, 0)
        mock_cap.release.return_value = None
        mock_video_capture.return_value = mock_cap
        
        metadata = get_video_metadata("zero_fps_video.mp4")
        
        assert metadata is not None
        assert metadata['duration'] == 0  # Should handle zero FPS gracefully
    
    @patch('cv2.VideoCapture')
    def test_get_video_metadata_long_duration(self, mock_video_capture):
        """Test metadata with long video duration"""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 1280,
            cv2.CAP_PROP_FRAME_HEIGHT: 720,
            cv2.CAP_PROP_FPS: 25.0,
            cv2.CAP_PROP_FRAME_COUNT: 9000  # 6 minutes
        }.get(prop, 0)
        mock_cap.release.return_value = None
        mock_video_capture.return_value = mock_cap
        
        metadata = get_video_metadata("long_video.mp4")
        
        assert metadata is not None
        assert metadata['duration'] == 360.0  # 6 minutes
        assert metadata['duration_formatted'] == "6:00"


class TestThumbnailGeneration:
    """Test thumbnail generation"""
    
    @patch('cv2.VideoCapture')
    @patch('cv2.imwrite')
    @patch('cv2.resize')
    def test_generate_thumbnail_success(self, mock_resize, mock_imwrite, mock_video_capture):
        """Test successful thumbnail generation"""
        # Mock video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 100  # frame count
        mock_cap.set.return_value = None
        
        # Mock frame reading
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)
        mock_cap.release.return_value = None
        mock_video_capture.return_value = mock_cap
        
        # Mock resize and imwrite
        resized_frame = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        mock_resize.return_value = resized_frame
        mock_imwrite.return_value = True
        
        result = generate_thumbnail("test_video.mp4", "thumbnail.jpg")
        
        assert result == True
        mock_imwrite.assert_called_once()
        mock_resize.assert_called_once()
    
    @patch('cv2.VideoCapture')
    def test_generate_thumbnail_video_open_failure(self, mock_video_capture):
        """Test thumbnail generation with video open failure"""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap
        
        result = generate_thumbnail("invalid_video.mp4", "thumbnail.jpg")
        
        assert result == False
    
    @patch('cv2.VideoCapture')
    def test_generate_thumbnail_frame_read_failure(self, mock_video_capture):
        """Test thumbnail generation with frame read failure"""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 100
        mock_cap.set.return_value = None
        mock_cap.read.return_value = (False, None)  # Failed to read frame
        mock_cap.release.return_value = None
        mock_video_capture.return_value = mock_cap
        
        result = generate_thumbnail("corrupt_video.mp4", "thumbnail.jpg")
        
        assert result == False
    
    @patch('cv2.VideoCapture')
    @patch('cv2.imwrite')
    @patch('cv2.resize')
    def test_generate_thumbnail_with_frame_number(self, mock_resize, mock_imwrite, mock_video_capture):
        """Test thumbnail generation with specific frame number"""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.set.return_value = None
        
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)
        mock_cap.release.return_value = None
        mock_video_capture.return_value = mock_cap
        
        resized_frame = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        mock_resize.return_value = resized_frame
        mock_imwrite.return_value = True
        
        result = generate_thumbnail("test_video.mp4", "thumbnail.jpg", frame_number=50)
        
        assert result == True
        # Verify that specific frame was set
        mock_cap.set.assert_called_with(cv2.CAP_PROP_POS_FRAMES, 50)


class TestSaveViolenceClip:
    """Test violence clip saving"""
    
    @patch('cv2.VideoCapture')
    @patch('cv2.VideoWriter')
    def test_save_violence_clip_success(self, mock_video_writer, mock_video_capture):
        """Test successful clip saving"""
        # Mock video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480
        }.get(prop, 0)
        mock_cap.set.return_value = None
        
        # Mock frame reading
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)
        mock_cap.release.return_value = None
        mock_video_capture.return_value = mock_cap
        
        # Mock video writer
        mock_writer = Mock()
        mock_writer.write.return_value = None
        mock_writer.release.return_value = None
        mock_video_writer.return_value = mock_writer
        
        result = save_violence_clip("input_video.mp4", 10.0, 15.0, "output_clip.mp4")
        
        assert result == True
        mock_writer.write.assert_called()
        mock_writer.release.assert_called_once()
    
    @patch('cv2.VideoCapture')
    def test_save_violence_clip_video_open_failure(self, mock_video_capture):
        """Test clip saving with video open failure"""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap
        
        result = save_violence_clip("invalid_video.mp4", 10.0, 15.0, "output_clip.mp4")
        
        assert result == False
    
    @patch('cv2.VideoCapture')
    @patch('cv2.VideoWriter')
    def test_save_violence_clip_exception_handling(self, mock_video_writer, mock_video_capture):
        """Test clip saving with exception"""
        # Mock video capture to raise exception
        mock_video_capture.side_effect = Exception("Capture error")
        
        result = save_violence_clip("video.mp4", 10.0, 15.0, "output_clip.mp4")
        
        assert result == False


class TestProcessAndSaveEvents:
    """Test event processing and saving"""
    
    @patch('main.event_db')
    def test_process_and_save_events_success(self, mock_event_db):
        """Test successful event processing"""
        job_id = "test-job-123"
        result = {
            'timestamp': '2024-01-01 12:00:00',
            'filename': 'test_video.mp4',
            'thumbnail': '/test/thumb.jpg',
            'segments': [
                {
                    'start': 10.0,
                    'end': 15.0,
                    'confidence': 0.85
                }
            ],
            'overall_result': {'confidence': 0.85},
            'model_info': {'architecture': 'X3D-M'},
            'metadata': {'duration': 30.0}
        }
        
        mock_event_db.update_daily_processed_count.return_value = None
        mock_event_db.save_event.return_value = 1
        
        # Should not raise exception
        with patch('main.save_violence_clip', return_value=True):
            process_and_save_events(job_id, result, "test_video.mp4")
        
        # Verify database calls
        mock_event_db.update_daily_processed_count.assert_called_once_with(1)
        mock_event_db.save_event.assert_called_once()
    
    @patch('main.event_db')
    def test_process_and_save_events_no_segments(self, mock_event_db):
        """Test event processing with no violence segments"""
        job_id = "test-job-456"
        result = {
            'timestamp': '2024-01-01 12:00:00',
            'filename': 'safe_video.mp4',
            'segments': [],  # No violence detected
            'overall_result': {'confidence': 0.2},
            'model_info': {'architecture': 'X3D-M'},
            'metadata': {'duration': 30.0}
        }
        
        mock_event_db.update_daily_processed_count.return_value = None
        
        process_and_save_events(job_id, result, "safe_video.mp4")
        
        # Should still update processed count
        mock_event_db.update_daily_processed_count.assert_called_once_with(1)
        # Should not save violence events
        mock_event_db.save_event.assert_not_called()
    
    @patch('main.event_db', None)
    def test_process_and_save_events_no_database(self):
        """Test event processing when database is not initialized"""
        job_id = "test-job-789"
        result = {
            'timestamp': '2024-01-01 12:00:00',
            'filename': 'test_video.mp4',
            'segments': []
        }
        
        # Should handle gracefully when database is None
        process_and_save_events(job_id, result, "test_video.mp4")
        # Should not raise exception
    
    @patch('main.event_db')
    def test_process_and_save_events_exception_handling(self, mock_event_db):
        """Test event processing with exception"""
        job_id = "test-job-error"
        result = {
            'timestamp': '2024-01-01 12:00:00',
            'filename': 'test_video.mp4',
            'segments': []
        }
        
        # Mock database to raise exception
        mock_event_db.update_daily_processed_count.side_effect = Exception("Database error")
        
        # Should handle exception gracefully
        process_and_save_events(job_id, result, "test_video.mp4")
        # Should not crash


class TestUtilityIntegration:
    """Integration tests for utility functions"""
    
    def test_filename_security_and_validation(self):
        """Test combined filename security and validation"""
        dangerous_filename = "../../../etc/passwd.mp4"
        
        # Should be made safe
        safe_name = secure_filename(dangerous_filename)
        assert ".." not in safe_name
        assert "/" not in safe_name
        
        # Should still be allowed (has .mp4 extension)
        assert allowed_file(safe_name) == True
    
    def test_filename_pipeline(self):
        """Test typical filename processing pipeline"""
        original = "My Vacation Video (2024)!.MP4"
        
        # Make secure
        safe = secure_filename(original)
        
        # Check if allowed
        is_allowed = allowed_file(safe)
        
        assert is_allowed == True
        assert "vacation" in safe.lower() or "my" in safe.lower()
        assert ".mp4" in safe.lower()
    
    @patch('cv2.VideoCapture')
    def test_video_processing_pipeline(self, mock_video_capture):
        """Test combined video metadata and thumbnail generation"""
        # Mock video capture for both functions
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 900
        }.get(prop, 0)
        mock_cap.set.return_value = None
        
        dummy_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)
        mock_cap.release.return_value = None
        mock_video_capture.return_value = mock_cap
        
        # Get metadata
        metadata = get_video_metadata("test_video.mp4")
        
        assert metadata is not None
        assert metadata['width'] == 1920
        assert metadata['height'] == 1080
        
        # Generate thumbnail
        with patch('cv2.imwrite', return_value=True), \
             patch('cv2.resize', return_value=dummy_frame):
            
            thumbnail_success = generate_thumbnail("test_video.mp4", "thumb.jpg")
            
            assert thumbnail_success == True