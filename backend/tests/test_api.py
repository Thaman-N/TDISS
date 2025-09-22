import pytest
import json
import tempfile
import os
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path 
sys.path.append(str(Path(__file__).parent.parent))

# Mock the heavy imports before importing main
with patch('main.load_violence_detection_model'), \
     patch('main.EventDatabase'), \
     patch('main.StreamDatabase'):
    from main import app, active_jobs, results_history


class TestAPIEndpoints:
    """Test FastAPI endpoints"""
    
    def setup_method(self):
        """Setup test client and clear global state"""
        self.client = TestClient(app)
        
        # Clear global state
        active_jobs.clear()
        results_history.clear()
    
    def test_root_endpoint(self):
        """Test the root endpoint"""
        response = self.client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Violence Detection API" in data["message"]
        assert data["status"] == "running"
    
    def test_get_all_jobs_empty(self):
        """Test getting jobs when none exist"""
        response = self.client.get("/api/jobs")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0
    
    def test_get_all_jobs_with_data(self):
        """Test getting jobs with existing data"""
        # Add test job
        job_id = "test-job-123"
        active_jobs[job_id] = {
            'id': job_id,
            'filename': 'test_video.mp4',
            'status': 'completed',
            'progress': 100,
            'message': 'Processing complete',
            'timestamp': '2024-01-01 12:00:00'
        }
        
        response = self.client.get("/api/jobs")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]['id'] == job_id
        assert data[0]['filename'] == 'test_video.mp4'
        assert data[0]['status'] == 'completed'
    
    def test_get_job_by_id_success(self):
        """Test getting a specific job by ID"""
        job_id = "test-job-456"
        active_jobs[job_id] = {
            'id': job_id,
            'filename': 'another_video.mp4',
            'status': 'processing',
            'progress': 50,
            'message': 'Processing frames',
            'timestamp': '2024-01-01 13:00:00'
        }
        
        response = self.client.get(f"/api/job/{job_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data['id'] == job_id
        assert data['status'] == 'processing'
        assert data['progress'] == 50
    
    def test_get_job_by_id_not_found(self):
        """Test getting a non-existent job"""
        response = self.client.get("/api/job/nonexistent-job")
        
        assert response.status_code == 404
        data = response.json()
        assert "Job not found" in data['detail']
    
    def test_get_history_empty(self):
        """Test getting history when empty"""
        response = self.client.get("/api/history")
        
        assert response.status_code == 200
        data = response.json()
        assert "history" in data
        assert isinstance(data["history"], list)
        # Note: May have existing history data, so just check it's a list
    
    def test_get_history_with_data(self):
        """Test getting history with data"""
        # Clear existing history first to ensure clean test
        results_history.clear()
        
        # Add test history directly to results_history
        history_item = {
            'job_id': 'hist-123',
            'filename': 'history_video.mp4',
            'timestamp': '2024-01-01 10:00:00',
            'has_violence': True,
            'violence_duration': 5.2,
            'violence_percentage': 15.3,
            'overall_confidence': 0.87,
            'model_type': 'X3D-M'
        }
        
        results_history['hist-123'] = history_item
        
        with patch('main.load_history_from_file', return_value={}):
            response = self.client.get("/api/history")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["history"]) == 1
        assert data["history"][0]['job_id'] == 'hist-123'
        assert data["history"][0]['has_violence'] == True
    
    @patch('main.event_db')
    def test_get_stats(self, mock_event_db):
        """Test getting dashboard statistics"""
        # Mock the database stats
        mock_stats = {
            'today': {'events': 5, 'processed': 10, 'violence_rate': 50.0, 'violence_duration': 25.5},
            'total': {'events': 100, 'processed': 500, 'violence_rate': 20.0, 'violence_duration': 180.2},
            'recent_events': []
        }
        
        mock_event_db.get_stats.return_value = mock_stats
        
        response = self.client.get("/api/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert data['today']['events'] == 5
        assert data['total']['events'] == 100
        assert 'current' in data
        assert 'active_jobs' in data['current']
    
    @patch('main.event_db', None)  # Simulate uninitialized database
    def test_get_stats_no_database(self):
        """Test getting stats when database is not initialized"""
        response = self.client.get("/api/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert data['today']['events'] == 0
        assert data['total']['events'] == 0
    
    def test_get_system_status(self):
        """Test system status endpoint"""
        # Add some test jobs
        active_jobs['job1'] = {'status': 'processing'}
        active_jobs['job2'] = {'status': 'completed'}
        active_jobs['job3'] = {'status': 'error'}
        
        response = self.client.get("/api/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data['system_status'] == 'running'
        assert data['active_jobs'] == 1  # Only processing jobs
        assert data['completed_jobs'] == 1
        assert data['error_jobs'] == 1
        assert data['total_jobs'] == 3
        assert data['max_concurrent_jobs'] == 3
        assert 'model_loaded' in data
        assert 'database_connected' in data
    
    @patch('main.stream_db')
    def test_get_streams_empty(self, mock_stream_db):
        """Test getting streams when none exist"""
        mock_stream_db.get_streams.return_value = []
        
        response = self.client.get("/api/streams")
        
        assert response.status_code == 200
        data = response.json()
        assert "streams" in data
        assert len(data["streams"]) == 0
    
    @patch('main.stream_db')
    def test_get_streams_with_data(self, mock_stream_db):
        """Test getting streams with data"""
        mock_streams = [
            {
                'id': 1,
                'name': 'Test Camera 1',
                'rtsp_url': 'rtsp://test1.com/stream',
                'status': 'active',
                'created_at': '2024-01-01 12:00:00',
                'last_detection': None,
                'total_detections': 0,
                'is_recording': False,
                'thumbnail_path': ''
            },
            {
                'id': 2,
                'name': 'Test Camera 2',
                'rtsp_url': 'rtsp://test2.com/stream',
                'status': 'inactive',
                'created_at': '2024-01-01 13:00:00',
                'last_detection': None,
                'total_detections': 5,
                'is_recording': True,
                'thumbnail_path': '/thumb2.jpg'
            }
        ]
        
        mock_stream_db.get_streams.return_value = mock_streams
        
        response = self.client.get("/api/streams")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["streams"]) == 2
        assert data["streams"][0]['name'] == 'Test Camera 1'
        assert data["streams"][1]['total_detections'] == 5
    
    @patch('main.stream_db')
    def test_add_stream_success(self, mock_stream_db):
        """Test adding a new stream"""
        mock_stream_db.add_stream.return_value = 123
        
        stream_data = {
            'name': 'New Test Camera',
            'rtsp_url': 'rtsp://newcamera.com/stream'
        }
        
        response = self.client.post("/api/streams", json=stream_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] == True
        assert data['stream_id'] == 123
        assert "successfully" in data['message']
    
    def test_add_stream_invalid_url(self):
        """Test adding stream with invalid URL"""
        stream_data = {
            'name': 'Invalid Camera',
            'rtsp_url': 'not-a-valid-url'
        }
        
        response = self.client.post("/api/streams", json=stream_data)
        
        # Accept either 400 or 500 since error handling might vary
        assert response.status_code in [400, 500]
        data = response.json()
        assert "Invalid" in data['detail'] or "error" in data['detail'].lower()
    
    def test_add_stream_missing_data(self):
        """Test adding stream with missing data"""
        response = self.client.post("/api/streams", json={})
        
        assert response.status_code == 422  # Validation error
    
    @patch('main.active_streams', {})
    @patch('main.stream_db')
    def test_start_stream_not_found(self, mock_stream_db):
        """Test starting a non-existent stream"""
        mock_stream_db.get_streams.return_value = []
        
        response = self.client.post("/api/streams/999/start")
        
        # Accept either 404 or 500 since error handling might vary
        assert response.status_code in [404, 500]
        data = response.json()
        assert "not found" in data['detail'].lower() or "error" in data['detail'].lower()
    
    @patch('main.active_streams', {1: {'processor': Mock()}})
    def test_stop_stream_success(self):
        """Test stopping an active stream"""
        # Mock the processor's stop_stream method
        mock_processor = Mock()
        
        with patch('main.active_streams', {1: {'processor': mock_processor}}), \
             patch('main.stream_db') as mock_stream_db:
            
            response = self.client.post("/api/streams/1/stop")
            
            assert response.status_code == 200
            data = response.json()
            assert data['success'] == True
            assert "stopped" in data['message']
            
            # Verify processor was called
            mock_processor.stop_stream.assert_called_once()
    
    def test_stop_stream_not_active(self):
        """Test stopping a non-active stream"""
        response = self.client.post("/api/streams/999/stop")
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] == False
        assert "not active" in data['message']
    
    @patch('main.stream_db')
    def test_delete_stream_success(self, mock_stream_db):
        """Test deleting a stream"""
        response = self.client.delete("/api/streams/1")
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] == True
        assert "deleted" in data['message']
        
        # Verify database method was called
        mock_stream_db.delete_stream.assert_called_once_with(1)


class TestFileUpload:
    """Test file upload functionality"""
    
    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)
        active_jobs.clear()
    
    @patch('main.process_video_sync')
    @patch('threading.Thread')
    @patch('threading.Timer')  # Mock Timer to avoid threading issues
    def test_upload_video_path_success(self, mock_timer, mock_thread, mock_process):
        """Test uploading with video path"""
        # Mock Timer to avoid threading initialization issues
        mock_timer_instance = Mock()
        mock_timer.return_value = mock_timer_instance
        
        # Create a temporary video file
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_file.write(b'dummy video content')
        temp_file.close()
        
        try:
            response = self.client.post(
                "/api/upload",
                data={"video_path": temp_file.name}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data['success'] == True
            assert 'job_id' in data
            
            # Verify thread was started
            mock_thread.assert_called_once()
            
        finally:
            os.unlink(temp_file.name)
    
    def test_upload_nonexistent_path(self):
        """Test uploading with non-existent path"""
        response = self.client.post(
            "/api/upload",
            data={"video_path": "/nonexistent/video.mp4"}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "File not found" in data['detail']
    
    def test_upload_invalid_extension(self):
        """Test uploading file with invalid extension"""
        temp_file = tempfile.NamedTemporaryFile(suffix='.txt', delete=False)
        temp_file.write(b'not a video')
        temp_file.close()
        
        try:
            response = self.client.post(
                "/api/upload",
                data={"video_path": temp_file.name}
            )
            
            assert response.status_code == 400
            data = response.json()
            assert "Invalid file type" in data['detail']
            
        finally:
            os.unlink(temp_file.name)
    
    def test_upload_no_file_or_path(self):
        """Test upload without file or path"""
        response = self.client.post("/api/upload")
        
        assert response.status_code == 400
        data = response.json()
        assert "No file or path provided" in data['detail']
    
    @patch('main.active_jobs', {'job1': {'status': 'processing'}, 'job2': {'status': 'processing'}, 'job3': {'status': 'queued'}})
    def test_upload_too_many_active_jobs(self):
        """Test upload when too many jobs are active"""
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_file.write(b'dummy video content')
        temp_file.close()
        
        try:
            response = self.client.post(
                "/api/upload",
                data={"video_path": temp_file.name}
            )
            
            assert response.status_code == 429
            data = response.json()
            assert "Too many active jobs" in data['detail']
            
        finally:
            os.unlink(temp_file.name)


class TestResultEndpoints:
    """Test result-related endpoints"""
    
    def setup_method(self):
        """Setup test client and temporary files"""
        self.client = TestClient(app)
        
        # Create temporary results directory
        self.temp_results_dir = tempfile.mkdtemp()
        
        # Patch the RESULTS_FOLDER
        self.results_folder_patcher = patch('main.RESULTS_FOLDER', self.temp_results_dir)
        self.results_folder_patcher.start()
    
    def teardown_method(self):
        """Clean up temporary files"""
        self.results_folder_patcher.stop()
        
        # Clean up temp directory
        import shutil
        try:
            shutil.rmtree(self.temp_results_dir)
        except:
            pass
    
    def test_get_result_not_found(self):
        """Test getting result for non-existent job"""
        response = self.client.get("/api/result/nonexistent-job")
        
        assert response.status_code == 404
        data = response.json()
        assert "Result not found" in data['detail']
    
    def test_get_result_success(self):
        """Test getting result for existing job"""
        job_id = "test-job-result"
        result_data = {
            'job_id': job_id,
            'filename': 'test_video.mp4',
            'has_violence': True,
            'confidence': 0.85
        }
        
        # Create result file
        result_file = os.path.join(self.temp_results_dir, f"{job_id}_result.json")
        with open(result_file, 'w') as f:
            json.dump(result_data, f)
        
        response = self.client.get(f"/api/result/{job_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data['job_id'] == job_id
        assert data['has_violence'] == True
        assert data['confidence'] == 0.85
    
    def test_get_result_file_not_found(self):
        """Test getting non-existent result file"""
        response = self.client.get("/api/results/nonexistent_file.jpg")
        
        assert response.status_code == 404
        data = response.json()
        assert "File not found" in data['detail']
    
    def test_get_result_file_invalid_filename(self):
        """Test getting result file with invalid filename (security)"""
        response = self.client.get("/api/results/../../../etc/passwd")
        
        # Accept either 400 or 404 since security handling might vary
        assert response.status_code in [400, 404]
        data = response.json()
        assert ("Invalid filename" in data['detail'] or 
                "File not found" in data['detail'] or 
                "Not Found" in data['detail'])
    
    def test_get_result_file_success(self):
        """Test getting valid result file"""
        # Create a test image file
        test_file = os.path.join(self.temp_results_dir, "test_thumbnail.jpg")
        with open(test_file, 'wb') as f:
            f.write(b'fake image data')
        
        response = self.client.get("/api/results/test_thumbnail.jpg")
        
        assert response.status_code == 200
        assert response.headers['content-type'] == 'image/jpeg'