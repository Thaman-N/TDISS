import pytest
import sqlite3
import tempfile
import os
import json
from datetime import datetime, timedelta
from unittest.mock import patch

# Import the classes from main.py
import sys
from pathlib import Path 
sys.path.append(str(Path(__file__).parent.parent))

from main import EventDatabase, StreamDatabase, ViolenceEvent, RTSPStream


class TestEventDatabase:
    """Test the EventDatabase class"""
    
    def setup_method(self):
        """Setup test database in temporary file"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db = EventDatabase(self.temp_db.name)
        
    def teardown_method(self):
        """Clean up temporary database"""
        try:
            os.unlink(self.temp_db.name)
        except:
            pass
    
    def test_database_initialization(self):
        """Test that database tables are created correctly"""
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        
        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        assert 'violence_events' in tables
        assert 'daily_stats' in tables
        conn.close()
    
    def test_save_event(self):
        """Test saving a violence event"""
        event = ViolenceEvent(
            timestamp='2024-01-01 12:00:00',
            source_type='upload',
            source_id='test_job_123',
            filename='test_video.mp4',
            start_time=10.0,
            end_time=15.0,
            duration=5.0,
            confidence=0.85,
            thumbnail_path='/test/thumb.jpg',
            clip_path='/test/clip.mp4',
            metadata='{"test": "data"}'
        )
        
        event_id = self.db.save_event(event)
        assert isinstance(event_id, int)
        assert event_id > 0
    
    def test_get_stats(self):
        """Test getting statistics"""
        # Add a test event
        event = ViolenceEvent(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            source_type='upload',
            source_id='test_job',
            filename='test.mp4',
            start_time=0.0,
            end_time=5.0,
            duration=5.0,
            confidence=0.9,
            thumbnail_path='',
            clip_path='',
            metadata=''
        )
        
        self.db.save_event(event)
        self.db.update_daily_processed_count(1)
        
        stats = self.db.get_stats()
        
        assert 'today' in stats
        assert 'total' in stats
        assert 'recent_events' in stats
        assert stats['today']['events'] == 1
        assert stats['today']['processed'] == 1
    
    def test_get_events_by_date_range(self):
        """Test getting events by date range"""
        # Add test events with different dates
        event1 = ViolenceEvent(
            timestamp='2024-01-01 12:00:00',
            source_type='upload',
            source_id='job1',
            filename='video1.mp4',
            start_time=0.0,
            end_time=5.0,
            duration=5.0,
            confidence=0.8,
            thumbnail_path='',
            clip_path='',
            metadata=''
        )
        
        event2 = ViolenceEvent(
            timestamp='2024-01-02 12:00:00',
            source_type='stream',
            source_id='stream1',
            filename='live_feed',
            start_time=0.0,
            end_time=3.0,
            duration=3.0,
            confidence=0.9,
            thumbnail_path='',
            clip_path='',
            metadata=''
        )
        
        self.db.save_event(event1)
        self.db.save_event(event2)
        
        events = self.db.get_events_by_date_range('2024-01-01', '2024-01-02 23:59:59')
        assert len(events) == 2
        
        events = self.db.get_events_by_date_range('2024-01-01', '2024-01-01 23:59:59')
        assert len(events) == 1


class TestStreamDatabase:
    """Test the StreamDatabase class"""
    
    def setup_method(self):
        """Setup test database in temporary file"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db = StreamDatabase(self.temp_db.name)
        
    def teardown_method(self):
        """Clean up temporary database"""
        try:
            os.unlink(self.temp_db.name)
        except:
            pass
    
    def test_database_initialization(self):
        """Test that stream tables are created correctly"""
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        assert 'rtsp_streams' in tables
        conn.close()
    
    def test_add_stream(self):
        """Test adding a new RTSP stream"""
        stream = RTSPStream(
            name='Test Camera',
            rtsp_url='rtsp://test.example.com/stream',
            status='inactive',
            thumbnail_path=''
        )
        
        stream_id = self.db.add_stream(stream)
        assert isinstance(stream_id, int)
        assert stream_id > 0
    
    def test_get_streams(self):
        """Test getting all streams"""
        # Add test streams
        stream1 = RTSPStream(
            name='Camera 1',
            rtsp_url='rtsp://cam1.example.com/stream',
            status='active',
            thumbnail_path='/thumb1.jpg'
        )
        
        stream2 = RTSPStream(
            name='Camera 2',
            rtsp_url='rtsp://cam2.example.com/stream',
            status='inactive',
            thumbnail_path=''
        )
        
        id1 = self.db.add_stream(stream1)
        id2 = self.db.add_stream(stream2)
        
        streams = self.db.get_streams()
        assert len(streams) == 2
        
        # Check the streams contain expected data
        stream_names = [s['name'] for s in streams]
        assert 'Camera 1' in stream_names
        assert 'Camera 2' in stream_names
    
    def test_update_stream_status(self):
        """Test updating stream status"""
        stream = RTSPStream(
            name='Test Stream',
            rtsp_url='rtsp://test.com/stream',
            status='inactive',
            thumbnail_path=''
        )
        
        stream_id = self.db.add_stream(stream)
        
        # Update status
        self.db.update_stream_status(stream_id, 'active', '/new_thumb.jpg')
        
        streams = self.db.get_streams()
        updated_stream = next(s for s in streams if s['id'] == stream_id)
        
        assert updated_stream['status'] == 'active'
        assert updated_stream['thumbnail_path'] == '/new_thumb.jpg'
    
    def test_delete_stream(self):
        """Test deleting a stream"""
        stream = RTSPStream(
            name='Delete Me',
            rtsp_url='rtsp://delete.com/stream',
            status='inactive',
            thumbnail_path=''
        )
        
        stream_id = self.db.add_stream(stream)
        
        # Verify it exists
        streams = self.db.get_streams()
        assert len(streams) == 1
        
        # Delete it
        self.db.delete_stream(stream_id)
        
        # Verify it's gone
        streams = self.db.get_streams()
        assert len(streams) == 0
    
    def test_increment_detection_count(self):
        """Test incrementing detection count"""
        stream = RTSPStream(
            name='Detection Test',
            rtsp_url='rtsp://detect.com/stream',
            status='active',
            thumbnail_path=''
        )
        
        stream_id = self.db.add_stream(stream)
        
        # Increment detection count
        self.db.increment_detection_count(stream_id)
        self.db.increment_detection_count(stream_id)
        
        streams = self.db.get_streams()
        test_stream = next(s for s in streams if s['id'] == stream_id)
        
        assert test_stream['total_detections'] == 2