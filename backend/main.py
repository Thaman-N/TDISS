from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, List
import os
import time
import json
import uuid
import asyncio
from pathlib import Path
import shutil
import re
from datetime import datetime, timedelta
import sqlite3
from dataclasses import dataclass
import threading

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Add these imports at the top of main.py after existing imports
import cv2
import base64
import numpy as np
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
from collections import deque
import signal
import sys
import atexit

def secure_filename(filename):
    """Make a filename safe for use in URLs and file systems."""
    # Remove any path components
    filename = filename.replace('\\', '/').split('/')[-1]
    # Remove or replace dangerous characters
    filename = re.sub(r'[^\w\-_.]', '_', filename)
    # Remove multiple consecutive underscores
    filename = re.sub(r'_+', '_', filename)
    # Remove leading/trailing underscores and dots
    filename = filename.strip('_.')
    # Ensure filename is not empty
    if not filename:
        filename = 'unnamed_file'
    return filename

# Import your PyTorch detection module (copy these files to the same directory)
from torch_detection import load_violence_detection_model, extract_frames, preprocess_frames, predict_violence, extract_consecutive_frame_sequences

# Event storage classes
DB_PATH = "violence_events.db"
MAX_EVENT_CLIP_DURATION = 30  # seconds

@dataclass
class ViolenceEvent:
    """Violence event data structure"""
    id: Optional[int] = None
    timestamp: str = ""
    source_type: str = ""  # 'upload', 'stream', 'webcam'
    source_id: str = ""    # job_id, stream_id, etc.
    filename: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    confidence: float = 0.0
    thumbnail_path: str = ""
    clip_path: str = ""    # Short clip of the incident
    metadata: str = ""     # JSON string

# Add these classes after the existing ViolenceEvent class
@dataclass
class RTSPStream:
    """RTSP Stream configuration"""
    id: Optional[int] = None
    name: str = ""
    rtsp_url: str = ""
    status: str = "inactive"  # inactive, connecting, active, error
    created_at: str = ""
    last_detection: str = ""
    total_detections: int = 0
    is_recording: bool = False
    thumbnail_path: str = ""

class StreamDatabase:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.init_stream_tables()

    def init_stream_tables(self):
        """Initialize stream-related database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create streams table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rtsp_streams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                rtsp_url TEXT NOT NULL UNIQUE,
                status TEXT DEFAULT 'inactive',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_detection TIMESTAMP,
                total_detections INTEGER DEFAULT 0,
                is_recording BOOLEAN DEFAULT FALSE,
                thumbnail_path TEXT,
                settings TEXT DEFAULT '{}'
            )
        ''')

        conn.commit()
        conn.close()

    def add_stream(self, stream: RTSPStream) -> int:
        """Add a new RTSP stream"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO rtsp_streams (name, rtsp_url, status, thumbnail_path)
            VALUES (?, ?, ?, ?)
        ''', (stream.name, stream.rtsp_url, stream.status, stream.thumbnail_path))

        stream_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return stream_id

    def get_streams(self) -> List[Dict]:
        """Get all RTSP streams"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM rtsp_streams ORDER BY created_at DESC')
        streams = cursor.fetchall()
        conn.close()

        return [
            {
                'id': row[0],
                'name': row[1],
                'rtsp_url': row[2],
                'status': row[3],
                'created_at': row[4],
                'last_detection': row[5],
                'total_detections': row[6],
                'is_recording': row[7],
                'thumbnail_path': row[8]
            }
            for row in streams
        ]

    def update_stream_status(self, stream_id: int, status: str, thumbnail_path: str = None):
        """Update stream status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if thumbnail_path:
            cursor.execute('''
                UPDATE rtsp_streams 
                SET status = ?, thumbnail_path = ?, last_detection = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (status, thumbnail_path, stream_id))
        else:
            cursor.execute('''
                UPDATE rtsp_streams SET status = ? WHERE id = ?
            ''', (status, stream_id))

        conn.commit()
        conn.close()

    def delete_stream(self, stream_id: int):
        """Delete a stream"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM rtsp_streams WHERE id = ?', (stream_id,))
        conn.commit()
        conn.close()

    def increment_detection_count(self, stream_id: int):
        """Increment detection count for a stream"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE rtsp_streams 
            SET total_detections = total_detections + 1, last_detection = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (stream_id,))
        conn.commit()
        conn.close()


class EventDatabase:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize the events database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS violence_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_id TEXT NOT NULL,
                filename TEXT,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                duration REAL NOT NULL,
                confidence REAL NOT NULL,
                thumbnail_path TEXT,
                clip_path TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create daily stats table for fast lookups
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_stats (
                date TEXT PRIMARY KEY,
                total_events INTEGER DEFAULT 0,
                total_processed INTEGER DEFAULT 0,
                violence_duration REAL DEFAULT 0.0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create indices for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON violence_events(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON violence_events(source_type, source_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON daily_stats(date)')

        conn.commit()
        conn.close()

    def save_event(self, event: ViolenceEvent) -> int:
        """Save a violence event to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO violence_events 
            (timestamp, source_type, source_id, filename, start_time, end_time, 
             duration, confidence, thumbnail_path, clip_path, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event.timestamp, event.source_type, event.source_id, event.filename,
            event.start_time, event.end_time, event.duration, event.confidence,
            event.thumbnail_path, event.clip_path, event.metadata
        ))

        event_id = cursor.lastrowid

        # Update daily stats
        today = datetime.now().strftime('%Y-%m-%d')
        cursor.execute('''
            INSERT OR REPLACE INTO daily_stats (date, total_events, violence_duration, total_processed, last_updated)
            VALUES (?, 
                COALESCE((SELECT total_events FROM daily_stats WHERE date = ?), 0) + 1,
                COALESCE((SELECT violence_duration FROM daily_stats WHERE date = ?), 0) + ?,
                COALESCE((SELECT total_processed FROM daily_stats WHERE date = ?), 0),
                CURRENT_TIMESTAMP
            )
        ''', (today, today, today, event.duration, today))

        conn.commit()
        conn.close()
        return event_id

    def update_daily_processed_count(self, count: int = 1):
        """Update daily processed count"""
        today = datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO daily_stats (date, total_processed, total_events, violence_duration, last_updated)
            VALUES (?, 
                COALESCE((SELECT total_processed FROM daily_stats WHERE date = ?), 0) + ?,
                COALESCE((SELECT total_events FROM daily_stats WHERE date = ?), 0),
                COALESCE((SELECT violence_duration FROM daily_stats WHERE date = ?), 0),
                CURRENT_TIMESTAMP
            )
        ''', (today, today, count, today, today))

        conn.commit()
        conn.close()

    def get_stats(self) -> Dict:
        """Get comprehensive statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        today = datetime.now().strftime('%Y-%m-%d')

        # Today's stats
        cursor.execute('SELECT * FROM daily_stats WHERE date = ?', (today,))
        today_stats = cursor.fetchone()

        if today_stats:
            today_events = today_stats[1]
            today_processed = today_stats[2]
            today_violence_duration = today_stats[3]
        else:
            today_events = 0
            today_processed = 0
            today_violence_duration = 0.0

        # Total stats
        cursor.execute('SELECT COUNT(*) FROM violence_events')
        total_events = cursor.fetchone()[0]

        cursor.execute('SELECT SUM(total_processed) FROM daily_stats')
        total_processed_result = cursor.fetchone()[0]
        total_processed = total_processed_result if total_processed_result else 0

        cursor.execute('SELECT SUM(violence_duration) FROM daily_stats')
        total_violence_duration_result = cursor.fetchone()[0]
        total_violence_duration = total_violence_duration_result if total_violence_duration_result else 0.0

        # Violence rate
        violence_rate = (today_events / today_processed * 100) if today_processed > 0 else 0
        total_violence_rate = (total_events / total_processed * 100) if total_processed > 0 else 0

        # Recent events (last 24 hours)
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('''
            SELECT * FROM violence_events 
            WHERE timestamp >= ? 
            ORDER BY timestamp DESC 
            LIMIT 10
        ''', (yesterday,))
        recent_events = cursor.fetchall()

        conn.close()

        return {
            'today': {
                'events': today_events,
                'processed': today_processed,
                'violence_duration': round(today_violence_duration, 2),
                'violence_rate': round(violence_rate, 1)
            },
            'total': {
                'events': total_events,
                'processed': total_processed,
                'violence_duration': round(total_violence_duration, 2),
                'violence_rate': round(total_violence_rate, 1)
            },
            'recent_events': [
                {
                    'id': row[0],
                    'timestamp': row[1],
                    'source_type': row[2],
                    'filename': row[4],
                    'duration': row[7],
                    'confidence': round(row[8], 3),
                    'thumbnail': row[9]
                }
                for row in recent_events
            ]
        }

    def get_events_by_date_range(self, start_date: str, end_date: str) -> List[Dict]:
        """Get events within date range"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM violence_events 
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp DESC
        ''', (start_date, end_date))

        events = cursor.fetchall()
        conn.close()

        return [
            {
                'id': row[0],
                'timestamp': row[1],
                'source_type': row[2],
                'source_id': row[3],
                'filename': row[4],
                'start_time': row[5],
                'end_time': row[6],
                'duration': row[7],
                'confidence': row[8],
                'thumbnail_path': row[9],
                'clip_path': row[10],
                'metadata': json.loads(row[11]) if row[11] else {}
            }
            for row in events
        ]

def save_violence_clip(video_path: str, start_time: float, end_time: float, output_path: str) -> bool:
    """Save a short clip of the violence incident"""
    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        # Limit clip duration
        max_frames = int(MAX_EVENT_CLIP_DURATION * fps)
        if end_frame - start_frame > max_frames:
            end_frame = start_frame + max_frames

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while cap.isOpened() and frame_count < (end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break

            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()
        return True

    except Exception as e:
        print(f"Error saving violence clip: {e}")
        return False

def process_and_save_events(job_id: str, result: Dict, video_path: str):
    """Process detection result and save events to database"""

    # Check if database is initialized
    if event_db is None:
        print(f"Database not initialized, skipping event storage for job {job_id}")
        return

    try:
        # Update processed count
        event_db.update_daily_processed_count(1)
        print(f"Updated processed count for job {job_id}")

        # Save violence events
        if result['segments']:
            for i, segment in enumerate(result['segments']):
                # Create clip for significant events (high confidence or long duration)
                clip_path = ""
                if segment['confidence'] > 0.8 or segment['end'] - segment['start'] > 5.0:
                    clip_filename = f"{job_id}_clip_{i}.mp4"
                    clips_folder = os.path.join(RESULTS_FOLDER, "clips")
                    os.makedirs(clips_folder, exist_ok=True)
                    clip_full_path = os.path.join(clips_folder, clip_filename)

                    if save_violence_clip(video_path, segment['start'], segment['end'], clip_full_path):
                        clip_path = f"/api/results/clips/{clip_filename}"
                    else:
                        clip_path = ""

                # Create event
                event = ViolenceEvent(
                    timestamp=result['timestamp'],
                    source_type='upload',
                    source_id=job_id,
                    filename=result['filename'],
                    start_time=segment['start'],
                    end_time=segment['end'],
                    duration=segment['end'] - segment['start'],
                    confidence=segment['confidence'],
                    thumbnail_path=result.get('thumbnail', ''),
                    clip_path=clip_path,
                    metadata=json.dumps({
                        'overall_confidence': result['overall_result']['confidence'],
                        'model_info': result['model_info'],
                        'video_metadata': result['metadata']
                    })
                )

                event_id = event_db.save_event(event)
                print(f"Saved violence event {event_id} for job {job_id}")
        else:
            print(f"No violence segments found for job {job_id}")

    except Exception as e:
        print(f"Error in process_and_save_events for job {job_id}: {e}")
        import traceback
        traceback.print_exc()

app = FastAPI(
    title="Violence Detection API",
    description="X3D-based violence detection system with event storage",
    version="1.1.0"
)

# Rate limiting setup
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

shutdown_in_progress = False
shutdown_lock = threading.Lock()

# Configuration
UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB
MODEL_PATH = r'nineone75.pth'
DETECTION_THRESHOLD = 0.6

# Cleanup and resource management configuration
MAX_ACTIVE_JOBS = 100
MAX_HISTORY_ITEMS = 500
JOB_CLEANUP_AGE_HOURS = 24
FILE_CLEANUP_DELAY_HOURS = 24

# Create folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(os.path.join(RESULTS_FOLDER, "clips"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_FOLDER, "stream_thumbnails"), exist_ok=True)

# Store active jobs and results
active_jobs: Dict[str, dict] = {}
results_history: Dict[str, dict] = {}

# Initialize database - moved to startup to avoid blocking
event_db = None

# WebSocket connections for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_job_update(self, job_id: str, data: dict):
        if not self.active_connections:
            return

        message = {"type": "job_update", "job_id": job_id, "data": data}

        # Create a copy to avoid modification during iteration
        connections_copy = self.active_connections.copy()
        dead_connections = []

        for connection in connections_copy:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"WebSocket send failed: {e}")
                dead_connections.append(connection)

        # Remove dead connections
        for dead_conn in dead_connections:
            self.disconnect(dead_conn)

        if dead_connections:
            print(f"Removed {len(dead_connections)} dead connections")

manager = ConnectionManager()

# Load model at startup
model = None

# Stream management globals
active_streams: Dict[int, dict] = {}
stream_db = None
executor = ThreadPoolExecutor(max_workers=4)

# Graceful shutdown handling
global_shutdown_event = threading.Event()

def cleanup_all_streams():
    """Clean up all active streams gracefully"""
    global shutdown_in_progress
    
    with shutdown_lock:
        if shutdown_in_progress:
            print("Cleanup already in progress, skipping...")
            return
        shutdown_in_progress = True
    
    print("Cleaning up all active streams...")
    streams_to_stop = list(active_streams.keys())

    for stream_id in streams_to_stop:
        try:
            print(f"Stopping stream {stream_id}...")
            active_streams[stream_id]['processor'].stop_stream()
        except Exception as e:
            print(f"Error stopping stream {stream_id}: {e}")

    # Fix: Remove timeout parameter for older Python versions
    try:
        executor.shutdown(wait=True)
        print("Executor shutdown completed")
    except Exception as e:
        print(f"Error during executor shutdown: {e}")
    
    # Update database status for all streams
    if stream_db:
        try:
            conn = sqlite3.connect(stream_db.db_path)
            cursor = conn.cursor()
            cursor.execute("UPDATE rtsp_streams SET status = 'inactive'")
            conn.commit()
            conn.close()
            print("Updated all streams to inactive status")
        except Exception as e:
            print(f"Error updating stream status: {e}")
    
    print("All streams cleaned up")

# Add this new function to handle startup recovery:
def recover_stream_states():
    """Recover stream states on startup - mark all as inactive"""
    if stream_db:
        try:
            conn = sqlite3.connect(stream_db.db_path)
            cursor = conn.cursor()
            cursor.execute("UPDATE rtsp_streams SET status = 'inactive'")
            conn.commit()
            conn.close()
            print("Recovered stream states - all marked as inactive")
        except Exception as e:
            print(f"Error during stream state recovery: {e}")

async def check_stream_health():
    """Periodically check stream health and auto-recover if needed"""
    while not global_shutdown_event.is_set():
        try:
            await asyncio.sleep(30)  # Check every 30 seconds
            
            if global_shutdown_event.is_set():
                break
                
            streams_to_check = list(active_streams.keys())
            for stream_id in streams_to_check:
                if global_shutdown_event.is_set():
                    break
                    
                try:
                    processor = active_streams[stream_id]['processor']
                    
                    # Check if threads are alive
                    capture_alive = processor.capture_thread and processor.capture_thread.is_alive()
                    process_alive = processor.process_thread and processor.process_thread.is_alive()
                    
                    if not processor.is_running or not capture_alive or not process_alive:
                        print(f"Stream {stream_id} health check failed - marking as error")
                        
                        # Stop the problematic stream
                        try:
                            processor.stop_stream()
                        except:
                            pass
                        
                        # Update database status
                        if stream_db:
                            try:
                                stream_db.update_stream_status(stream_id, 'error')
                            except:
                                pass
                        
                except Exception as e:
                    print(f"Error checking health for stream {stream_id}: {e}")
                    
        except Exception as e:
            if not global_shutdown_event.is_set():
                print(f"Error during stream health check: {e}")
    
    print("Stream health check stopped")

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully - only run once"""
    global shutdown_in_progress
    
    with shutdown_lock:
        if shutdown_in_progress:
            print("\nForce exit...")
            sys.exit(1)
        shutdown_in_progress = True
    
    print(f"\nReceived signal {signum}. Shutting down gracefully...")
    global_shutdown_event.set()
    cleanup_all_streams()
    print("Shutdown complete")
    
    # Force exit to prevent hanging
    try:
        sys.exit(0)
    except SystemExit:
        pass

def cleanup_on_exit():
    """Cleanup function for atexit"""
    global shutdown_in_progress
    if not shutdown_in_progress:
        print("Exit handler triggered - cleaning up...")
        cleanup_all_streams()

# Register handlers - replace the existing signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup_on_exit)

class RTSPStreamProcessor:
    def __init__(self, stream_id: int, rtsp_url: str, stream_name: str):
        self.stream_id = stream_id
        self.rtsp_url = rtsp_url
        self.stream_name = stream_name

        # Capture objects and control
        self.cap = None
        self.is_running = False
        self.capture_thread = None
        self.process_thread = None

        # Frame buffers and processing
        self.raw_frame_queue = queue.Queue(maxsize=30)  # Raw frames from RTSP
        self.rgb_frame_buffer = deque(maxlen=16)  # RGB frames for model (224x224x3 uint8)
        self.last_display_frame = None  # Last frame for display/thumbnail

        # Timing and rate control
        self.last_detection_time = 0
        self.detection_interval = 3.0  # Process every 3 seconds for stability
        self.target_fps = 8  # Target FPS for model processing (not display)
        self.frame_skip_counter = 0

        # Model input requirements (from torch_detection.py)
        self.model_input_size = (336, 336)  # INPUT_SIZE = 336
        self.model_temporal_length = 16  # NUM_FRAMES = 16

        # Thread synchronization
        self._lock = threading.Lock()

    def validate_rtsp_url(self, url: str) -> bool:
        """Validate RTSP URL format"""
        try:
            parsed = urlparse(url)
            return parsed.scheme.lower() in ['rtsp', 'rtmp', 'http', 'https'] and parsed.netloc
        except:
            return False

    def preprocess_frame_for_buffer(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a single frame to match extract_frames() output format
        This creates RGB uint8 frames in [224, 224, 3] format - exactly like extract_frames()
        """
        try:
            # Step 1: Resize to model input size (like extract_frames does)
            frame_resized = cv2.resize(frame, self.model_input_size, interpolation=cv2.INTER_LINEAR)

            # Step 2: Convert BGR to RGB (like extract_frames does)
            if len(frame_resized.shape) == 3 and frame_resized.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame_resized.copy()

            # Step 3: Ensure uint8 type (like extract_frames output)
            if frame_rgb.dtype != np.uint8:
                frame_rgb = frame_rgb.astype(np.uint8)

            return frame_rgb  # Shape: [224, 224, 3], dtype: uint8, format: RGB

        except Exception as e:
            print(f"Error preprocessing frame for stream {self.stream_id}: {e}")
            return None

    def start_stream(self):
        """Start the RTSP stream with proper error handling"""
        if not self.validate_rtsp_url(self.rtsp_url):
            print(f"Invalid RTSP URL: {self.rtsp_url}")
            return False

        try:
            # Initialize capture with proper settings
            self.cap = cv2.VideoCapture(self.rtsp_url)

            # Critical settings for RTSP stability
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer to reduce latency

            # Try to set timeout properties (not all OpenCV versions support these)
            try:
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)  # 10 second open timeout
                self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)   # 5 second read timeout
            except:
                pass  # Ignore if not supported

            # Test connection
            if not self.cap.isOpened():
                print(f"Failed to open RTSP stream: {self.rtsp_url}")
                return False

            # Test frame reading
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                print(f"Failed to read test frame from stream: {self.rtsp_url}")
                self.cap.release()
                return False

            print(f"Stream {self.stream_id} frame size: {test_frame.shape}")

            # Start processing
            self.is_running = True
            active_streams[self.stream_id] = {
                'processor': self,
                'status': 'active',
                'name': self.stream_name,
                'rtsp_url': self.rtsp_url
            }

            # Start threads
            self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
            self.process_thread = threading.Thread(target=self._process_frames, daemon=True)

            self.capture_thread.start()
            self.process_thread.start()

            print(f"Successfully started stream {self.stream_id}: {self.stream_name}")
            return True

        except Exception as e:
            print(f"Error starting stream {self.stream_id}: {e}")
            if self.cap:
                self.cap.release()
            return False

    def stop_stream(self):
        """Stop the RTSP stream and cleanup resources"""
        if not self.is_running:
            print(f"Stream {self.stream_id} already stopped")
            return
            
        print(f"Stopping stream {self.stream_id}")
        self.is_running = False

        # Clean up capture first
        if self.cap:
            try:
                self.cap.release()
                self.cap = None
            except Exception as e:
                print(f"Error releasing capture for stream {self.stream_id}: {e}")

        # Set threads to None to avoid joining in signal handlers
        threads_to_join = []
        capture_alive = False
        process_alive = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            capture_alive = True
            threads_to_join.append(("capture", self.capture_thread))
        
        if self.process_thread and self.process_thread.is_alive():
            process_alive = True
            threads_to_join.append(("process", self.process_thread))
        
        # IMPORTANT: Don't join threads if we're in a global shutdown
        if not global_shutdown_event.is_set() and not shutdown_in_progress:
            for thread_name, thread in threads_to_join:
                try:
                    # Use a shorter timeout to avoid blocking
                    thread.join(timeout=1.0)
                    if thread.is_alive():
                        print(f"Warning: {thread_name} thread for stream {self.stream_id} did not stop gracefully")
                except Exception as e:
                    print(f"Error joining {thread_name} thread for stream {self.stream_id}: {e}")
        else:
            print(f"Skipping thread join for stream {self.stream_id} during shutdown")

        # Only report on thread status if we're not in shutdown
        if not global_shutdown_event.is_set() and not shutdown_in_progress:
            if capture_alive and self.capture_thread and self.capture_thread.is_alive():
                print(f"Note: Capture thread for stream {self.stream_id} still running")
            if process_alive and self.process_thread and self.process_thread.is_alive():
                print(f"Note: Process thread for stream {self.stream_id} still running")

        # Clear buffers safely - don't block
        try:
            # Clear raw frame queue without blocking
            try:
                while not self.raw_frame_queue.empty():
                    try:
                        self.raw_frame_queue.get_nowait()
                    except:
                        break
            except:
                pass
            
            # Clear RGB buffer
            try:
                self.rgb_frame_buffer.clear()
            except:
                pass
            
            # Clear display frame
            try:
                with self._lock:
                    self.last_display_frame = None
            except:
                pass
                
        except Exception as e:
            print(f"Error clearing buffers for stream {self.stream_id}: {e}")

        # Remove from active streams
        try:
            if self.stream_id in active_streams:
                del active_streams[self.stream_id]
        except Exception as e:
            print(f"Error removing stream {self.stream_id} from active streams: {e}")

        # Ensure we're not setting threads to None during shutdown
        if not global_shutdown_event.is_set() and not shutdown_in_progress:
            self.capture_thread = None
            self.process_thread = None

        print(f"Stream {self.stream_id} stopped successfully")

    def _capture_frames(self):
        """
        Dedicated thread for capturing frames from RTSP stream
        This runs at the stream's native frame rate
        """
        consecutive_failures = 0
        max_failures = 10

        # In _capture_frames
        while self.is_running and not global_shutdown_event.is_set() and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()

                if not ret or frame is None:
                    consecutive_failures += 1
                    print(f"Stream {self.stream_id}: Failed to read frame ({consecutive_failures}/{max_failures})")

                    if consecutive_failures >= max_failures:
                        print(f"Stream {self.stream_id}: Too many consecutive failures, stopping")
                        break

                    time.sleep(0.1)  # Brief pause before retry
                    continue

                # Reset failure counter on successful read
                consecutive_failures = 0

                # Store raw frame for display/thumbnail (keep BGR format)
                with self._lock:
                    self.last_display_frame = frame.copy()

                # Add to processing queue (non-blocking)
                try:
                    self.raw_frame_queue.put(frame, block=False)
                except queue.Full:
                    # Drop oldest frame if queue is full (maintain real-time processing)
                    try:
                        self.raw_frame_queue.get_nowait()
                        self.raw_frame_queue.put(frame, block=False)
                    except:
                        pass  # Continue if we can't manage the queue

                # Control capture rate to prevent overwhelming the processing
                time.sleep(0.033)  # ~30 FPS capture rate

            except Exception as e:
                print(f"Error in capture thread for stream {self.stream_id}: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    break
                time.sleep(0.5)  # Longer pause on errors

        print(f"Capture thread for stream {self.stream_id} ended")

    def _process_frames(self):
        """
        Dedicated thread for processing frames for ML model
        This runs at controlled rate and maintains temporal buffer in the EXACT format expected by the model
        """
        last_process_time = 0
        process_interval = 1.0 / self.target_fps  # Target processing interval
        frame_counter = 0
        last_buffer_add_time = 0
        buffer_interval = 0.2  # Add to buffer every 200ms for temporal diversity

        while self.is_running:
            try:
                current_time = time.time()

                # Rate control for processing
                if current_time - last_process_time < process_interval:
                    time.sleep(0.01)
                    continue

                # Get frame from capture queue
                try:
                    raw_frame = self.raw_frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                frame_counter += 1
                last_process_time = current_time

                # Only add to buffer periodically to ensure temporal diversity
                if current_time - last_buffer_add_time >= buffer_interval:
                    # Preprocess frame to match extract_frames() output format
                    rgb_frame = self.preprocess_frame_for_buffer(raw_frame)
                    if rgb_frame is not None:
                        self.rgb_frame_buffer.append(rgb_frame)
                        last_buffer_add_time = current_time
                        print(f"Stream {self.stream_id}: Added frame to buffer (size: {len(self.rgb_frame_buffer)}/16)")

                # Generate thumbnail periodically (every 60 processed frames)
                if frame_counter % 60 == 0:
                    self._save_thumbnail()

                # Run detection if we have enough frames and enough time has passed
                if (len(self.rgb_frame_buffer) >= self.model_temporal_length and
                    current_time - self.last_detection_time >= self.detection_interval):
                    self._run_detection()
                    self.last_detection_time = current_time

                # Send frame update via WebSocket (every 15 processed frames)
                if frame_counter % 15 == 0:
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(self._send_frame_update())
                        loop.close()
                    except Exception as e:
                        print(f"Error sending frame update for stream {self.stream_id}: {e}")

            except Exception as e:
                print(f"Error in process thread for stream {self.stream_id}: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1.0)

        print(f"Process thread for stream {self.stream_id} ended")

    def _run_detection(self):
        """
        Run violence detection using the EXACT same pipeline as torch_detection.py
        CRITICAL: This now matches extract_frames() → preprocess_frames() → predict_violence() flow
        """
        if model is None or len(self.rgb_frame_buffer) < self.model_temporal_length:
            return

        try:
            # Extract exactly 16 frames from buffer (same as extract_frames output)
            frames_list = list(self.rgb_frame_buffer)[-self.model_temporal_length:]

            # Convert to numpy array with shape [T, H, W, C] - exactly like extract_frames()
            # This should be [16, 224, 224, 3] RGB uint8
            frames_array = np.array(frames_list, dtype=np.uint8)

            # DEBUG: Check temporal diversity (are we seeing the same frame 16 times?)
            frame_differences = []
            for i in range(len(frames_array) - 1):
                diff = np.mean(np.abs(frames_array[i].astype(np.float32) - frames_array[i+1].astype(np.float32)))
                frame_differences.append(diff)

            avg_frame_diff = np.mean(frame_differences) if frame_differences else 0
            print(f"Stream {self.stream_id}: Avg frame difference: {avg_frame_diff:.2f} (should be > 0 for movement)")

            # DEBUG: Check if all frames are identical (bad temporal buffer)
            if avg_frame_diff < 1.0:
                print(f"WARNING: Stream {self.stream_id} has very low frame diversity - might be seeing same frame repeatedly")

            print(f"Stream {self.stream_id}: Detection input shape: {frames_array.shape}, dtype: {frames_array.dtype}")

            # Ensure model is in eval mode (CRITICAL)
            model.eval()

            # Check if model uses motion enhancement (from model architecture)
            use_motion = hasattr(model, 'use_motion_enhancement') and model.use_motion_enhancement
            print(f"Stream {self.stream_id}: Using motion enhancement: {use_motion}")

            # Use the EXACT same preprocessing pipeline as torch_detection.py
            processed_data = preprocess_frames(frames_array, compute_flow=use_motion)

            # DEBUG: Check preprocessed data shapes and ranges
            for key, tensor in processed_data.items():
                print(f"Stream {self.stream_id}: {key} tensor - shape: {tensor.shape}, dtype: {tensor.dtype}, range: [{tensor.min():.3f}, {tensor.max():.3f}]")

            # Run prediction using the exact same function WITH DEBUG
            is_violent, confidence, inference_time = predict_violence(
                model, processed_data, DETECTION_THRESHOLD, debug=True  # Enable debug!
            )

            print(f"Stream {self.stream_id}: Final result - Violence: {is_violent}, Confidence: {confidence:.3f}, Time: {inference_time:.3f}s")

            # Only save if confidence is reasonable (not 1.000 every time)
            if is_violent and confidence > DETECTION_THRESHOLD and confidence < 0.99:
                print(f"ALERT: Violence detected in stream {self.stream_id}: {confidence:.3f}")
                self._save_detection_event(confidence)
            elif is_violent and confidence >= 0.99:
                print(f"SUSPICIOUS: Stream {self.stream_id} showing max confidence {confidence:.3f} - possible model issue")
            else:
                print(f"Stream {self.stream_id}: No violence detected (confidence: {confidence:.3f})")

        except Exception as e:
            print(f"Error running detection on stream {self.stream_id}: {e}")
            import traceback
            traceback.print_exc()

    def _save_detection_event(self, confidence: float):
        """Save violence detection event to database with enhanced clip generation"""
        try:
            current_time = time.time()
            timestamp_str = time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Generate thumbnail from current frame
            thumbnail_filename = f"stream_{self.stream_id}_event_{int(current_time)}.jpg"
            thumbnail_dir = os.path.join(RESULTS_FOLDER, "stream_thumbnails")
            os.makedirs(thumbnail_dir, exist_ok=True)
            thumbnail_path = os.path.join(thumbnail_dir, thumbnail_filename)
            thumbnail_url = f"/api/results/stream_thumbnails/{thumbnail_filename}"
            
            # Save thumbnail
            try:
                with self._lock:
                    if self.last_display_frame is not None:
                        frame = self.last_display_frame.copy()
                        # Resize for thumbnail
                        height, width = frame.shape[:2]
                        max_dim = 400
                        if height > width:
                            new_height = max_dim
                            new_width = int(width * (max_dim / height))
                        else:
                            new_width = max_dim
                            new_height = int(height * (max_dim / width))
                        
                        thumbnail = cv2.resize(frame, (new_width, new_height))
                        cv2.imwrite(thumbnail_path, thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 85])
            except Exception as e:
                print(f"Error saving thumbnail for stream {self.stream_id}: {e}")
                thumbnail_url = ""

            # Generate a short clip if we have enough buffer frames
            clip_url = ""
            if len(self.rgb_frame_buffer) >= 8:  # At least 8 frames for a clip
                try:
                    clip_filename = f"stream_{self.stream_id}_clip_{int(current_time)}.mp4"
                    clips_dir = os.path.join(RESULTS_FOLDER, "stream_clips")
                    os.makedirs(clips_dir, exist_ok=True)
                    clip_path = os.path.join(clips_dir, clip_filename)
                    
                    # Create a short video clip from recent frames
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(clip_path, fourcc, 4.0, (336, 336))  # 4 FPS, model input size
                    
                    # Use last 8 frames from buffer (2 seconds at 4 FPS)
                    recent_frames = list(self.rgb_frame_buffer)[-8:]
                    for rgb_frame in recent_frames:
                        # Convert RGB back to BGR for video writing
                        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                        out.write(bgr_frame)
                    
                    out.release()
                    clip_url = f"/api/results/stream_clips/{clip_filename}"
                    print(f"Generated event clip for stream {self.stream_id}: {clip_url}")
                    
                except Exception as e:
                    print(f"Error generating clip for stream {self.stream_id}: {e}")

            # Create event record
            event = ViolenceEvent(
                timestamp=timestamp_str,
                source_type='stream',
                source_id=str(self.stream_id),
                filename=self.stream_name,
                start_time=0.0,  # For live streams, we don't have precise start/end times
                end_time=self.detection_interval,
                duration=self.detection_interval,
                confidence=confidence,
                thumbnail_path=thumbnail_url,
                clip_path=clip_url,
                metadata=json.dumps({
                    'stream_name': self.stream_name,
                    'rtsp_url': self.rtsp_url,
                    'detection_type': 'live_stream',
                    'buffer_size': len(self.rgb_frame_buffer),
                    'model_input_size': self.model_input_size,
                    'temporal_length': self.model_temporal_length,
                    'pipeline_version': 'enhanced_stream_v2',
                    'frame_timestamp': current_time,
                    'detection_interval': self.detection_interval
                })
            )

            # Save to database
            event_id = event_db.save_event(event)
            stream_db.increment_detection_count(self.stream_id)

            print(f"Saved enhanced detection event {event_id} for stream {self.stream_id}")

            # Send WebSocket notification about the violence detection
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(manager.send_job_update(f"violence_event_{event_id}", {
                    'type': 'violence_detected',
                    'event_id': event_id,
                    'stream_id': self.stream_id,
                    'stream_name': self.stream_name,
                    'confidence': confidence,
                    'thumbnail': thumbnail_url,
                    'clip': clip_url,
                    'timestamp': timestamp_str
                }))
                loop.close()
            except Exception as e:
                print(f"Error sending violence detection WebSocket update: {e}")

        except Exception as e:
            print(f"Error saving detection event for stream {self.stream_id}: {e}")
            import traceback
            traceback.print_exc()

    def _save_thumbnail(self):
        """Save current frame as thumbnail"""
        try:
            with self._lock:
                if self.last_display_frame is None:
                    return
                frame = self.last_display_frame.copy()

            thumbnail_dir = os.path.join(RESULTS_FOLDER, "stream_thumbnails")
            os.makedirs(thumbnail_dir, exist_ok=True)

            thumbnail_path = os.path.join(thumbnail_dir, f"stream_{self.stream_id}_thumb.jpg")

            # Resize for thumbnail (maintain aspect ratio)
            height, width = frame.shape[:2]
            max_dim = 300
            if height > width:
                new_height = max_dim
                new_width = int(width * (max_dim / height))
            else:
                new_width = max_dim
                new_height = int(height * (max_dim / width))

            thumbnail = cv2.resize(frame, (new_width, new_height))
            cv2.imwrite(thumbnail_path, thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 85])

            # Update database
            thumbnail_url = f"/api/results/stream_thumbnails/stream_{self.stream_id}_thumb.jpg"
            stream_db.update_stream_status(self.stream_id, 'active', thumbnail_url)

        except Exception as e:
            print(f"Error saving thumbnail for stream {self.stream_id}: {e}")

    def get_frame_base64(self):
        """Get the latest frame as base64 for WebSocket transmission"""
        try:
            with self._lock:
                if self.last_display_frame is None:
                    return None
                frame = self.last_display_frame.copy()

            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            return base64.b64encode(buffer).decode('utf-8')

        except Exception as e:
            print(f"Error encoding frame for stream {self.stream_id}: {e}")
            return None

    async def _send_frame_update(self):
        """Send frame update via WebSocket"""
        try:
            frame_data = self.get_frame_base64()
            if frame_data:
                await manager.send_job_update(f"stream_{self.stream_id}", {
                    'type': 'stream_frame',
                    'stream_id': self.stream_id,
                    'frame': frame_data,
                    'status': 'active',
                    'buffer_size': len(self.rgb_frame_buffer)
                })
        except Exception as e:
            print(f"Error sending frame update for stream {self.stream_id}: {e}")


async def cleanup_old_jobs():
    """Clean up old completed jobs to prevent memory leaks"""
    current_time = datetime.now()

    # Clean up active jobs older than specified hours
    jobs_to_remove = []
    for job_id, job in active_jobs.items():
        try:
            job_time = datetime.strptime(job['timestamp'], '%Y-%m-%d %H:%M:%S')
            if (current_time - job_time).total_seconds() > JOB_CLEANUP_AGE_HOURS * 3600:
                if job['status'] in ['completed', 'error']:
                    jobs_to_remove.append(job_id)
        except Exception as e:
            print(f"Error parsing job timestamp for {job_id}: {e}")
            # Remove jobs with invalid timestamps
            if job['status'] in ['completed', 'error']:
                jobs_to_remove.append(job_id)

    for job_id in jobs_to_remove:
        del active_jobs[job_id]
        print(f"Cleaned up old job: {job_id}")

    # Limit active jobs count
    if len(active_jobs) > MAX_ACTIVE_JOBS:
        # Remove oldest completed/error jobs
        completed_jobs = [(k, v) for k, v in active_jobs.items()
                             if v['status'] in ['completed', 'error']]
        completed_jobs.sort(key=lambda x: x[1]['timestamp'])

        jobs_to_remove = completed_jobs[:len(active_jobs) - MAX_ACTIVE_JOBS]
        for job_id, _ in jobs_to_remove:
            del active_jobs[job_id]

        print(f"Removed {len(jobs_to_remove)} jobs due to count limit")

    # Limit history size
    if len(results_history) > MAX_HISTORY_ITEMS:
        history_items = list(results_history.items())
        history_items.sort(key=lambda x: x[1]['timestamp'])

        items_to_remove = history_items[:len(results_history) - MAX_HISTORY_ITEMS]
        for job_id, _ in items_to_remove:
            del results_history[job_id]

        print(f"Removed {len(items_to_remove)} history items due to count limit")

async def periodic_cleanup():
    """Run cleanup every hour"""
    while True:
        await asyncio.sleep(3600)  # 1 hour
        try:
            await cleanup_old_jobs()
        except Exception as e:
            print(f"Error during periodic cleanup: {e}")

async def cleanup_uploaded_file(file_path: str, delay_hours: int = FILE_CLEANUP_DELAY_HOURS):
    """Clean up uploaded file after delay"""
    await asyncio.sleep(delay_hours * 3600)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up uploaded file: {file_path}")
    except Exception as e:
        print(f"Error cleaning up file {file_path}: {e}")

@app.on_event("startup")
async def startup_event():
    global model, event_db, stream_db
    try:
        print("Starting up Violence Detection API...")

        # Initialize databases
        print("Initializing databases...")
        event_db = EventDatabase()
        stream_db = StreamDatabase()
        print("Databases initialized successfully")

        # Recover stream states from previous shutdown
        recover_stream_states()

        print("Loading model...")
        model, _ = load_violence_detection_model(MODEL_PATH)
        print("Model loaded successfully")

        # Start background tasks
        asyncio.create_task(periodic_cleanup())
        asyncio.create_task(check_stream_health())
        print("Background tasks started")

        # Load existing history
        await load_history_from_file()
        print("History loaded")

        print("Violence Detection API started successfully")

    except Exception as e:
        print(f"Error during startup: {e}")
        import traceback
        traceback.print_exc()
        # Don't raise here - let the app start anyway
        print("Continuing startup despite errors...")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on FastAPI shutdown"""
    global shutdown_in_progress
    print("FastAPI shutdown event triggered")
    
    if not shutdown_in_progress:
        cleanup_all_streams()

async def load_history_from_file():
    """Load history from file if it exists"""
    history_path = os.path.join(RESULTS_FOLDER, 'history.json')
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
                for item in history:
                    results_history[item['job_id']] = item
                print(f"Loaded {len(history)} history items from file")
        except Exception as e:
            print(f"Error loading history: {e}")

# Pydantic models
class JobResponse(BaseModel):
    success: bool
    message: str
    job_id: str

class JobStatus(BaseModel):
    id: str
    filename: str
    status: str
    progress: int
    message: str
    timestamp: str
    metadata: Optional[dict] = None
    thumbnail: Optional[str] = None
    result: Optional[dict] = None

class UploadResponse(BaseModel):
    success: bool
    message: str
    job_id: Optional[str] = None

class StreamRequest(BaseModel):
    name: str
    rtsp_url: str

class StreamResponse(BaseModel):
    success: bool
    message: str
    stream_id: Optional[int] = None

# Helper functions
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_video_metadata(video_path: str) -> Optional[dict]:
    """Get basic metadata from video file"""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        cap.release()

        return {
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'duration_formatted': f"{int(duration//60)}:{int(duration%60):02d}"
        }
    except Exception as e:
        print(f"Error getting video metadata: {e}")
        return None

def generate_thumbnail(video_path: str, output_path: str, frame_number: int = 0) -> bool:
    """Generate a thumbnail from the video"""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        if frame_number == 0:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count > 0:
                frame_number = frame_count // 4

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return False

        height, width = frame.shape[:2]
        max_dim = 400
        if height > width:
            new_height = max_dim
            new_width = int(width * (max_dim / height))
        else:
            new_width = max_dim
            new_height = int(height * (max_dim / width))

        frame = cv2.resize(frame, (new_width, new_height))
        cv2.imwrite(output_path, frame)
        cap.release()
        return True
    except Exception as e:
        print(f"Error generating thumbnail: {e}")
        return False

def process_video_sync(job_id: str, video_path: str, threshold: float = None):
    """Process video using consecutive frame sequences with first violence inference time tracking"""

    if model is None:
        active_jobs[job_id]['status'] = 'error'
        active_jobs[job_id]['message'] = 'Model not loaded'
        return

    if threshold is None:
        threshold = DETECTION_THRESHOLD

    try:
        # Extract metadata
        metadata = get_video_metadata(video_path)
        if metadata is None:
            active_jobs[job_id]['status'] = 'error'
            active_jobs[job_id]['message'] = 'Could not read video file'
            return

        active_jobs[job_id]['metadata'] = metadata
        active_jobs[job_id]['status'] = 'processing'
        active_jobs[job_id]['progress'] = 5
        active_jobs[job_id]['message'] = 'Extracting frames'

        # Generate thumbnail
        thumbnail_path = os.path.join(RESULTS_FOLDER, f"{job_id}_thumbnail.jpg")
        generate_thumbnail(video_path, thumbnail_path)
        active_jobs[job_id]['thumbnail'] = f"/api/results/{job_id}_thumbnail.jpg"

        # Extract consecutive frame sequences
        hop_seconds = 2.0
        sequences, timestamps = extract_consecutive_frame_sequences(
            video_path, sequence_length=16, hop_seconds=hop_seconds
        )
        
        if not sequences:
            active_jobs[job_id]['status'] = 'error'
            active_jobs[job_id]['message'] = 'Failed to extract frame sequences'
            return

        active_jobs[job_id]['progress'] = 30
        active_jobs[job_id]['message'] = 'Processing sequences'

        # Determine motion enhancement
        use_motion = hasattr(model, 'use_motion_enhancement') and model.use_motion_enhancement

        # Process each sequence independently with individual timing
        segments = []
        total_sequences = len(sequences)
        first_violence_inference_time = None  # Track first violence detection time
        total_inference_time = 0.0  # Track total processing time for reference
        
        for i, (sequence, (start_time, end_time)) in enumerate(zip(sequences, timestamps)):
            progress = 30 + int(60 * i / total_sequences)
            active_jobs[job_id]['progress'] = progress
            active_jobs[job_id]['message'] = f'Analyzing sequence {i+1}/{total_sequences}'
            
            # Preprocess this sequence
            processed_data = preprocess_frames(sequence, compute_flow=use_motion)
            
            # Make prediction on this sequence WITH individual timing
            is_violent, confidence, inference_time = predict_violence(
                model, processed_data, threshold, debug=False
            )
            
            # Add to total inference time for reference
            total_inference_time += inference_time
            
            if is_violent and confidence > threshold:
                # Store inference time of FIRST violent event detected
                if first_violence_inference_time is None:
                    first_violence_inference_time = inference_time
                    print(f"First violence detected at {start_time:.1f}s with inference time: {inference_time:.3f}s")
                
                segments.append({
                    'start': start_time,
                    'end': end_time,
                    'confidence': float(confidence),
                    'inference_time': inference_time,  # Store individual inference time
                    'start_formatted': f"{int(start_time//60)}:{int(start_time%60):02d}",
                    'end_formatted': f"{int(end_time//60)}:{int(end_time%60):02d}"
                })
                print(f"Violence detected in sequence {i+1}: {start_time:.1f}-{end_time:.1f}s, confidence: {confidence:.3f}, inference: {inference_time:.3f}s")

        # Merge close segments (within 1 second) while preserving first inference time
        if segments:
            merged_segments = [segments[0]]
            for segment in segments[1:]:
                prev = merged_segments[-1]
                if segment['start'] <= prev['end'] + 1.0:
                    prev['end'] = max(prev['end'], segment['end'])
                    prev['confidence'] = max(prev['confidence'], segment['confidence'])
                    prev['end_formatted'] = f"{int(prev['end']//60)}:{int(prev['end']%60):02d}"
                    # Keep the earlier inference time when merging
                    if segment['inference_time'] < prev['inference_time']:
                        prev['inference_time'] = segment['inference_time']
                else:
                    merged_segments.append(segment)
            segments = merged_segments

        # Calculate results
        has_violence = len(segments) > 0
        violence_duration = sum(seg['end'] - seg['start'] for seg in segments) if segments else 0
        overall_confidence = max(seg['confidence'] for seg in segments) if segments else 0
        
        duration = metadata['duration']
        violence_percentage = (violence_duration / duration * 100) if duration > 0 else 0

        # Use first violence inference time, fallback to average if no violence detected
        display_inference_time = first_violence_inference_time if first_violence_inference_time is not None else (total_inference_time / total_sequences if total_sequences > 0 else 0.0)

        print(f"Analysis complete: {len(segments)} violent segments, {violence_duration:.1f}s total, {violence_percentage:.1f}%")
        print(f"First violence inference time: {first_violence_inference_time:.3f}s" if first_violence_inference_time else "No violence detected")
        print(f"Total processing time: {total_inference_time:.3f}s for {total_sequences} sequences")

        # Build result
        result = {
            'job_id': job_id,
            'video_path': video_path,
            'filename': os.path.basename(video_path),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'metadata': metadata,
            'thumbnail': active_jobs[job_id]['thumbnail'],
            'model_info': {
                'architecture': 'X3D-M',
                'motion_enhancement': use_motion,
                'input_frames': 16,
                'input_resolution': '336x336',
                'analysis_method': 'consecutive_sequences',
                'hop_seconds': hop_seconds,
                'total_sequences_processed': total_sequences
            },
            'overall_result': {
                'is_fight': has_violence,
                'confidence': float(overall_confidence),
                'inference_time': display_inference_time,  # First violence inference time
                'first_violence_inference_time': first_violence_inference_time,  # Explicit field
                'total_processing_time': total_inference_time,  # Total time for reference
                'sequences_processed': total_sequences
            },
            'segments': segments,
            'has_violence': has_violence,
            'violence_duration': violence_duration,
            'violence_percentage': violence_percentage,
            'processing_stats': {
                'total_sequences': total_sequences,
                'violent_sequences': len(segments),
                'total_inference_time': total_inference_time,
                'first_violence_time': first_violence_inference_time,
                'avg_inference_per_sequence': total_inference_time / total_sequences if total_sequences > 0 else 0
            }
        }

        # Save events and results (existing code)
        try:
            process_and_save_events(job_id, result, video_path)
        except Exception as e:
            print(f"Error saving events for job {job_id}: {e}")

        result_path = os.path.join(RESULTS_FOLDER, f"{job_id}_result.json")
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)

        active_jobs[job_id]['status'] = 'completed'
        active_jobs[job_id]['progress'] = 100
        active_jobs[job_id]['message'] = 'Processing complete'
        active_jobs[job_id]['result'] = result

        results_history[job_id] = {
            'job_id': job_id,
            'filename': os.path.basename(video_path),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'has_violence': has_violence,
            'violence_duration': violence_duration,
            'violence_percentage': violence_percentage,
            'overall_confidence': float(overall_confidence),
            'model_type': 'X3D-M (Sequence Analysis)',
            'thumbnail': active_jobs[job_id]['thumbnail'],
            'first_violence_inference_time': first_violence_inference_time,
            'total_sequences': total_sequences
        }

        try:
            with open(os.path.join(RESULTS_FOLDER, 'history.json'), 'w') as f:
                json.dump(list(results_history.values()), f, indent=2)
        except Exception as e:
            print(f"Error saving history: {e}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        active_jobs[job_id]['status'] = 'error'
        active_jobs[job_id]['message'] = f'Error: {str(e)}'

# API Routes
@app.get("/")
async def root():
    return {"message": "Violence Detection API with Event Storage", "docs": "/docs", "status": "running"}

@app.get("/api/jobs", response_model=List[JobStatus])
async def get_all_jobs():
    """Get all active jobs"""
    return [
        JobStatus(
            id=job['id'],
            filename=job['filename'],
            status=job['status'],
            progress=job['progress'],
            message=job['message'],
            timestamp=job['timestamp'],
            metadata=job.get('metadata'),
            thumbnail=job.get('thumbnail'),
            result=job.get('result')
        )
        for job in active_jobs.values()
    ]

@app.get("/api/history")
async def get_history():
    """Get processing history"""
    # Reload history from file to ensure consistency
    await load_history_from_file()
    return {"history": list(results_history.values())}

@app.get("/api/stats")
async def get_dashboard_stats():
    """Get dashboard statistics that persist across restarts"""
    try:
        # Check if database is initialized
        if event_db is None:
            print("Database not initialized, returning default stats")
            return {
                'today': {'events': 0, 'processed': 0, 'violence_rate': 0, 'violence_duration': 0},
                'total': {'events': 0, 'processed': 0, 'violence_rate': 0, 'violence_duration': 0},
                'current': {
                    'active_jobs': sum(1 for job in active_jobs.values() if job['status'] in ['queued', 'processing']),
                    'websocket_connections': len(manager.active_connections)
                },
                'recent_events': []
            }

        stats = event_db.get_stats()

        # Add current active jobs to stats
        active_count = sum(1 for job in active_jobs.values()
                           if job['status'] in ['queued', 'processing'])

        stats['current'] = {
            'active_jobs': active_count,
            'websocket_connections': len(manager.active_connections)
        }

        return stats
    except Exception as e:
        print(f"Error getting stats: {e}")
        return {
            'error': str(e),
            'today': {'events': 0, 'processed': 0, 'violence_rate': 0, 'violence_duration': 0},
            'total': {'events': 0, 'processed': 0, 'violence_rate': 0, 'violence_duration': 0},
            'current': {'active_jobs': 0, 'websocket_connections': 0},
            'recent_events': []
        }

@app.get("/api/events")
async def get_events(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = 50
):
    """Get violence events with optional date filtering"""
    try:
        if not start_date:
            # Default to last 7 days
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        events = event_db.get_events_by_date_range(start_date, end_date)

        # Limit results
        if limit:
            events = events[:limit]

        return {
            'events': events,
            'count': len(events),
            'date_range': {
                'start': start_date,
                'end': end_date
            }
        }
    except Exception as e:
        return {
            'error': str(e),
            'events': [],
            'count': 0
        }

@app.get("/api/events/{event_id}")
async def get_event_details(event_id: int):
    """Get detailed information about a specific event"""
    try:
        conn = sqlite3.connect(event_db.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM violence_events WHERE id = ?', (event_id,))
        event = cursor.fetchone()
        conn.close()

        if not event:
            raise HTTPException(status_code=404, detail="Event not found")

        return {
            'id': event[0],
            'timestamp': event[1],
            'source_type': event[2],
            'source_id': event[3],
            'filename': event[4],
            'start_time': event[5],
            'end_time': event[6],
            'duration': event[7],
            'confidence': event[8],
            'thumbnail_path': event[9],
            'clip_path': event[10],
            'metadata': json.loads(event[11]) if event[11] else {}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload", response_model=UploadResponse)
@limiter.limit("5/minute")  # 5 uploads per minute per IP
async def upload_file(
    request: Request,
    file: Optional[UploadFile] = File(None),
    video_path: Optional[str] = Form(None)
):
    """Handle file upload or local path with rate limiting and resource management"""

    # Check if too many active jobs (limit to 3 like before)
    active_count = sum(1 for job in active_jobs.values()
                       if job['status'] in ['queued', 'processing'])
    if active_count >= 3:
        raise HTTPException(
            status_code=429,
            detail=f"Too many active jobs ({active_count}/3). Please wait."
        )

    if not file and not video_path:
        raise HTTPException(status_code=400, detail="No file or path provided")

    job_id = str(uuid.uuid4())

    if file and file.filename:
        # Handle file upload
        if not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail="Invalid file type")

        # Read and check file size
        contents = await file.read()
        if len(contents) > MAX_CONTENT_LENGTH:
            raise HTTPException(status_code=413, detail=f"File too large. Max size: {MAX_CONTENT_LENGTH // (1024*1024)}MB")

        # Save the file using fixed method
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{filename}")

        with open(file_path, "wb") as f:
            f.write(contents)

        # Note: File cleanup removed for simplicity - can be added back later

        # Create job
        active_jobs[job_id] = {
            'id': job_id,
            'file_path': file_path,
            'filename': filename,
            'status': 'queued',
            'progress': 0,
            'message': 'Queued for processing',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        # Start processing using threading (like working Flask version)
        print(f"Starting thread for job {job_id}")
        threading.Thread(target=process_video_sync, args=(job_id, file_path)).start()
        print(f"Thread started for job {job_id}")

        return UploadResponse(
            success=True,
            message="File uploaded successfully",
            job_id=job_id
        )

    elif video_path and video_path.strip():
        # Handle local path
        video_path = video_path.strip()
        if not os.path.exists(video_path):
            raise HTTPException(status_code=400, detail="File not found at specified path")

        # Check file size for local files too
        file_size = os.path.getsize(video_path)
        if file_size > MAX_CONTENT_LENGTH:
            raise HTTPException(status_code=413, detail=f"File too large. Max size: {MAX_CONTENT_LENGTH // (1024*1024)}MB")

        # Check if it's a valid video file
        _, ext = os.path.splitext(video_path)
        if ext.lower()[1:] not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Invalid file type")

        # Create job
        active_jobs[job_id] = {
            'id': job_id,
            'file_path': video_path,
            'filename': os.path.basename(video_path),
            'status': 'queued',
            'progress': 0,
            'message': 'Queued for processing',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        # Start processing using threading (like working Flask version)
        print(f"Starting thread for job {job_id} (local path)")
        threading.Thread(target=process_video_sync, args=(job_id, video_path)).start()
        print(f"Thread started for job {job_id} (local path)")

        return UploadResponse(
            success=True,
            message="Video path submitted successfully",
            job_id=job_id
        )

    raise HTTPException(status_code=400, detail="No file or valid path provided")

@app.get("/api/job/{job_id}", response_model=JobStatus)
async def get_job(job_id: str):
    """Get job status"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = active_jobs[job_id]
    return JobStatus(
        id=job['id'],
        filename=job['filename'],
        status=job['status'],
        progress=job['progress'],
        message=job['message'],
        timestamp=job['timestamp'],
        metadata=job.get('metadata'),
        thumbnail=job.get('thumbnail'),
        result=job.get('result')
    )

@app.get("/api/result/{job_id}")
async def get_result(job_id: str):
    """Get detailed result for a job"""
    result_path = os.path.join(RESULTS_FOLDER, f"{job_id}_result.json")
    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Result not found")

    try:
        with open(result_path, 'r') as f:
            result = json.load(f)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading result: {str(e)}")

@app.get("/api/results/{filename}")
async def get_result_file(filename: str, request: Request):
    """Serve files from the results folder - enhanced version with proper headers"""
    # Security check: prevent directory traversal
    if '..' in filename or '/' in filename or '\\' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = os.path.join(RESULTS_FOLDER, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    # Get file stats for content-length and range support
    stat_result = os.stat(file_path)
    file_size = stat_result.st_size

    # Determine content type
    file_ext = os.path.splitext(file_path)[1].lower()
    
    content_type_map = {
        '.mp4': 'video/mp4',
        '.avi': 'video/x-msvideo', 
        '.mov': 'video/quicktime',
        '.mkv': 'video/x-matroska',
        '.webm': 'video/webm',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.json': 'application/json'
    }
    
    media_type = content_type_map.get(file_ext, 'application/octet-stream')
    
    # Handle range requests for video seeking
    range_header = request.headers.get('range')
    
    if range_header and file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        # Parse range header
        range_match = re.match(r'bytes=(\d+)-(\d*)', range_header)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2)) if range_match.group(2) else file_size - 1
            
            # Ensure valid range
            start = min(start, file_size - 1)
            end = min(end, file_size - 1)
            
            if start <= end:
                from fastapi.responses import Response
                
                def iterfile(file_path: str, start: int, end: int, chunk_size: int = 8192):
                    with open(file_path, 'rb') as file:
                        file.seek(start)
                        remaining = end - start + 1
                        while remaining:
                            chunk = file.read(min(chunk_size, remaining))
                            if not chunk:
                                break
                            remaining -= len(chunk)
                            yield chunk
                
                content_length = end - start + 1
                headers = {
                    'Accept-Ranges': 'bytes',
                    'Content-Range': f'bytes {start}-{end}/{file_size}',
                    'Content-Length': str(content_length),
                    'Content-Type': media_type,
                    'Cache-Control': 'no-cache',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Range'
                }
                
                return StreamingResponse(
                    iterfile(file_path, start, end),
                    status_code=206,
                    headers=headers,
                    media_type=media_type  # or 'video/mp4' for video files
                )
    
    # Regular file response
    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-cache",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Range, Content-Type",
        "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS"
    }
    
    return FileResponse(
        file_path, 
        media_type=media_type,
        headers=headers
    )

@app.get("/api/results/clips/{filename}")
async def get_clip_file(filename: str, request: Request):
    """Serve clip files from the clips folder with range support"""
    # Security check
    if '..' in filename or '/' in filename or '\\' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    clips_folder = os.path.join(RESULTS_FOLDER, "clips")
    file_path = os.path.join(clips_folder, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Clip not found")

    # Get file stats
    stat_result = os.stat(file_path)
    file_size = stat_result.st_size

    # Handle range requests
    range_header = request.headers.get('range')
    
    if range_header:
        # Parse range header
        import re
        range_match = re.match(r'bytes=(\d+)-(\d*)', range_header)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2)) if range_match.group(2) else file_size - 1
            
            start = min(start, file_size - 1)
            end = min(end, file_size - 1)
            
            if start <= end:
                from fastapi.responses import Response
                
                def iterfile(file_path: str, start: int, end: int, chunk_size: int = 8192):
                    with open(file_path, 'rb') as file:
                        file.seek(start)
                        remaining = end - start + 1
                        while remaining:
                            chunk = file.read(min(chunk_size, remaining))
                            if not chunk:
                                break
                            remaining -= len(chunk)
                            yield chunk
                
                content_length = end - start + 1
                headers = {
                    'Accept-Ranges': 'bytes',
                    'Content-Range': f'bytes {start}-{end}/{file_size}',
                    'Content-Length': str(content_length),
                    'Content-Type': 'video/mp4',
                    'Cache-Control': 'no-cache',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Range'
                }
                
                return StreamingResponse(
                    iterfile(file_path, start, end),
                    status_code=206,
                    headers=headers,
                    media_type=media_type  # or 'video/mp4' for video files
                )

    # Regular response
    return FileResponse(
        file_path, 
        media_type='video/mp4',
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
            "Access-Control-Allow-Origin": "*"
        }
    )

@app.get("/api/results/stream_thumbnails/{filename}")
async def get_stream_thumbnail(filename: str):
    """Serve stream thumbnail files"""
    if '..' in filename or '/' in filename or '\\' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    thumbnails_folder = os.path.join(RESULTS_FOLDER, "stream_thumbnails")
    file_path = os.path.join(thumbnails_folder, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    return FileResponse(file_path)

@app.post("/api/streams", response_model=StreamResponse)
async def add_stream(stream_request: StreamRequest):
    """Add a new RTSP stream"""
    try:
        # Validate URL format
        processor = RTSPStreamProcessor(0, stream_request.rtsp_url, stream_request.name)
        if not processor.validate_rtsp_url(stream_request.rtsp_url):
            raise HTTPException(status_code=400, detail="Invalid RTSP URL format")

        # Create stream record
        stream = RTSPStream(
            name=stream_request.name,
            rtsp_url=stream_request.rtsp_url,
            status='inactive'
        )

        stream_id = stream_db.add_stream(stream)

        return StreamResponse(
            success=True,
            message="Stream added successfully",
            stream_id=stream_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/streams")
async def get_streams():
    """Get all RTSP streams"""
    try:
        streams = stream_db.get_streams()

        # Add active status from memory
        for stream in streams:
            if stream['id'] in active_streams:
                stream['status'] = 'active'
                stream['live_data'] = {
                    'frame': active_streams[stream['id']]['processor'].get_frame_base64()
                }

        return {"streams": streams}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/streams/{stream_id}/start")
async def start_stream(stream_id: int):
    """Start an RTSP stream"""
    try:
        if stream_id in active_streams:
            return {"success": False, "message": "Stream already active"}

        # Get stream from database
        streams = stream_db.get_streams()
        stream_data = next((s for s in streams if s['id'] == stream_id), None)

        if not stream_data:
            raise HTTPException(status_code=404, detail="Stream not found")

        # Start stream processor
        processor = RTSPStreamProcessor(stream_id, stream_data['rtsp_url'], stream_data['name'])

        if processor.start_stream():
            stream_db.update_stream_status(stream_id, 'active')
            return {"success": True, "message": "Stream started"}
        else:
            stream_db.update_stream_status(stream_id, 'error')
            return {"success": False, "message": "Failed to start stream"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/streams/{stream_id}/stop")
async def stop_stream(stream_id: int):
    """Stop an RTSP stream"""
    try:
        if stream_id in active_streams:
            active_streams[stream_id]['processor'].stop_stream()
            stream_db.update_stream_status(stream_id, 'inactive')
            return {"success": True, "message": "Stream stopped"}
        else:
            return {"success": False, "message": "Stream not active"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/streams/{stream_id}")
async def delete_stream(stream_id: int):
    """Delete an RTSP stream"""
    try:
        # Stop stream if active
        if stream_id in active_streams:
            active_streams[stream_id]['processor'].stop_stream()

        # Delete from database
        stream_db.delete_stream(stream_id)

        return {"success": True, "message": "Stream deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/streams/{stream_id}/frame")
async def get_stream_frame(stream_id: int):
    """Get current frame from active stream"""
    try:
        if stream_id not in active_streams:
            raise HTTPException(status_code=404, detail="Stream not active")

        frame_data = active_streams[stream_id]['processor'].get_frame_base64()
        if frame_data:
            return {"frame": frame_data, "status": "active"}
        else:
            return {"frame": None, "status": "no_frame"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_system_status():
    """Get system status and statistics"""
    active_count = sum(1 for job in active_jobs.values()
                       if job['status'] in ['queued', 'processing'])
    completed_count = sum(1 for job in active_jobs.values()
                          if job['status'] == 'completed')
    error_count = sum(1 for job in active_jobs.values()
                      if job['status'] == 'error')

    return {
        "system_status": "running",
        "model_loaded": model is not None,
        "active_jobs": active_count,
        "completed_jobs": completed_count,
        "error_jobs": error_count,
        "total_jobs": len(active_jobs),
        "history_count": len(results_history),
        "max_concurrent_jobs": 3,
        "websocket_connections": len(manager.active_connections),
        "database_connected": True
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time job updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle any incoming messages
            data = await websocket.receive_text()
            # You can add message handling here if needed
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.get("/api/stream-events")
async def get_stream_events(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    stream_id: Optional[str] = None,
    limit: Optional[int] = 50,
    min_confidence: Optional[float] = 0.0
):
    """Get violence events from live streams with filtering"""
    try:
        if not start_date:
            # Default to last 24 hours
            start_date = (datetime.now() - timedelta(hours=24)).strftime('%Y-%m-%d %H:%M:%S')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        conn = sqlite3.connect(event_db.db_path)
        cursor = conn.cursor()

        # Build query with filters
        query = '''
            SELECT * FROM violence_events 
            WHERE timestamp BETWEEN ? AND ?
            AND source_type = 'stream'
        '''
        params = [start_date, end_date]
        
        if stream_id and stream_id != 'all':
            query += ' AND source_id = ?'
            params.append(stream_id)
            
        if min_confidence > 0:
            query += ' AND confidence >= ?'
            params.append(min_confidence)
            
        query += ' ORDER BY timestamp DESC'
        
        if limit:
            query += ' LIMIT ?'
            params.append(limit)

        cursor.execute(query, params)
        events = cursor.fetchall()
        conn.close()

        # Format events
        formatted_events = []
        for event in events:
            formatted_events.append({
                'id': event[0],
                'timestamp': event[1],
                'source_type': event[2],
                'source_id': event[3],
                'filename': event[4],
                'start_time': event[5],
                'end_time': event[6],
                'duration': event[7],
                'confidence': event[8],
                'thumbnail_path': event[9],
                'clip_path': event[10],
                'metadata': json.loads(event[11]) if event[11] else {}
            })

        return {
            'events': formatted_events,
            'count': len(formatted_events),
            'filters': {
                'start_date': start_date,
                'end_date': end_date,
                'stream_id': stream_id,
                'min_confidence': min_confidence
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/streams/{stream_id}/events")
async def get_stream_specific_events(
    stream_id: int,
    limit: Optional[int] = 20,
    hours: Optional[int] = 24
):
    """Get events for a specific stream"""
    try:
        start_date = (datetime.now() - timedelta(hours=hours)).strftime('%Y-%m-%d %H:%M:%S')
        end_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        conn = sqlite3.connect(event_db.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM violence_events 
            WHERE source_type = 'stream' 
            AND source_id = ?
            AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (str(stream_id), start_date, end_date, limit))

        events = cursor.fetchall()
        conn.close()

        formatted_events = []
        for event in events:
            formatted_events.append({
                'id': event[0],
                'timestamp': event[1],
                'source_type': event[2],
                'source_id': event[3],
                'filename': event[4],
                'start_time': event[5],
                'end_time': event[6],
                'duration': event[7],
                'confidence': event[8],
                'thumbnail_path': event[9],
                'clip_path': event[10],
                'metadata': json.loads(event[11]) if event[11] else {}
            })

        return {
            'stream_id': stream_id,
            'events': formatted_events,
            'count': len(formatted_events),
            'time_range_hours': hours
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stream-stats")
async def get_stream_statistics():
    """Get comprehensive stream statistics"""
    try:
        if not event_db or not stream_db:
            return {
                'error': 'Database not initialized',
                'total_streams': 0,
                'active_streams': 0,
                'total_events': 0,
                'events_24h': 0,
                'avg_confidence': 0,
                'top_streams': []
            }

        # Get stream counts
        streams = stream_db.get_streams()
        total_streams = len(streams)
        active_streams = len([s for s in streams if s['status'] == 'active'])

        # Get event statistics
        conn = sqlite3.connect(event_db.db_path)
        cursor = conn.cursor()

        # Total events from streams
        cursor.execute("SELECT COUNT(*) FROM violence_events WHERE source_type = 'stream'")
        total_events = cursor.fetchone()[0]

        # Events in last 24 hours
        yesterday = (datetime.now() - timedelta(hours=24)).strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('''
            SELECT COUNT(*) FROM violence_events 
            WHERE source_type = 'stream' AND timestamp >= ?
        ''', (yesterday,))
        events_24h = cursor.fetchone()[0]

        # Average confidence
        cursor.execute("SELECT AVG(confidence) FROM violence_events WHERE source_type = 'stream'")
        avg_confidence_result = cursor.fetchone()[0]
        avg_confidence = avg_confidence_result if avg_confidence_result else 0

        # Top streams by event count
        cursor.execute('''
            SELECT source_id, COUNT(*) as event_count, AVG(confidence) as avg_conf
            FROM violence_events 
            WHERE source_type = 'stream'
            GROUP BY source_id
            ORDER BY event_count DESC
            LIMIT 5
        ''')
        top_streams_data = cursor.fetchall()

        top_streams = []
        for stream_data in top_streams_data:
            source_id = stream_data[0]
            event_count = stream_data[1]
            avg_conf = stream_data[2]
            
            # Find stream name
            stream_info = next((s for s in streams if str(s['id']) == source_id), None)
            stream_name = stream_info['name'] if stream_info else f"Stream {source_id}"
            
            top_streams.append({
                'stream_id': source_id,
                'stream_name': stream_name,
                'event_count': event_count,
                'avg_confidence': round(avg_conf, 3) if avg_conf else 0
            })

        # Events by hour (last 24 hours)
        cursor.execute('''
            SELECT 
                strftime('%H', timestamp) as hour,
                COUNT(*) as count
            FROM violence_events 
            WHERE source_type = 'stream' 
            AND timestamp >= ?
            GROUP BY hour
            ORDER BY hour
        ''', (yesterday,))
        hourly_data = cursor.fetchall()

        hourly_events = []
        for hour_data in hourly_data:
            hourly_events.append({
                'hour': int(hour_data[0]),
                'count': hour_data[1]
            })

        conn.close()

        return {
            'total_streams': total_streams,
            'active_streams': active_streams,
            'total_events': total_events,
            'events_24h': events_24h,
            'avg_confidence': round(avg_confidence, 3) if avg_confidence else 0,
            'top_streams': top_streams,
            'hourly_events': hourly_events,
            'detection_rate': round((events_24h / active_streams), 2) if active_streams > 0 else 0
        }

    except Exception as e:
        print(f"Error getting stream statistics: {e}")
        return {
            'error': str(e),
            'total_streams': 0,
            'active_streams': 0,
            'total_events': 0,
            'events_24h': 0,
            'avg_confidence': 0,
            'top_streams': [],
            'hourly_events': []
        }

@app.delete("/api/stream-events/{event_id}")
async def delete_stream_event(event_id: int):
    """Delete a specific stream event"""
    try:
        conn = sqlite3.connect(event_db.db_path)
        cursor = conn.cursor()

        # Get event details first to clean up files
        cursor.execute('SELECT * FROM violence_events WHERE id = ?', (event_id,))
        event = cursor.fetchone()
        
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")

        # Delete associated files if they exist
        if event[9]:  # thumbnail_path
            thumbnail_file = event[9].replace('/api/results/', '')
            thumbnail_full_path = os.path.join(RESULTS_FOLDER, thumbnail_file)
            if os.path.exists(thumbnail_full_path):
                try:
                    os.remove(thumbnail_full_path)
                except:
                    pass

        if event[10]:  # clip_path
            clip_file = event[10].replace('/api/results/', '')
            clip_full_path = os.path.join(RESULTS_FOLDER, clip_file)
            if os.path.exists(clip_full_path):
                try:
                    os.remove(clip_full_path)
                except:
                    pass

        # Delete from database
        cursor.execute('DELETE FROM violence_events WHERE id = ?', (event_id,))
        conn.commit()
        conn.close()

        return {"success": True, "message": "Event deleted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/results/stream_clips/{filename}")
async def get_stream_clip(filename: str, request: Request):
    """Serve stream clip files with range support"""
    if '..' in filename or '/' in filename or '\\' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    clips_folder = os.path.join(RESULTS_FOLDER, "stream_clips")
    file_path = os.path.join(clips_folder, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Clip not found")

    # Get file stats
    stat_result = os.stat(file_path)
    file_size = stat_result.st_size

    # Handle range requests
    range_header = request.headers.get('range')
    
    if range_header:
        # Parse range header
        import re
        range_match = re.match(r'bytes=(\d+)-(\d*)', range_header)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2)) if range_match.group(2) else file_size - 1
            
            start = min(start, file_size - 1)
            end = min(end, file_size - 1)
            
            if start <= end:
                from fastapi.responses import Response
                
                def iterfile(file_path: str, start: int, end: int, chunk_size: int = 8192):
                    with open(file_path, 'rb') as file:
                        file.seek(start)
                        remaining = end - start + 1
                        while remaining:
                            chunk = file.read(min(chunk_size, remaining))
                            if not chunk:
                                break
                            remaining -= len(chunk)
                            yield chunk
                
                content_length = end - start + 1
                headers = {
                    'Accept-Ranges': 'bytes',
                    'Content-Range': f'bytes {start}-{end}/{file_size}',
                    'Content-Length': str(content_length),
                    'Content-Type': 'video/mp4',
                    'Cache-Control': 'no-cache',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Range'
                }
                
                return StreamingResponse(
                    iterfile(file_path, start, end),
                    status_code=206,
                    headers=headers,
                    media_type=media_type  # or 'video/mp4' for video files
                )

    # Regular response
    return FileResponse(
        file_path, 
        media_type='video/mp4',
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Range, Content-Type",
            "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS"
        }
    )

@app.get("/api/uploads/{filename}")
async def get_uploaded_file(filename: str, request: Request):
    """Serve uploaded video files with proper range support"""
    # Security check: prevent directory traversal
    if '..' in filename or '/' in filename or '\\' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Look for the file in uploads folder
    file_path = None
    
    # Check if file exists directly
    direct_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(direct_path):
        file_path = direct_path
    else:
        # Search for files that contain the filename (in case of UUID prefix)
        try:
            for file in os.listdir(UPLOAD_FOLDER):
                if filename in file or file.endswith(filename):
                    file_path = os.path.join(UPLOAD_FOLDER, file)
                    break
        except Exception as e:
            print(f"Error searching for file {filename}: {e}")
    
    if not file_path or not os.path.exists(file_path):
        print(f"Video file not found: {filename}")
        print(f"Searched in: {UPLOAD_FOLDER}")
        print(f"Direct path tried: {direct_path}")
        try:
            print(f"Available files: {os.listdir(UPLOAD_FOLDER)}")
        except:
            print("Could not list upload folder contents")
        raise HTTPException(status_code=404, detail="Video file not found")

    # Get file stats
    stat_result = os.stat(file_path)
    file_size = stat_result.st_size
    
    print(f"Serving video file: {file_path} (size: {file_size} bytes)")

    # Determine content type based on file extension
    file_ext = os.path.splitext(file_path)[1].lower()
    content_type_map = {
        '.mp4': 'video/mp4',
        '.avi': 'video/x-msvideo', 
        '.mov': 'video/quicktime',
        '.mkv': 'video/x-matroska',
        '.webm': 'video/webm'
    }
    
    media_type = content_type_map.get(file_ext, 'video/mp4')
    print(f"Content type: {media_type}")
    
    # Handle range requests for video seeking
    range_header = request.headers.get('range')
    print(f"Range header: {range_header}")
    
    if range_header:
        # Parse range header (e.g., "bytes=0-1023")
        import re
        range_match = re.match(r'bytes=(\d+)-(\d*)', range_header)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2)) if range_match.group(2) else file_size - 1
            
            # Ensure valid range
            start = min(start, file_size - 1)
            end = min(end, file_size - 1)
            
            print(f"Range request: {start}-{end} of {file_size}")
            
            if start <= end:
                from fastapi.responses import Response
                
                def iterfile(file_path: str, start: int, end: int, chunk_size: int = 8192):
                    """Generator function to stream file content in chunks"""
                    with open(file_path, 'rb') as file:
                        file.seek(start)
                        remaining = end - start + 1
                        while remaining:
                            chunk = file.read(min(chunk_size, remaining))
                            if not chunk:
                                break
                            remaining -= len(chunk)
                            yield chunk
                
                content_length = end - start + 1
                headers = {
                    'Accept-Ranges': 'bytes',
                    'Content-Range': f'bytes {start}-{end}/{file_size}',
                    'Content-Length': str(content_length),
                    'Content-Type': media_type,
                    'Cache-Control': 'no-cache',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Range, Content-Type',
                    'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS'
                }
                
                return StreamingResponse(
                    iterfile(file_path, start, end),
                    status_code=206,
                    headers=headers,
                    media_type=media_type  # or 'video/mp4' for video files
                )
    
    # Regular file response (no range request)
    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-cache",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Range, Content-Type",
        "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS"
    }
    
    print(f"Serving full file with headers: {headers}")
    
    return FileResponse(
        file_path, 
        media_type=media_type,
        headers=headers
    )

@app.get("/api/stream-event/{event_id}")
async def get_stream_event_result(event_id: int):
    """Get stream event data formatted for ResultsViewer compatibility"""
    try:
        if not event_db:
            raise HTTPException(status_code=500, detail="Database not initialized")
            
        conn = sqlite3.connect(event_db.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM violence_events WHERE id = ? AND source_type = "stream"', (event_id,))
        event = cursor.fetchone()
        conn.close()
        
        if not event:
            raise HTTPException(status_code=404, detail="Stream event not found")
        
        # Parse metadata if it exists
        metadata_dict = {}
        if event[11]:  # metadata column
            try:
                metadata_dict = json.loads(event[11])
            except:
                metadata_dict = {}
        
        # Get stream information
        stream_info = None
        if stream_db:
            streams = stream_db.get_streams()
            stream_info = next((s for s in streams if str(s['id']) == event[3]), None)
        
        stream_name = stream_info['name'] if stream_info else event[4]
        
        # Format duration
        duration = event[7]  # duration column
        duration_formatted = f"{int(duration//60)}:{int(duration%60):02d}"
        
        # Create segments (live stream events typically have one segment)
        segments = [{
            'start': event[5],  # start_time
            'end': event[6],    # end_time
            'confidence': float(event[8]),  # confidence
            'inference_time': metadata_dict.get('detection_interval', 3.0),  # Individual inference time
            'start_formatted': f"{int(event[5]//60)}:{int(event[5]%60):02d}",
            'end_formatted': f"{int(event[6]//60)}:{int(event[6]%60):02d}"
        }]
        
        # Extract timing information from metadata
        detection_interval = metadata_dict.get('detection_interval', 3.0)
        
        # Format result to match ResultsViewer expectations
        result = {
            'job_id': f"stream_event_{event[0]}",
            'filename': stream_name,
            'timestamp': event[1],  # timestamp
            'has_violence': True,   # Stream events are always violence detections
            'video_path': event[10] if event[10] else None,  # clip_path
            'thumbnail': event[9] if event[9] else None,     # thumbnail_path
            
            # Overall result info with enhanced timing
            'overall_result': {
                'is_fight': True,
                'confidence': float(event[8]),
                'inference_time': detection_interval,  # Main inference time (for compatibility)
                'first_violence_inference_time': detection_interval,  # Same as inference_time for streams
                'total_processing_time': detection_interval,  # Same for single detection
                'sequences_processed': 1  # Stream events are single detections
            },
            
            # Segments data
            'segments': segments,
            'violence_duration': duration,
            'violence_percentage': 100.0,  # Assume entire clip is violent
            
            # Metadata for video player
            'metadata': {
                'duration': duration,
                'duration_formatted': duration_formatted,
                'width': metadata_dict.get('frame_width', 640),  # Default values
                'height': metadata_dict.get('frame_height', 480),
                'fps': metadata_dict.get('fps', 4.0),  # Typical for event clips
                'frame_count': int(duration * metadata_dict.get('fps', 4.0)),
                'source_type': 'live_stream'
            },
            
            # Model information with enhanced fields
            'model_info': {
                'architecture': 'X3D-M (Live Stream)',
                'motion_enhancement': metadata_dict.get('motion_enhancement', True),
                'input_frames': metadata_dict.get('temporal_length', 16),
                'input_resolution': f"{metadata_dict.get('model_input_size', [336, 336])[0]}x{metadata_dict.get('model_input_size', [336, 336])[1]}",
                'analysis_method': 'real_time_stream',
                'hop_seconds': metadata_dict.get('detection_interval', 3.0),
                'total_sequences_processed': 1,  # Single stream detection
                'source_stream': stream_name,
                'stream_id': event[3]
            },
            
            # Stream-specific metadata
            'stream_metadata': {
                'stream_id': event[3],
                'stream_name': stream_name,
                'rtsp_url': stream_info['rtsp_url'] if stream_info else 'N/A',
                'detection_type': 'real_time_monitoring',
                'event_id': event[0],
                'pipeline_version': metadata_dict.get('pipeline_version', 'stream_v1')
            },
            
            # Processing statistics (new section for consistency)
            'processing_stats': {
                'total_sequences': 1,
                'violent_sequences': 1,
                'total_inference_time': detection_interval,
                'first_violence_time': detection_interval,
                'avg_inference_per_sequence': detection_interval
            }
        }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting stream event {event_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.options("/api/uploads/{filename}")
@app.options("/api/results/{filename}")  
@app.options("/api/results/clips/{filename}")
@app.options("/api/results/stream_clips/{filename}")
async def handle_video_options():
    """Handle CORS preflight requests for video endpoints"""
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
            "Access-Control-Allow-Headers": "Range, Content-Type, Authorization",
            "Access-Control-Max-Age": "3600"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)