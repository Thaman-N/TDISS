from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Request
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
from torch_detection import load_violence_detection_model, extract_frames, preprocess_frames, predict_violence

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

# Configuration
UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB
MODEL_PATH = r"trainingpipeline\checkpoints\best_model.pth"
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
    global model, event_db
    try:
        print("Starting up Violence Detection API...")
        
        # Initialize database
        print("Initializing database...")
        event_db = EventDatabase()
        print("Database initialized successfully")
        
        print("Loading model...")
        model, _ = load_violence_detection_model(MODEL_PATH)
        print("Model loaded successfully")
        
        # Start cleanup task
        asyncio.create_task(periodic_cleanup())
        print("Periodic cleanup task started")
        
        # Load existing history
        await load_history_from_file()
        print("History loaded")
        
        print("Violence Detection API started successfully")
        
    except Exception as e:
        print(f"Error during startup: {e}")
        import traceback
        traceback.print_exc()
        raise

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
    """Process a video for violence detection using X3D model - SYNC version like Flask"""
    
    print(f"Starting process_video_sync for job {job_id}")
    
    if model is None:
        print(f"Model not loaded for job {job_id}")
        active_jobs[job_id]['status'] = 'error'
        active_jobs[job_id]['message'] = 'Model not loaded'
        # Can't await in sync function, so we'll skip websocket updates for now
        return
    
    if threshold is None:
        threshold = DETECTION_THRESHOLD
    
    try:
        print(f"Starting video processing for job {job_id}")
        
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
        
        # Extract frames
        frames = extract_frames(video_path)
        if frames is None or len(frames) == 0:
            active_jobs[job_id]['status'] = 'error'
            active_jobs[job_id]['message'] = 'Failed to extract frames'
            return
        
        active_jobs[job_id]['progress'] = 30
        active_jobs[job_id]['message'] = 'Preprocessing frames'
        
        # Determine if model uses motion enhancement
        use_motion = hasattr(model, 'use_motion_enhancement') and model.use_motion_enhancement
        
        # Preprocess frames
        processed_data = preprocess_frames(frames, compute_flow=use_motion)
        
        active_jobs[job_id]['progress'] = 50
        active_jobs[job_id]['message'] = 'Running violence detection'
        
        # Make prediction on the full video
        is_fight, confidence, inference_time = predict_violence(model, processed_data, threshold, debug=True)
        
        # Process video with sliding window for detailed timeline
        segments = []
        window_size = 16
        stride = 4
        
        total_frames = len(frames)
        duration = metadata['duration']
        
        if total_frames > window_size:
            window_count = (total_frames - window_size) // stride + 1
            
            for i in range(window_count):
                active_jobs[job_id]['progress'] = 50 + int(40 * i / window_count)
                
                start_idx = i * stride
                end_idx = start_idx + window_size
                
                window_frames = frames[start_idx:end_idx]
                if len(window_frames) == window_size:
                    window_data = preprocess_frames(window_frames, compute_flow=use_motion)
                    segment_threshold = threshold + 0.1
                    is_violent, prob, _ = predict_violence(model, window_data, segment_threshold, debug=False)
                    
                    if is_violent and prob > segment_threshold:
                        start_time = (start_idx / total_frames) * duration
                        end_time = (end_idx / total_frames) * duration
                        
                        segments.append({
                            'start': start_time,
                            'end': end_time,
                            'confidence': float(prob),
                            'start_formatted': f"{int(start_time//60)}:{int(start_time%60):02d}",
                            'end_formatted': f"{int(end_time//60)}:{int(end_time%60):02d}"
                        })
        else:
            if is_fight and confidence > threshold:
                segments.append({
                    'start': 0,
                    'end': duration,
                    'confidence': float(confidence),
                    'start_formatted': "0:00",
                    'end_formatted': metadata['duration_formatted']
                })
        
        # Merge overlapping segments
        if segments:
            merged_segments = [segments[0]]
            for segment in segments[1:]:
                prev = merged_segments[-1]
                if segment['start'] <= prev['end'] + 2.0:
                    prev['end'] = max(prev['end'], segment['end'])
                    prev['confidence'] = max(prev['confidence'], segment['confidence'])
                    prev['end_formatted'] = f"{int(prev['end']//60)}:{int(prev['end']%60):02d}"
                else:
                    merged_segments.append(segment)
            segments = merged_segments
        
        # Save final results
        result = {
            'job_id': job_id,
            'video_path': video_path,
            'filename': os.path.basename(video_path),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'metadata': metadata,
            'thumbnail': active_jobs[job_id]['thumbnail'],
            'model_info': {
                'architecture': 'X3D-S',
                'motion_enhancement': use_motion,
                'input_frames': 16,
                'input_resolution': '224x224'
            },
            'overall_result': {
                'is_fight': is_fight,
                'confidence': float(confidence),
                'inference_time': inference_time
            },
            'segments': segments,
            'has_violence': len(segments) > 0,
            'violence_duration': sum(seg['end'] - seg['start'] for seg in segments) if segments else 0,
            'violence_percentage': (sum(seg['end'] - seg['start'] for seg in segments) / duration * 100) if segments and duration > 0 else 0
        }
        
        # Save events to database
        try:
            process_and_save_events(job_id, result, video_path)
        except Exception as e:
            print(f"Error saving events for job {job_id}: {e}")
        
        # Save result to JSON file
        result_path = os.path.join(RESULTS_FOLDER, f"{job_id}_result.json")
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Update active job and add to history
        active_jobs[job_id]['status'] = 'completed'
        active_jobs[job_id]['progress'] = 100
        active_jobs[job_id]['message'] = 'Processing complete'
        active_jobs[job_id]['result'] = result
        
        # Add to history
        results_history[job_id] = {
            'job_id': job_id,
            'filename': os.path.basename(video_path),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'has_violence': len(segments) > 0,
            'violence_duration': result['violence_duration'],
            'violence_percentage': result['violence_percentage'],
            'overall_confidence': float(confidence),
            'model_type': 'X3D-S',
            'thumbnail': active_jobs[job_id]['thumbnail']
        }
        
        # Save history to file
        try:
            with open(os.path.join(RESULTS_FOLDER, 'history.json'), 'w') as f:
                json.dump(list(results_history.values()), f, indent=2)
        except Exception as e:
            print(f"Error saving history: {e}")
        
        print(f"Processing completed for job {job_id}")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        active_jobs[job_id]['status'] = 'error'
        active_jobs[job_id]['message'] = f'Error: {str(e)}'
        print(f"Error processing video {video_path}: {e}")

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
async def get_result_file(filename: str):
    """Serve files from the results folder"""
    # Security check: prevent directory traversal
    if '..' in filename or '/' in filename or '\\' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    file_path = os.path.join(RESULTS_FOLDER, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path)

@app.get("/api/results/clips/{filename}")
async def get_clip_file(filename: str):
    """Serve clip files from the clips folder"""
    # Security check
    if '..' in filename or '/' in filename or '\\' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    clips_folder = os.path.join(RESULTS_FOLDER, "clips")
    file_path = os.path.join(clips_folder, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Clip not found")
    
    return FileResponse(file_path)

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)