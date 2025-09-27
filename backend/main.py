from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, List
import numpy as np
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
from contextlib import contextmanager
import threading
from dotenv import load_dotenv
import torch

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Add these imports at the top of main.py after existing imports
import cv2
import base64
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
from collections import deque
import signal
import sys
import atexit

# Add these imports at the top of main.py (after existing imports)
import requests
from datetime import datetime
import base64
from io import BytesIO

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


main_loop = None

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
    incident_status: str = "completed"  # 'active', 'finalizing', 'completed'
    incident_id: str = ""  # Groups related detections

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

@dataclass
class ActiveIncident:
    """Tracks active violence incidents for stitching"""
    incident_id: str
    stream_id: int
    stream_name: str
    start_time: float
    last_detection_time: float
    confidence_scores: List[float]
    detection_timestamps: List[float]  # When each detection occurred
    frame_buffer: List[np.ndarray]
    event_ids: List[int]
    
class EventStitchingManager:
    """Manages stitching of continuous violent events - FIXED: Per-stream locking"""
    
    def __init__(self, stitch_window: float = 10.0, max_incident_duration: float = 60.0):
        self.stitch_window = stitch_window
        self.max_incident_duration = max_incident_duration
        self.active_incidents: Dict[int, ActiveIncident] = {}
        # REMOVED: self._lock = threading.Lock()  # This was the global bottleneck
        self.finalization_timers: Dict[int, threading.Timer] = {}
        self.stream_locks: Dict[int, threading.Lock] = {}  # Per-stream locks instead
    
    def _get_stream_lock(self, stream_id: int) -> threading.Lock:
        """Get or create lock for specific stream"""
        if stream_id not in self.stream_locks:
            self.stream_locks[stream_id] = threading.Lock()
        return self.stream_locks[stream_id]
    
    def cleanup_stream_incidents(self, stream_id: int):
        """Cleanup and finalize any active incidents for a stream"""
        with self._get_stream_lock(stream_id):  # Only lock this stream
            if stream_id in self.active_incidents:
                print(f"Finalizing orphaned incident for stream {stream_id}")
                if stream_id in self.finalization_timers:
                    self.finalization_timers[stream_id].cancel()
                    del self.finalization_timers[stream_id]
                self.finalize_incident(stream_id)
    
    def should_stitch_to_existing(self, stream_id: int, current_time: float) -> bool:
        """Check if this detection should be stitched to an existing incident"""
        with self._get_stream_lock(stream_id):  # Only lock this stream
            if stream_id not in self.active_incidents:
                return False
                
            incident = self.active_incidents[stream_id]
            time_since_last = current_time - incident.last_detection_time
            total_duration = current_time - incident.start_time
            
            return (time_since_last <= self.stitch_window and 
                    total_duration <= self.max_incident_duration)
    
    def start_new_incident(self, stream_id: int, stream_name: str, current_time: float, 
                        confidence: float, frame_buffer: List[np.ndarray], event_id: int) -> str:
        """Start a new incident - FIXED: Only locks specific stream"""
        incident_id = f"incident_{stream_id}_{int(current_time)}"
        
        with self._get_stream_lock(stream_id):  # Only lock this stream
            # Verify the event_id exists
            try:
                conn = sqlite3.connect(event_db.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT id FROM violence_events WHERE id = ?', (event_id,))
                if not cursor.fetchone():
                    print(f"Warning: Event {event_id} not found in database")
                conn.close()
            except Exception as e:
                print(f"Error verifying event {event_id}: {e}")
            
            self.active_incidents[stream_id] = ActiveIncident(
                incident_id=incident_id,
                stream_id=stream_id,
                stream_name=stream_name,
                start_time=current_time,
                last_detection_time=current_time,
                confidence_scores=[confidence],
                detection_timestamps=[current_time],
                frame_buffer=frame_buffer.copy() if frame_buffer else [],
                event_ids=[event_id]
            )
            
            if stream_id in self.finalization_timers:
                self.finalization_timers[stream_id].cancel()
            
            self._schedule_incident_finalization(stream_id)
            
        print(f"Started new incident {incident_id} for stream {stream_id} with event {event_id}")
        return incident_id
    
    def extend_incident(self, stream_id: int, current_time: float, 
                       confidence: float, additional_frames: List[np.ndarray], event_id: int):
        """Extend existing incident - FIXED: Only locks specific stream"""
        with self._get_stream_lock(stream_id):  # Only lock this stream
            if stream_id in self.active_incidents:
                incident = self.active_incidents[stream_id]
                incident.last_detection_time = current_time
                incident.confidence_scores.append(confidence)
                incident.detection_timestamps.append(current_time)
                incident.event_ids.append(event_id)
                
                if additional_frames and len(incident.frame_buffer) < 200:
                    frames_to_add = min(len(additional_frames), 200 - len(incident.frame_buffer))
                    incident.frame_buffer.extend(additional_frames[-frames_to_add:])
                
                if stream_id in self.finalization_timers:
                    self.finalization_timers[stream_id].cancel()
                self._schedule_incident_finalization(stream_id)
                
                print(f"Extended incident {incident.incident_id}, duration: {current_time - incident.start_time:.1f}s")
    
    def _schedule_incident_finalization(self, stream_id: int):
        """Schedule incident finalization after stitch window expires"""
        def finalize():
            self.finalize_incident(stream_id)
        
        timer = threading.Timer(self.stitch_window + 2.0, finalize)  # Extra 2s buffer
        self.finalization_timers[stream_id] = timer
        timer.start()
    
    def finalize_incident(self, stream_id: int):
        """Finalize incident - FIXED: Only locks specific stream"""
        with self._get_stream_lock(stream_id):  # Only lock this stream
            if stream_id not in self.active_incidents:
                print(f"No active incident found for stream {stream_id}")
                return
                
            incident = self.active_incidents[stream_id]
            print(f"Finalizing incident {incident.incident_id} for stream {stream_id}")
            
            try:
                stitched_clip_path = self._create_stitched_clip(incident)
                print(f"Stitched clip created: {stitched_clip_path}")
                
                success = self._update_incident_events(incident, stitched_clip_path)
                
                if success:
                    self._send_incident_finalized_notification(incident, stitched_clip_path)
                    print(f"Successfully finalized incident {incident.incident_id} with {len(incident.event_ids)} events")
                else:
                    print(f"Database update failed for incident {incident.incident_id}")
                
            except Exception as e:
                print(f"Error finalizing incident {incident.incident_id}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                del self.active_incidents[stream_id]
                if stream_id in self.finalization_timers:
                    del self.finalization_timers[stream_id]
    
    def _create_stitched_clip(self, incident: ActiveIncident) -> str:
        """Create stitched video clip from frame buffer - REAL-TIME SPEED"""
        if not incident.frame_buffer:
            return ""
            
        try:
            clip_filename = f"stitched_{incident.incident_id}.mp4"
            clip_dir = os.path.join(RESULTS_FOLDER, "stream_clips")
            os.makedirs(clip_dir, exist_ok=True)
            clip_path = os.path.join(clip_dir, clip_filename)
            
            print(f"Creating real-time stitched clip at: {clip_path}")
            
            if len(incident.frame_buffer) > 0:
                height, width = incident.frame_buffer[0].shape[:2]
                
                # Use H.264 codec for better web compatibility
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                
                # FIXED: Calculate FPS to match real incident duration
                total_duration = incident.last_detection_time - incident.start_time
                frame_count = len(incident.frame_buffer)
                
                # Target: make clip duration approximately match real incident duration
                target_fps = frame_count / total_duration
                
                # Clamp FPS to reasonable video range (higher minimum for smooth playback)
                output_fps = min(max(target_fps, 8.0), 30.0)  # 8-30 fps range
                
                # Calculate actual output duration
                output_duration = frame_count / output_fps
                
                print(f"Incident: {total_duration:.1f}s real-time, {frame_count} frames")
                print(f"Output: {output_fps:.1f}fps → {output_duration:.1f}s clip (speed ratio: {output_duration/total_duration:.2f}x)")
                
                out = cv2.VideoWriter(clip_path, fourcc, output_fps, (width, height), isColor=True)
                
                if out.isOpened():
                    for frame in incident.frame_buffer:
                        out.write(frame)
                    out.release()
                    
                    # Verify clip was created successfully
                    if os.path.exists(clip_path) and os.path.getsize(clip_path) > 5000:
                        file_size = os.path.getsize(clip_path)
                        print(f"Real-time stitched clip created: {clip_path} ({file_size} bytes)")
                        print(f"Playback duration: {output_duration:.1f}s (vs {total_duration:.1f}s real incident)")
                        
                        return f"/api/results/stream_clips/{clip_filename}"
                    else:
                        print(f"Stitched clip too small or failed: {clip_path}")
                        if os.path.exists(clip_path):
                            os.remove(clip_path)
                else:
                    print(f"Could not open video writer for stitched clip")
            
            return ""
            
        except Exception as e:
            print(f"Error creating stitched clip: {e}")
            import traceback
            traceback.print_exc()
            return ""

    
    def _update_incident_events(self, incident, stitched_clip_path):
        """Update individual events and create stitched incident record - IMPROVED VERSION"""
        if not incident.event_ids:
            print(f"Warning: No event_ids for incident {incident.incident_id}")
            return
            
        print(f"Processing incident {incident.incident_id} with {len(incident.event_ids)} events")
        
        # Use event_db connection to ensure consistency
        conn = sqlite3.connect(event_db.db_path)
        cursor = conn.cursor()
        
        try:
            # Start explicit transaction
            cursor.execute('BEGIN TRANSACTION')
            
            # Calculate final incident data
            total_duration = incident.last_detection_time - incident.start_time
            avg_confidence = sum(incident.confidence_scores) / len(incident.confidence_scores)
            max_confidence = max(incident.confidence_scores)
            
            print(f"Incident stats: {total_duration:.1f}s, {avg_confidence:.3f} avg conf, {len(incident.event_ids)} events")
            
            # Create timeline segments from detection timestamps
            timeline_segments = []
            for i, (timestamp, confidence) in enumerate(zip(incident.detection_timestamps, incident.confidence_scores)):
                relative_time = timestamp - incident.start_time
                timeline_segments.append({
                    'start': relative_time,
                    'end': relative_time + 3.0,  # detection_interval
                    'confidence': confidence,
                    'detection_number': i + 1,
                    'absolute_timestamp': timestamp
                })
            
            # STEP 1: Update individual events to 'completed' status
            updated_count = 0
            for event_id in incident.event_ids:
                cursor.execute('''
                    UPDATE violence_events 
                    SET incident_status = 'completed',
                        incident_id = ?
                    WHERE id = ?
                ''', (incident.incident_id, event_id))
                
                if cursor.rowcount == 0:
                    print(f"Warning: Event {event_id} not found or not updated")
                else:
                    updated_count += 1
            
            print(f"Updated {updated_count}/{len(incident.event_ids)} events to completed status")
            
            if updated_count == 0:
                raise Exception(f"Failed to update any events for incident {incident.incident_id}")
            
            # STEP 2: Create stitched incident record using the SAME connection
            start_timestamp = datetime.fromtimestamp(incident.start_time).strftime('%Y-%m-%d %H:%M:%S')
            end_timestamp = datetime.fromtimestamp(incident.last_detection_time).strftime('%Y-%m-%d %H:%M:%S')
            event_ids_json = json.dumps(incident.event_ids)
            timeline_json = json.dumps(timeline_segments)
            
            cursor.execute('''
                INSERT INTO stitched_incidents 
                (incident_id, stream_id, stream_name, start_timestamp, end_timestamp, 
                total_duration, detection_count, avg_confidence, max_confidence, 
                stitched_clip_path, timeline_data, event_ids)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                incident.incident_id, incident.stream_id, incident.stream_name, 
                start_timestamp, end_timestamp, total_duration, len(incident.confidence_scores),
                avg_confidence, max_confidence, stitched_clip_path, timeline_json, event_ids_json
            ))
            
            stitched_id = cursor.lastrowid
            print(f"Created stitched incident record with ID {stitched_id}")
            
            # STEP 3: Commit transaction
            cursor.execute('COMMIT')
            print(f"Successfully processed incident {incident.incident_id}: {updated_count} events updated, stitched record created")
            
            return True
            
        except Exception as e:
            # STEP 4: Rollback on any failure
            print(f"Error updating incident events for {incident.incident_id}: {e}")
            cursor.execute('ROLLBACK')
            import traceback
            traceback.print_exc()
            return False
        finally:
            conn.close()
    
    def _send_incident_finalized_notification(self, incident: ActiveIncident, stitched_clip_path: str):
        """Send notification that incident has been finalized"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            total_duration = incident.last_detection_time - incident.start_time
            avg_confidence = sum(incident.confidence_scores) / len(incident.confidence_scores)
            
            # Send WebSocket notification
            loop.run_until_complete(manager.send_job_update(f"incident_finalized_{incident.incident_id}", {
                'type': 'incident_finalized',
                'incident_id': incident.incident_id,
                'stream_id': incident.stream_id,
                'stream_name': incident.stream_name,
                'total_duration': total_duration,
                'avg_confidence': avg_confidence,
                'detection_count': len(incident.confidence_scores),
                'stitched_clip': stitched_clip_path,
                'event_ids': incident.event_ids,
                'individual_events_preserved': True  # NEW: Indicate events are preserved
            }))
            
            # Send Discord incident summary (existing)
            if discord_notifier and discord_notifier.enabled:
                full_clip_url = stitched_clip_path
                if stitched_clip_path and not stitched_clip_path.startswith('http'):
                    full_clip_url = f"http://localhost:8000{stitched_clip_path}"
                
                discord_notifier.send_incident_summary(
                    incident_id=incident.incident_id,
                    stream_name=incident.stream_name,
                    duration=total_duration,
                    detection_count=len(incident.confidence_scores),
                    avg_confidence=avg_confidence,
                    clip_url=full_clip_url
                )
            
            loop.close()
            
        except Exception as e:
            print(f"Error sending incident finalized notification: {e}")

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

class DatabaseConnectionPool:
    """Simple connection pool for SQLite to reduce contention"""
    def __init__(self, db_path: str, pool_size: int = 3):
        self.db_path = db_path
        self.pool_size = pool_size
        self.connections = queue.Queue(maxsize=pool_size)
        self.lock = threading.Lock()
        
        # Pre-create connections
        for _ in range(pool_size):
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")  # Enable WAL mode for better concurrency
            self.connections.put(conn)
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool"""
        try:
            conn = self.connections.get(timeout=5.0)
            yield conn
        except queue.Empty:
            # Fallback: create temporary connection
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
            conn.close()
            return
        finally:
            self.connections.put(conn)

class EventDatabase:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.connection_pool = DatabaseConnectionPool(db_path)  # Add this line
        self.init_database()

    def init_database(self):
        """Initialize the events database"""
        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

        # Create events table (existing)
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
                incident_status TEXT DEFAULT 'completed',
                incident_id TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # NEW: Create stitched incidents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stitched_incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_id TEXT UNIQUE NOT NULL,
                stream_id INTEGER NOT NULL,
                stream_name TEXT NOT NULL,
                start_timestamp TEXT NOT NULL,
                end_timestamp TEXT NOT NULL,
                total_duration REAL NOT NULL,
                detection_count INTEGER NOT NULL,
                avg_confidence REAL NOT NULL,
                max_confidence REAL NOT NULL,
                stitched_clip_path TEXT,
                timeline_data TEXT,
                event_ids TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create daily stats table (existing)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_stats (
                date TEXT PRIMARY KEY,
                total_events INTEGER DEFAULT 0,
                total_processed INTEGER DEFAULT 0,
                violence_duration REAL DEFAULT 0.0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Add new columns if they don't exist (existing)
        try:
            cursor.execute('ALTER TABLE violence_events ADD COLUMN incident_status TEXT DEFAULT "completed"')
            cursor.execute('ALTER TABLE violence_events ADD COLUMN incident_id TEXT DEFAULT ""')
            print("Added incident tracking columns to existing database")
        except sqlite3.OperationalError:
            pass

        # Create indices for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON violence_events(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON violence_events(source_type, source_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_incident ON violence_events(incident_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON daily_stats(date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_stitched_incident ON stitched_incidents(incident_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_stitched_stream ON stitched_incidents(stream_id)')

        conn.commit()
        conn.close()

    def save_event(self, event: ViolenceEvent) -> int:
        """Save a violence event to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO violence_events 
            (timestamp, source_type, source_id, filename, start_time, end_time, 
             duration, confidence, thumbnail_path, clip_path, metadata, incident_status, incident_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event.timestamp, event.source_type, event.source_id, event.filename,
            event.start_time, event.end_time, event.duration, event.confidence,
            event.thumbnail_path, event.clip_path, event.metadata, 
            event.incident_status, event.incident_id
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

    def save_stitched_incident(self, incident_id: str, stream_id: int, stream_name: str, 
                            start_time: float, end_time: float, detection_count: int,
                            avg_confidence: float, max_confidence: float, stitched_clip_path: str,
                            timeline_data: str, event_ids: List[int]) -> int:
        """Save a stitched incident to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        start_timestamp = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
        end_timestamp = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
        total_duration = end_time - start_time
        event_ids_json = json.dumps(event_ids)

        cursor.execute('''
            INSERT INTO stitched_incidents 
            (incident_id, stream_id, stream_name, start_timestamp, end_timestamp, 
            total_duration, detection_count, avg_confidence, max_confidence, 
            stitched_clip_path, timeline_data, event_ids)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            incident_id, stream_id, stream_name, start_timestamp, end_timestamp,
            total_duration, detection_count, avg_confidence, max_confidence,
            stitched_clip_path, timeline_data, event_ids_json
        ))

        incident_record_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return incident_record_id

    def get_stitched_incidents(self, start_date: str = None, end_date: str = None, 
                            stream_id: int = None, limit: int = 50) -> List[Dict]:
        """Get stitched incidents with optional filtering"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = 'SELECT * FROM stitched_incidents WHERE 1=1'
        params = []
        
        if start_date:
            query += ' AND start_timestamp >= ?'
            params.append(start_date)
        
        if end_date:
            query += ' AND end_timestamp <= ?'
            params.append(end_date)
        
        if stream_id:
            query += ' AND stream_id = ?'
            params.append(stream_id)
        
        query += ' ORDER BY start_timestamp DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        incidents = cursor.fetchall()
        conn.close()
        
        return [
            {
                'id': row[0],
                'incident_id': row[1],
                'stream_id': row[2],
                'stream_name': row[3],
                'start_timestamp': row[4],
                'end_timestamp': row[5],
                'total_duration': row[6],
                'detection_count': row[7],
                'avg_confidence': row[8],
                'max_confidence': row[9],
                'stitched_clip_path': row[10],
                'timeline_data': json.loads(row[11]) if row[11] else [],
                'event_ids': json.loads(row[12]) if row[12] else [],
                'created_at': row[13]
            }
            for row in incidents
        ]

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
MAX_CONTENT_LENGTH = 4* 500 * 1024 * 1024  # 500MB changed to 2 GB
MODEL_PATH = r'models/rwf9425.pth'
DETECTION_THRESHOLD = 0.6

# Cleanup and resource management configuration
MAX_ACTIVE_JOBS = 100
MAX_HISTORY_ITEMS = 500
JOB_CLEANUP_AGE_HOURS = 24
FILE_CLEANUP_DELAY_HOURS = 24

load_dotenv()

# Discord Configuration
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL', '')
DISCORD_NOTIFICATIONS_ENABLED = os.getenv('DISCORD_NOTIFICATIONS_ENABLED', 'True').lower() == 'true'
DISCORD_MENTION_EVERYONE = os.getenv('DISCORD_MENTION_EVERYONE', 'False').lower() == 'true'

class DiscordNotifier:
    """Discord notification service for violence detection alerts"""

    def __init__(self, webhook_url: str = None, enabled: bool = True):
        self.webhook_url = webhook_url or DISCORD_WEBHOOK_URL
        self.enabled = enabled and bool(self.webhook_url)

        # NEW: Track thumbnails sent per stream to avoid spam
        self.thumbnails_sent = {}  # stream_id -> set of incident_ids
        self.incident_clips_sent = set()  # Track which incident clips we've sent

        if self.enabled:
            print(f"Discord notifications enabled for webhook: {self.webhook_url[:50]}...")
        else:
            print("Discord notifications disabled")

    def send_violence_alert(self,
                            stream_id: int,
                            stream_name: str,
                            confidence: float,
                            timestamp: str,
                            thumbnail_path: str = None,
                            clip_path: str = None,
                            incident_id: str = None,
                            is_ongoing: bool = False):
        """Send violence detection alert to Discord - WITH THUMBNAIL CONTROL"""

        if not self.enabled:
            return False

        try:
            # Determine alert type and color
            if is_ongoing:
                alert_type = "ONGOING INCIDENT"
                color = 0xFF6B6B  # Red
                description = f"Violence continues to be detected in {stream_name}"
            else:
                alert_type = "VIOLENCE DETECTED"
                color = 0xFF9500  # Orange
                description = f"New violence detected in {stream_name}"

            # Build embed
            embed = {
                "title": f"🚨 {alert_type}",
                "description": description,
                "color": color,
                "timestamp": datetime.now().isoformat(),
                "fields": [
                    {
                        "name": "Stream",
                        "value": f"**{stream_name}** (ID: {stream_id})",
                        "inline": True
                    },
                    {
                        "name": "Confidence",
                        "value": f"**{confidence:.1%}**",
                        "inline": True
                    },
                    {
                        "name": "Time",
                        "value": f"`{timestamp}`",
                        "inline": True
                    }
                ],
                "footer": {
                    "text": "Violence Detection System"
                }
            }

            # Add incident info if available
            if incident_id:
                embed["fields"].append({
                    "name": "Incident ID",
                    "value": f"`{incident_id}`",
                    "inline": False
                })

            # Build message content
            content = ""
            if DISCORD_MENTION_EVERYONE:
                content = "@everyone "

            content += f"Security Alert: Violence detected in **{stream_name}**"

            # Prepare payload
            payload = {
                "content": content,
                "embeds": [embed]
            }

            # Send main notification
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )

            if response.status_code in [200, 204]:
                print(f"Discord alert sent successfully for stream {stream_id}")

                # NEW: Send thumbnail ONLY for first detection in an incident
                should_send_thumbnail = self._should_send_thumbnail(stream_id, incident_id, is_ongoing)

                if should_send_thumbnail and thumbnail_path:
                    if self._send_thumbnail(stream_id, stream_name, thumbnail_path):
                        print(f"Discord thumbnail sent for stream {stream_id} (first detection)")
                        # Mark thumbnail as sent for this incident
                        self._mark_thumbnail_sent(stream_id, incident_id)

                return True
            else:
                print(f"Discord webhook failed: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            print(f"Error sending Discord notification: {e}")
            return False

    def _should_send_thumbnail(self, stream_id: int, incident_id: str, is_ongoing: bool) -> bool:
        """Determine if we should send a thumbnail for this detection"""
        if not incident_id:
            return True  # Send for non-incident detections

        # Initialize tracking for this stream if needed
        if stream_id not in self.thumbnails_sent:
            self.thumbnails_sent[stream_id] = set()

        # For new incidents, always send thumbnail
        if incident_id not in self.thumbnails_sent[stream_id]:
            return True

        # For ongoing incidents, don't send more thumbnails
        return False

    def _mark_thumbnail_sent(self, stream_id: int, incident_id: str):
        """Mark that we've sent a thumbnail for this incident"""
        if incident_id and stream_id in self.thumbnails_sent:
            self.thumbnails_sent[stream_id].add(incident_id)

    def _send_thumbnail(self, stream_id: int, stream_name: str, thumbnail_path: str) -> bool:
        """Send thumbnail image as a follow-up message"""
        try:
            # Convert URL path to file path
            if thumbnail_path.startswith('/api/results/'):
                file_path = os.path.join(RESULTS_FOLDER, thumbnail_path.replace('/api/results/', ''))
            else:
                file_path = thumbnail_path

            if not os.path.exists(file_path):
                print(f"Thumbnail file not found: {file_path}")
                return False

            # Read and send image
            with open(file_path, 'rb') as f:
                files = {
                    'file': (f'detection_{stream_id}.jpg', f, 'image/jpeg')
                }

                payload = {
                    'content': f'📸 Detection snapshot from **{stream_name}**'
                }

                response = requests.post(
                    self.webhook_url,
                    data=payload,
                    files=files,
                    timeout=15
                )

                return response.status_code in [200, 204]

        except Exception as e:
            print(f"Error sending Discord thumbnail: {e}")
            return False

    def send_incident_summary(self,
                              incident_id: str,
                              stream_name: str,
                              duration: float,
                              detection_count: int,
                              avg_confidence: float,
                              clip_url: str = None):
        """Send incident summary when an incident is finalized - WITH ACTUAL CLIP"""

        if not self.enabled:
            return False

        try:
            embed = {
                "title": "📋 INCIDENT SUMMARY",
                "description": f"Violence incident concluded in **{stream_name}**",
                "color": 0x4CAF50,  # Green
                "timestamp": datetime.now().isoformat(),
                "fields": [
                    {
                        "name": "Incident ID",
                        "value": f"`{incident_id}`",
                        "inline": False
                    },
                    {
                        "name": "Duration",
                        "value": f"**{duration:.1f} seconds**",
                        "inline": True
                    },
                    {
                        "name": "Detections",
                        "value": f"**{detection_count}** alerts",
                        "inline": True
                    },
                    {
                        "name": "Avg Confidence",
                        "value": f"**{avg_confidence:.1%}**",
                        "inline": True
                    }
                ],
                "footer": {
                    "text": "Incident processed and archived"
                }
            }
            
            payload = {
                "content": f"✅ Incident **{incident_id}** has been processed and archived.",
                "embeds": [embed]
            }

            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )

            if response.status_code in [200, 204]:
                print(f"Discord incident summary sent for {incident_id}")

                # NEW: Send the actual video file if available and not already sent
                if clip_url and incident_id not in self.incident_clips_sent:
                    if self._send_incident_clip_file(incident_id, stream_name, clip_url):
                        print(f"Discord incident clip sent for {incident_id}")
                        self.incident_clips_sent.add(incident_id)

                return True
            else:
                print(f"Discord incident summary webhook failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"Error sending Discord incident summary: {e}")
            return False

    def _send_incident_clip_file(self, incident_id: str, stream_name: str, clip_url: str) -> bool:
        """Send incident video file to Discord - FIXED PATH CONVERSION"""
        try:
            # Convert URL to file path - HANDLE FULL URLs
            if clip_url.startswith('http://') or clip_url.startswith('https://'):
                # Extract the path part from full URL
                # http://localhost:8000/api/results/stream_clips/file.mp4 -> /api/results/stream_clips/file.mp4
                from urllib.parse import urlparse
                parsed_url = urlparse(clip_url)
                url_path = parsed_url.path  # This gives us "/api/results/stream_clips/file.mp4"
                
                if url_path.startswith('/api/results/'):
                    # Remove /api/results/ prefix and join with RESULTS_FOLDER
                    relative_path = url_path.replace('/api/results/', '')
                    file_path = os.path.join(RESULTS_FOLDER, relative_path)
                else:
                    print(f"Unexpected URL path format: {url_path}")
                    return False
                    
            elif clip_url.startswith('/api/results/'):
                # Handle relative URL paths
                file_path = os.path.join(RESULTS_FOLDER, clip_url.replace('/api/results/', ''))
            else:
                # Assume it's already a file path
                file_path = clip_url
            
            print(f"Discord clip lookup: {clip_url} -> {file_path}")
            
            if not os.path.exists(file_path):
                print(f"Incident clip file not found: {file_path}")
                
                # Try to find the file by pattern matching (fallback)
                clips_dir = os.path.join(RESULTS_FOLDER, "stream_clips")
                if os.path.exists(clips_dir):
                    # Look for files matching the incident ID pattern
                    import glob
                    pattern = os.path.join(clips_dir, f"*{incident_id}*.mp4")
                    matching_files = glob.glob(pattern)
                    
                    if matching_files:
                        file_path = matching_files[0]  # Use first match
                        print(f"Found clip via pattern matching: {file_path}")
                    else:
                        print(f"No matching clip files found for incident {incident_id}")
                        
                        # Send a message explaining no clip is available
                        payload = {
                            'content': f'🎥 Incident clip for **{stream_name}** (ID: `{incident_id}`) was not found on disk.\n' +
                                    f'The incident summary has been recorded, but video clip is missing.'
                        }
                        
                        requests.post(self.webhook_url, json=payload, timeout=10)
                        return False
                else:
                    print(f"Stream clips directory not found: {clips_dir}")
                    return False
            
            # Check original file size
            original_size = os.path.getsize(file_path)
            max_size = 25 * 1024 * 1024  # 25MB limit
            
            final_file_path = file_path
            
            # If file is too large, compress it
            if original_size > max_size:
                print(f"Incident clip too large ({original_size / (1024*1024):.1f}MB), compressing...")
                
                compressed_path = self._compress_video_for_discord(file_path, max_size)
                
                if compressed_path and os.path.exists(compressed_path):
                    final_file_path = compressed_path
                    compressed_size = os.path.getsize(compressed_path)
                    print(f"Compressed from {original_size / (1024*1024):.1f}MB to {compressed_size / (1024*1024):.1f}MB")
                else:
                    # Compression failed, send explanation message
                    payload = {
                        'content': f'🎥 Incident clip for **{stream_name}** is too large for Discord ({original_size / (1024*1024):.1f}MB > 25MB)\n' +
                                f'Compression failed. Clip saved locally: `{os.path.basename(file_path)}`'
                    }
                    
                    requests.post(self.webhook_url, json=payload, timeout=10)
                    return False
            
            # Send the file (original or compressed)
            try:
                with open(final_file_path, 'rb') as f:
                    files = {
                        'file': (f'incident_{incident_id}.mp4', f, 'video/mp4')
                    }
                    
                    # Add compression info to message if file was compressed
                    content = f'🎥 Full incident recording from **{stream_name}** (ID: `{incident_id}`)'
                    if final_file_path != file_path:
                        content += f'\n📉 Compressed from {original_size / (1024*1024):.1f}MB to fit Discord limits'
                    
                    payload = {
                        'content': content
                    }
                    
                    print(f"Uploading incident clip to Discord: {os.path.basename(final_file_path)} ({os.path.getsize(final_file_path) / (1024*1024):.1f}MB)")
                    
                    response = requests.post(
                        self.webhook_url,
                        data=payload,
                        files=files,
                        timeout=60  # Longer timeout for video uploads
                    )
                    
                    # Clean up compressed file if it was created
                    if final_file_path != file_path:
                        try:
                            os.remove(final_file_path)
                            print(f"Cleaned up temporary compressed file: {final_file_path}")
                        except:
                            pass
                    
                    if response.status_code in [200, 204]:
                        print(f"Successfully sent incident clip to Discord for {incident_id}")
                        return True
                    else:
                        print(f"Discord upload failed: {response.status_code} - {response.text}")
                        return False
                    
            except Exception as e:
                print(f"Error uploading video file to Discord: {e}")
                return False
                
        except Exception as e:
            print(f"Error sending Discord incident clip: {e}")
            return False

    # Also add this helper method for better debugging
    def _debug_file_paths(self, clip_url: str, stream_clips_dir: str = None):
        """Debug helper to show file path resolution"""
        if not stream_clips_dir:
            stream_clips_dir = os.path.join(RESULTS_FOLDER, "stream_clips")
        
        print(f"=== DEBUG: File Path Resolution ===")
        print(f"Input clip_url: {clip_url}")
        print(f"Stream clips directory: {stream_clips_dir}")
        print(f"Directory exists: {os.path.exists(stream_clips_dir)}")
        
        if os.path.exists(stream_clips_dir):
            print(f"Files in directory:")
            try:
                for file in os.listdir(stream_clips_dir):
                    file_path = os.path.join(stream_clips_dir, file)
                    file_size = os.path.getsize(file_path) / (1024*1024)
                    print(f"  - {file} ({file_size:.1f}MB)")
            except Exception as e:
                print(f"  Error listing files: {e}")
        
        print("=== END DEBUG ===")

    def _compress_video_for_discord(self, input_path: str, max_size_bytes: int) -> Optional[str]:
        """Compress video to fit under Discord's size limit"""
        try:
            # Create temporary output file
            temp_dir = tempfile.gettempdir()
            temp_filename = f"discord_compressed_{int(time.time())}.mp4"
            output_path = os.path.join(temp_dir, temp_filename)
            
            # Use FFmpeg command for more reliable compression
            # This is generally more effective and robust than manipulating frames with OpenCV
            # It requires FFmpeg to be installed and in the system's PATH.
            # We use a two-pass encoding to better target the final file size.
            
            # Get video duration using OpenCV
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                print(f"Cannot open video for compression: {input_path}")
                return None
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = frame_count / fps if fps > 0 else 0
            cap.release()
            
            if duration_sec <= 0:
                print("Could not determine video duration. Cannot compress.")
                return None
            
            # Target bitrate in bits per second (leaving a 5% buffer)
            target_bitrate_bps = int((max_size_bytes * 8 / duration_sec) * 0.95)
            
            # FFmpeg command
            # -y: overwrite output file
            # -i: input file
            # -c:v libx264: use H.264 codec
            # -b:v: target video bitrate
            # -pass 1/2: two-pass encoding
            # -an: no audio
            # -f mp4: output format
            command_pass1 = f'ffmpeg -y -i "{input_path}" -c:v libx264 -b:v {target_bitrate_bps} -pass 1 -an -f mp4 /dev/null'
            command_pass2 = f'ffmpeg -y -i "{input_path}" -c:v libx264 -b:v {target_bitrate_bps} -pass 2 -c:a aac -b:a 128k "{output_path}"'
            
            # On Windows, the null output is 'NUL'
            if os.name == 'nt':
                command_pass1 = f'ffmpeg -y -i "{input_path}" -c:v libx264 -b:v {target_bitrate_bps} -pass 1 -an -f mp4 NUL'

            print(f"Running FFmpeg pass 1...")
            os.system(command_pass1)
            
            print(f"Running FFmpeg pass 2...")
            os.system(command_pass2)

            # Check if compression was successful and within limits
            if os.path.exists(output_path) and os.path.getsize(output_path) <= max_size_bytes:
                 print(f"FFmpeg compression successful.")
                 return output_path
            else:
                 if os.path.exists(output_path):
                    os.remove(output_path)
                 print("FFmpeg compression failed or output file is still too large.")
                 return None

        except Exception as e:
            print(f"Error during video compression with FFmpeg: {e}")
            return None


    def send_batch_upload_start(self, batch_id: str, video_count: int, video_names: List[str]):
        """Send notification when multiple videos are uploaded for batch processing"""
        
        if not self.enabled:
            return False
            
        try:
            # Truncate video list if too long
            display_videos = video_names[:5]
            more_count = len(video_names) - 5
            
            video_list = "• " + "\n• ".join(f"`{name}`" for name in display_videos)
            if more_count > 0:
                video_list += f"\n• ... and {more_count} more videos"
            
            embed = {
                "title": "📁 BATCH UPLOAD STARTED",
                "description": f"Processing **{video_count} videos** for violence detection",
                "color": 0x2196F3,  # Blue
                "timestamp": datetime.now().isoformat(),
                "fields": [
                    {
                        "name": "Batch ID",
                        "value": f"`{batch_id}`",
                        "inline": True
                    },
                    {
                        "name": "Video Count",
                        "value": f"**{video_count}** files",
                        "inline": True
                    },
                    {
                        "name": "Status",
                        "value": "🟡 **Processing**",
                        "inline": True
                    },
                    {
                        "name": "Videos",
                        "value": video_list,
                        "inline": False
                    }
                ],
                "footer": {
                    "text": "Multi-Video Analysis Pipeline"
                }
            }
            
            content = f"📋 **Batch Upload**: {video_count} videos queued for analysis"
            
            payload = {
                "content": content,
                "embeds": [embed]
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code in [200, 204]:
                print(f"Discord batch start notification sent for {video_count} videos")
                return True
            else:
                print(f"Discord batch webhook failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Error sending Discord batch start notification: {e}")
            return False

    def send_batch_video_complete(self, batch_id: str, video_name: str, completed_count: int, 
                                  total_count: int, has_violence: bool, confidence: float = None,
                                  job_id: str = None, thumbnail_path: str = None, 
                                  clip_paths: List[str] = None):
        """Send notification when an individual video in a batch completes - REDUCED THUMBNAILS"""
        
        if not self.enabled:
            return False
            
        try:
            # Progress calculation
            progress_percent = (completed_count / total_count) * 100
            
            # Status based on violence detection
            if has_violence:
                status_emoji = "🚨"
                status_text = f"**VIOLENCE DETECTED** ({confidence:.1%})"
                color = 0xFF6B6B  # Red
            else:
                status_emoji = "✅"
                status_text = "**No Violence**"
                color = 0x4CAF50  # Green
            
            embed = {
                "title": f"{status_emoji} VIDEO ANALYSIS COMPLETE",
                "description": f"Video **{completed_count}** of **{total_count}** processed",
                "color": color,
                "timestamp": datetime.now().isoformat(),
                "fields": [
                    {
                        "name": "Video File",
                        "value": f"`{video_name}`",
                        "inline": False
                    },
                    {
                        "name": "Result",
                        "value": status_text,
                        "inline": True
                    },
                    {
                        "name": "Progress",
                        "value": f"**{completed_count}/{total_count}** ({progress_percent:.0f}%)",
                        "inline": True
                    },
                    {
                        "name": "Batch ID",
                        "value": f"`{batch_id}`",
                        "inline": True
                    }
                ],
                "footer": {
                    "text": f"Job ID: {job_id}" if job_id else "Multi-Video Analysis"
                }
            }
            
            # Add view link if job_id provided
            if job_id:
                # Assuming a local dashboard URL structure
                embed["fields"].append({
                    "name": "View Results",
                    "value": f"[Open Dashboard](http://localhost:8000/dashboard?job={job_id})",
                    "inline": False
                })
            
            payload = {"embeds": [embed]}
            
            # Send main notification
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code in [200, 204]:
                print(f"Discord batch video complete notification sent for {video_name}")
                
                # NEW: Only send thumbnail for HIGH confidence violent videos (>90%)
                if has_violence and thumbnail_path and confidence and confidence > 0.90:
                    if self._send_batch_thumbnail(job_id, video_name, thumbnail_path):
                        print(f"Discord batch thumbnail sent for high-confidence detection: {video_name}")
                
                # NEW: Only send clips for VERY HIGH confidence (>95%) and limit to 1
                if has_violence and clip_paths and confidence and confidence > 0.95:
                    # Send only the first clip
                    first_clip = clip_paths[0] if clip_paths else None
                    if first_clip and self._send_batch_clip(job_id, video_name, first_clip):
                        print(f"Discord batch clip sent for very high confidence: {video_name}")
                
                return True
            else:
                print(f"Discord batch video webhook failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Error sending Discord batch video complete notification: {e}")
            return False

    def send_batch_complete_summary(self, batch_id: str, total_videos: int, violence_count: int, 
                                    processing_time: float, video_results: List[Dict]):
        """Send final summary when an entire batch is complete - SMART THUMBNAIL SELECTION"""
        
        if not self.enabled:
            return False
            
        try:
            # Calculate statistics
            violence_rate = (violence_count / total_videos) * 100 if total_videos > 0 else 0
            safe_count = total_videos - violence_count
            
            # Determine overall alert level
            if violence_count == 0:
                alert_emoji = "✅"
                alert_title = "BATCH COMPLETE - ALL SAFE"
                color = 0x4CAF50  # Green
            elif violence_count == total_videos:
                alert_emoji = "🚨"
                alert_title = "BATCH COMPLETE - ALL VIOLENT"
                color = 0xFF6B6B  # Red
            else:
                alert_emoji = "⚠️"
                alert_title = "BATCH COMPLETE - MIXED RESULTS"
                color = 0xFF9500  # Orange
            
            embed = {
                "title": f"{alert_emoji} {alert_title}",
                "description": f"Analysis complete for **{total_videos} videos**",
                "color": color,
                "timestamp": datetime.now().isoformat(),
                "fields": [
                    {
                        "name": "Batch ID",
                        "value": f"`{batch_id}`",
                        "inline": False
                    },
                    {
                        "name": "🚨 Violence Detected",
                        "value": f"**{violence_count}** videos ({violence_rate:.0f}%)",
                        "inline": True
                    },
                    {
                        "name": "✅ Safe Videos",
                        "value": f"**{safe_count}** videos ({100-violence_rate:.0f}%)",
                        "inline": True
                    },
                    {
                        "name": "⏱️ Processing Time",
                        "value": f"**{processing_time:.1f}** seconds",
                        "inline": True
                    }
                ],
                "footer": {
                    "text": "Multi-Video Analysis Complete"
                }
            }
            
            # Add detailed results if not too many
            if len(video_results) <= 8:
                results_text = ""
                for result in video_results:
                    status_icon = "🚨" if result.get('has_violence') else "✅"
                    conf_text = f" ({result.get('confidence', 0):.1%})" if result.get('has_violence') else ""
                    results_text += f"{status_icon} `{result.get('filename', 'N/A')}`{conf_text}\n"
                
                embed["fields"].append({
                    "name": "📋 Detailed Results",
                    "value": results_text,
                    "inline": False
                })
            
            # Add multi-analysis dashboard link
            job_ids = [r.get('job_id') for r in video_results if r.get('job_id')]
            if job_ids:
                jobs_param = ','.join(job_ids)
                # Assuming local dashboard URL
                embed["fields"].append({
                    "name": "📊 View All Results",
                    "value": f"[Open Multi-Analysis Dashboard](http://localhost:8000/multi-analysis?jobs={jobs_param})",
                    "inline": False
                })
            
            content = ""
            if DISCORD_MENTION_EVERYONE and violence_count > 0:
                content = "@everyone "
            
            if violence_count > 0:
                content += f"🚨 **ALERT**: {violence_count}/{total_videos} videos contain violence!"
            else:
                content += f"✅ **ALL CLEAR**: All {total_videos} videos are safe."
            
            payload = {
                "content": content,
                "embeds": [embed]
            }
            
            # Send main summary
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code in [200, 204]:
                print(f"Discord batch summary sent for {total_videos} videos")
                
                # NEW: SMART thumbnail selection - only send 1 thumbnail from the HIGHEST confidence detection
                violent_results = [r for r in video_results if r.get('has_violence', False)]
                
                if violent_results:
                    # Sort by confidence and get the highest one
                    violent_results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                    best_result = violent_results[0]
                    
                    job_id = best_result.get('job_id')
                    filename = best_result.get('filename', 'unknown')
                    confidence = best_result.get('confidence', 0)
                    
                    if job_id and confidence > 0.85:  # Only for high confidence
                        # Try to send thumbnail from the best detection
                        thumbnail_path = os.path.join(RESULTS_FOLDER, f"{job_id}_thumbnail.jpg")
                        if os.path.exists(thumbnail_path):
                            if self._send_batch_thumbnail(job_id, f"Best Detection ({confidence:.1%}): {filename}", f"/api/results/{job_id}_thumbnail.jpg"):
                                print(f"Sent single best thumbnail for batch summary: {filename}")
                
                return True
            else:
                print(f"Discord batch summary webhook failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Error sending Discord batch summary: {e}")
            return False

    def _send_batch_thumbnail(self, job_id: str, filename: str, thumbnail_path: str) -> bool:
        """Send thumbnail image for a batch upload video"""
        try:
            if thumbnail_path.startswith('/api/results/'):
                file_path = os.path.join(RESULTS_FOLDER, thumbnail_path.replace('/api/results/', ''))
            else:
                file_path = thumbnail_path
            
            if not os.path.exists(file_path):
                print(f"Batch thumbnail file not found: {file_path}")
                return False
            
            with open(file_path, 'rb') as f:
                files = {
                    'file': (f'batch_{job_id}.jpg', f, 'image/jpeg')
                }
                
                payload = {
                    'content': f'📸 {filename}'
                }
                
                response = requests.post(
                    self.webhook_url,
                    data=payload,
                    files=files,
                    timeout=15
                )
                
                return response.status_code in [200, 204]
                
        except Exception as e:
            print(f"Error sending Discord batch thumbnail: {e}")
            return False

    def _send_batch_clip(self, job_id: str, filename: str, clip_path: str) -> bool:
        """Send video clip for a batch upload video - WITH COMPRESSION"""
        try:
            if clip_path.startswith('/api/results/'):
                file_path = os.path.join(RESULTS_FOLDER, clip_path.replace('/api/results/', ''))
            else:
                file_path = clip_path
            
            if not os.path.exists(file_path):
                print(f"Batch clip file not found: {file_path}")
                return False
            
            # Check file size and compress if needed
            original_size = os.path.getsize(file_path)
            max_size = 25 * 1024 * 1024  # 25MB
            
            final_file_path = file_path
            
            if original_size > max_size:
                print(f"Batch clip too large ({original_size / (1024*1024):.1f}MB), compressing...")
                
                compressed_path = self._compress_video_for_discord(file_path, max_size)
                
                if compressed_path and os.path.exists(compressed_path):
                    final_file_path = compressed_path
                    compressed_size = os.path.getsize(compressed_path)
                    print(f"Batch clip compressed from {original_size / (1024*1024):.1f}MB to {compressed_size / (1024*1024):.1f}MB")
                else:
                    print(f"Batch clip compression failed, skipping Discord upload")
                    return False
            
            with open(final_file_path, 'rb') as f:
                files = {
                    'file': (f'batch_{job_id}.mp4', f, 'video/mp4')
                }
                
                content = f'🎥 Violence clip from **{filename}**'
                if final_file_path != file_path:
                    content += f' (compressed to fit Discord)'
                
                payload = {'content': content}
                
                response = requests.post(
                    self.webhook_url,
                    data=payload,
                    files=files,
                    timeout=30
                )
                
                # Clean up compressed file
                if final_file_path != file_path:
                    try:
                        os.remove(final_file_path)
                    except OSError:
                        pass
                
                return response.status_code in [200, 204]
                
        except Exception as e:
            print(f"Error sending Discord batch clip: {e}")
            return False

    def send_system_status(self, message: str, status_type: str = "info"):
        """Send system status messages"""
        
        if not self.enabled:
            return False
            
        try:
            color_map = {
                "info": 0x2196F3,    # Blue
                "warning": 0xFF9800, # Orange  
                "error": 0xF44336,   # Red
                "success": 0x4CAF50  # Green
            }
            
            emoji_map = {
                "info": "ℹ️",
                "warning": "⚠️", 
                "error": "❌",
                "success": "✅"
            }
            
            embed = {
                "title": f"{emoji_map.get(status_type, 'ℹ️')} System Status",
                "description": message,
                "color": color_map.get(status_type, 0x2196F3),
                "timestamp": datetime.now().isoformat(),
                "footer": {
                    "text": "Violence Detection System"
                }
            }
            
            payload = {"embeds": [embed]}
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            
            return response.status_code in [200, 204]
            
        except Exception as e:
            print(f"Error sending Discord status message: {e}")
            return False
            
def create_batch_id() -> str:
    """Generate unique batch ID"""
    return f"batch_{int(time.time())}_{str(uuid.uuid4())[:8]}"

def track_batch_upload(batch_id: str, job_ids: List[str], filenames: List[str]):
    """Track a new batch upload"""
    with batch_lock:
        batch_uploads[batch_id] = {
            'job_ids': job_ids,
            'filenames': filenames,
            'total_count': len(job_ids),
            'completed_count': 0,
            'results': {},
            'start_time': time.time(),
            'completed_jobs': set()
        }
    print(f"Tracking batch {batch_id} with {len(job_ids)} videos")

def update_batch_completion(job_id: str, result: Dict):
    """Update batch when a job completes - WITH MEDIA DATA"""
    with batch_lock:
        # Find which batch this job belongs to
        for batch_id, batch_data in batch_uploads.items():
            if job_id in batch_data['job_ids'] and job_id not in batch_data['completed_jobs']:
                batch_data['completed_jobs'].add(job_id)
                batch_data['completed_count'] += 1
                batch_data['results'][job_id] = result
                
                # Get filename for this job
                job_index = batch_data['job_ids'].index(job_id)
                filename = batch_data['filenames'][job_index]
                
                # Prepare media paths for Discord
                thumbnail_path = result.get('thumbnail', '')
                clip_paths = []
                
                # Extract clip paths from segments if violence was detected
                if result.get('has_violence', False) and 'segments' in result:
                    for i, segment in enumerate(result['segments']):
                        # Look for clips in the results folder
                        clip_filename = f"{job_id}_clip_{i}.mp4"
                        clip_path = os.path.join(RESULTS_FOLDER, "clips", clip_filename)
                        if os.path.exists(clip_path):
                            clip_paths.append(f"/api/results/clips/{clip_filename}")
                
                # Send individual completion notification WITH MEDIA
                if discord_notifier and discord_notifier.enabled:
                    discord_notifier.send_batch_video_complete(
                        batch_id=batch_id,
                        video_name=filename,
                        completed_count=batch_data['completed_count'],
                        total_count=batch_data['total_count'],
                        has_violence=result.get('has_violence', False),
                        confidence=result.get('overall_result', {}).get('confidence', 0),
                        job_id=job_id,
                        thumbnail_path=thumbnail_path,  # NEW: Pass thumbnail
                        clip_paths=clip_paths  # NEW: Pass clip paths
                    )
                
                # Check if batch is complete
                if batch_data['completed_count'] >= batch_data['total_count']:
                    finalize_batch(batch_id)
                
                break

def finalize_batch(batch_id: str):
    """Send final batch summary and cleanup"""
    batch_data = batch_uploads.get(batch_id)
    if not batch_data:
        return
    
    # Calculate final statistics
    processing_time = time.time() - batch_data['start_time']
    violence_count = sum(1 for result in batch_data['results'].values() 
                         if result.get('has_violence', False))
    
    # Prepare results for summary
    video_results = []
    for i, job_id in enumerate(batch_data['job_ids']):
        result = batch_data['results'].get(job_id, {})
        video_results.append({
            'filename': batch_data['filenames'][i],
            'job_id': job_id,
            'has_violence': result.get('has_violence', False),
            'confidence': result.get('overall_result', {}).get('confidence', 0)
        })
    
    # Send final summary
    if discord_notifier and discord_notifier.enabled:
        discord_notifier.send_batch_complete_summary(
            batch_id=batch_id,
            total_videos=batch_data['total_count'],
            violence_count=violence_count,
            processing_time=processing_time,
            video_results=video_results
        )
    
    # Cleanup
    del batch_uploads[batch_id]
    print(f"Finalized batch {batch_id}: {violence_count}/{batch_data['total_count']} videos had violence")

# Cleanup old batches periodically
def cleanup_old_batches():
    """Remove batch tracking for very old batches (in case of missed completions)"""
    current_time = time.time()
    with batch_lock:
        to_remove = []
        for batch_id, batch_data in batch_uploads.items():
            # Remove batches older than 2 hours
            if current_time - batch_data['start_time'] > 7200:
                to_remove.append(batch_id)
        
        for batch_id in to_remove:
            del batch_uploads[batch_id]
            print(f"Cleaned up old batch: {batch_id}")

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
stitching_manager = None
discord_notifier = None
batch_uploads = {}
batch_lock = threading.Lock()

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
    
    # Finalize any orphaned incidents
    if stitching_manager:
        try:
            streams_with_incidents = list(stitching_manager.active_incidents.keys())
            for stream_id in streams_with_incidents:
                stitching_manager.cleanup_stream_incidents(stream_id)
            print(f"Finalized {len(streams_with_incidents)} orphaned incidents")
        except Exception as e:
            print(f"Error finalizing orphaned incidents: {e}")
    
    streams_to_stop = list(active_streams.keys())

    for stream_id in streams_to_stop:
        try:
            print(f"Stopping stream {stream_id}...")
            active_streams[stream_id]['processor'].stop_stream()
        except Exception as e:
            print(f"Error stopping stream {stream_id}: {e}")

    # Shutdown background executor
    try:
        background_executor.shutdown(wait=True)
        print("Background executor shutdown completed")
    except Exception as e:
        print(f"Error during background executor shutdown: {e}")

    # Shutdown main executor
    try:
        executor.shutdown(wait=True)
        print("Main executor shutdown completed")
    except Exception as e:
        print(f"Error during main executor shutdown: {e}")
    
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
    """Enhanced stream health monitoring"""
    while not global_shutdown_event.is_set():
        try:
            await asyncio.sleep(30)
            
            if global_shutdown_event.is_set():
                break
                
            streams_to_check = list(active_streams.keys())
            for stream_id in streams_to_check:
                if global_shutdown_event.is_set():
                    break
                    
                try:
                    processor = active_streams[stream_id]['processor']
                    health = processor.check_health()
                    
                    if 'error' in health:
                        print(f"Stream {stream_id} health check error: {health['error']}")
                        continue
                    
                    # Check if any critical component is down
                    if not health['capture_thread'] or not health['process_thread']:
                        print(f"Stream {stream_id} critical thread failure: {health}")
                        
                        # Try to restart the stream
                        try:
                            processor.stop_stream()
                        except:
                            pass
                        
                        if stream_db:
                            stream_db.update_stream_status(stream_id, 'error')
                    
                    # Log health status occasionally
                    elif stream_id % 2 == 0:  # Log every other stream to reduce spam
                        print(f"Stream {stream_id} health: threads OK, queue: {health['queue_size']}, buffer: {health['buffer_size']}")
                        
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

# Add this to your main.py imports at the top
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
from collections import deque

# Add this global executor for background tasks (add after other globals)
background_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="background_task")

class RTSPStreamProcessor:
    def __init__(self, stream_id: int, rtsp_url: str, stream_name: str):
        self.stream_id = stream_id
        self.rtsp_url = rtsp_url
        self.stream_name = stream_name

        # Each stream gets its own model instance
        self.model = None
        self.model_device = None

        # Capture objects and control
        self.cap = None
        self.is_running = False
        self.capture_thread = None
        self.process_thread = None

        # Frame buffers and processing
        self.raw_frame_queue = queue.Queue(maxsize=30)
        self.rgb_frame_buffer = deque(maxlen=16)
        self.display_frame_buffer = deque(maxlen=50)
        self.last_display_frame = None

        # Timing and rate control
        self.last_detection_time = 0
        self.detection_interval = 3.0
        self.target_fps = 8
        self.frame_skip_counter = 0

        # Model input requirements
        self.model_input_size = (336, 336)
        self.model_temporal_length = 16

        # Thread synchronization
        self._lock = threading.Lock()
        
        # Background task queue
        self.detection_queue = queue.Queue(maxsize=10)

    def _load_model_instance(self):
        """Load dedicated model instance for this stream"""
        try:
            print(f"Loading model instance for stream {self.stream_id}...")
            
            # Load model with same device detection logic
            device = None
            if torch.cuda.is_available():
                try:
                    test_tensor = torch.zeros(1, device='cuda')
                    test_tensor = test_tensor + 1
                    device = torch.device('cuda')
                    print(f"Stream {self.stream_id}: Using CUDA")
                except Exception as e:
                    print(f"Stream {self.stream_id}: CUDA incompatible ({e}), using CPU")
                    device = torch.device('cpu')
            else:
                device = torch.device('cpu')
                print(f"Stream {self.stream_id}: CUDA not available, using CPU")
            
            self.model_device = device
            
            # Load the model (reuse existing function but for this instance)
            model, _ = load_violence_detection_model(MODEL_PATH, device)
            
            print(f"Stream {self.stream_id}: Model loaded successfully on {device}")
            return model
            
        except Exception as e:
            print(f"Error loading model for stream {self.stream_id}: {e}")
            return None

    def _run_detection(self):
        """Run violence detection using dedicated model instance"""
        if self.model is None or len(self.rgb_frame_buffer) < self.model_temporal_length:
            return

        try:
            # Extract frames and run detection
            frames_list = list(self.rgb_frame_buffer)[-self.model_temporal_length:]
            frames_array = np.array(frames_list, dtype=np.uint8)

            print(f"Stream {self.stream_id}: Detection input shape: {frames_array.shape}")

            # Ensure model is in eval mode
            self.model.eval()

            # Check if model uses motion enhancement
            use_motion = hasattr(self.model, 'use_motion_enhancement') and self.model.use_motion_enhancement
            print(f"Stream {self.stream_id}: Using motion enhancement: {use_motion}")

            # Run preprocessing and prediction using dedicated model
            processed_data = preprocess_frames(frames_array, compute_flow=use_motion)

            # Ensure tensors are on correct device
            for key, tensor in processed_data.items():
                if tensor.device != self.model_device:
                    processed_data[key] = tensor.to(self.model_device)

            # Run prediction with dedicated model (no locking needed)
            is_violent, confidence, inference_time = predict_violence(
                self.model, processed_data, DETECTION_THRESHOLD, debug=True
            )

            print(f"Stream {self.stream_id}: Final result - Violence: {is_violent}, Confidence: {confidence:.3f}, Time: {inference_time:.3f}s")

            # Queue detection event if violence detected
            if is_violent and confidence > DETECTION_THRESHOLD:
                print(f"ALERT: Violence detected in stream {self.stream_id}: {confidence:.3f}")
                
                current_time = time.time()
                
                # Get snapshot of current frame and buffer state
                recent_frames = []
                current_frame = None
                with self._lock:
                    if self.display_frame_buffer:
                        recent_frames = list(self.display_frame_buffer)[-30:]
                    if self.last_display_frame is not None:
                        current_frame = self.last_display_frame.copy()
                
                # Submit to background queue
                try:
                    detection_data = {
                        'stream_id': self.stream_id,
                        'stream_name': self.stream_name,
                        'confidence': confidence,
                        'timestamp': current_time,
                        'current_frame': current_frame,
                        'recent_frames': recent_frames,
                        'rtsp_url': self.rtsp_url
                    }
                    
                    self.detection_queue.put_nowait(detection_data)
                    print(f"Stream {self.stream_id}: Queued detection event for background processing")
                    
                    # Send immediate lightweight WebSocket notification
                    self._send_immediate_alert(confidence, current_time)
                    
                except queue.Full:
                    print(f"WARNING: Detection queue full for stream {self.stream_id}, dropping event")
            else:
                print(f"Stream {self.stream_id}: No violence detected (confidence: {confidence:.3f})")

        except Exception as e:
            print(f"Error running detection on stream {self.stream_id}: {e}")
            import traceback
            traceback.print_exc()

    def _send_immediate_alert(self, confidence: float, timestamp: float):
        """Send immediate lightweight alert - no heavy processing"""
        try:
            if main_loop and main_loop.is_running():
                payload = {
                    'type': 'violence_detected_immediate',
                    'stream_id': self.stream_id,
                    'stream_name': self.stream_name,
                    'confidence': confidence,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'is_processing': True  # Indicate full processing is happening in background
                }
                
                asyncio.run_coroutine_threadsafe(
                    manager.send_job_update(f"violence_immediate_{self.stream_id}_{int(timestamp)}", payload),
                    main_loop
                )
                print(f"Stream {self.stream_id}: Sent immediate alert via WebSocket")
        except Exception as e:
            print(f"Error sending immediate alert for stream {self.stream_id}: {e}")

    def start_stream(self):
        """Start the RTSP stream with dedicated model"""
        if not self.validate_rtsp_url(self.rtsp_url):
            print(f"Invalid RTSP URL: {self.rtsp_url}")
            return False

        try:
            # Load model instance first
            self.model = self._load_model_instance()
            if self.model is None:
                print(f"Failed to load model for stream {self.stream_id}")
                return False

            # Initialize capture with proper settings
            self.cap = cv2.VideoCapture(self.rtsp_url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            try:
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
                self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            except:
                pass

            if not self.cap.isOpened():
                print(f"Failed to open RTSP stream: {self.rtsp_url}")
                return False

            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                print(f"Failed to read test frame from stream: {self.rtsp_url}")
                self.cap.release()
                return False

            print(f"Stream {self.stream_id} frame size: {test_frame.shape}")

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
            self.background_thread = threading.Thread(target=self._background_detection_processor, daemon=True)

            self.capture_thread.start()
            self.process_thread.start()
            self.background_thread.start()

            print(f"Successfully started stream {self.stream_id}: {self.stream_name} with dedicated model")
            return True

        except Exception as e:
            print(f"Error starting stream {self.stream_id}: {e}")
            if self.cap:
                self.cap.release()
            return False

    def _background_detection_processor(self):
        """NEW: Background thread to handle heavy detection processing"""
        print(f"Background detection processor started for stream {self.stream_id}")
        
        while self.is_running:
            try:
                # Wait for detection events (blocking, but in separate thread)
                detection_data = self.detection_queue.get(timeout=1.0)
                
                if detection_data is None:  # Shutdown signal
                    break
                
                print(f"Stream {self.stream_id}: Processing detection event in background...")
                
                # Now do all the heavy processing without blocking the main detection loop
                self._process_detection_event_heavy(detection_data)
                
                # Mark task as done
                self.detection_queue.task_done()
                
            except queue.Empty:
                continue  # Timeout is normal, keep checking
            except Exception as e:
                print(f"Error in background detection processor for stream {self.stream_id}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"Background detection processor stopped for stream {self.stream_id}")

    def _process_detection_event_heavy(self, detection_data):
        """Process detection event without blocking other streams"""
        try:
            stream_id = detection_data['stream_id']
            stream_name = detection_data['stream_name']
            confidence = detection_data['confidence']
            current_time = detection_data['timestamp']
            current_frame = detection_data['current_frame']
            recent_frames = detection_data['recent_frames']
            rtsp_url = detection_data['rtsp_url']
            
            timestamp_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))
            
            print(f"Stream {stream_id}: Starting background processing for detection at {timestamp_str}")
            
            # CRITICAL: Check stitching BEFORE heavy operations
            should_stitch = stitching_manager.should_stitch_to_existing(stream_id, current_time)
            
            # Lightweight operations first
            incident_id = ""
            incident_status = "active"
            
            # Create event record FIRST (lightweight)
            event = ViolenceEvent(
                timestamp=timestamp_str,
                source_type='stream',
                source_id=str(stream_id),
                filename=stream_name,
                start_time=current_time,
                end_time=current_time + self.detection_interval,
                duration=self.detection_interval,
                confidence=confidence,
                thumbnail_path="",  # Set later
                clip_path="",      # Set later
                incident_status=incident_status,
                incident_id=incident_id,
                metadata=json.dumps({
                    'stream_name': stream_name,
                    'rtsp_url': rtsp_url,
                    'detection_type': 'live_stream',
                    'buffer_size': len(self.rgb_frame_buffer),
                    'model_input_size': self.model_input_size,
                    'temporal_length': self.model_temporal_length,
                    'pipeline_version': 'stitched_stream_v3_async',
                    'frame_timestamp': current_time,
                    'detection_interval': self.detection_interval
                })
            )

            # Save to database (uses connection pool now)
            event_id = event_db.save_event(event)
            stream_db.increment_detection_count(stream_id)

            # Handle incident stitching (per-stream locking now)
            if should_stitch:
                incident = stitching_manager.active_incidents.get(stream_id)
                if incident:
                    incident_id = incident.incident_id
                stitching_manager.extend_incident(stream_id, current_time, confidence, recent_frames, event_id)
            else:
                incident_id = stitching_manager.start_new_incident(
                    stream_id, stream_name, current_time, confidence, recent_frames, event_id
                )
                # Update event with incident_id (quick operation)
                with event_db.connection_pool.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('UPDATE violence_events SET incident_id = ? WHERE id = ?', (incident_id, event_id))
                    conn.commit()

            print(f"Stream {stream_id}: Completed core processing for event {event_id}")

            # Heavy I/O operations at the end (these can be slow without affecting other streams)
            thumbnail_url = self._generate_thumbnail_background(stream_id, current_time, current_frame)
            clip_url = self._generate_clip_background(stream_id, current_time, current_frame)
            
            # Update event with media URLs
            if thumbnail_url or clip_url:
                with event_db.connection_pool.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        'UPDATE violence_events SET thumbnail_path = ?, clip_path = ? WHERE id = ?',
                        (thumbnail_url, clip_url, event_id)
                    )
                    conn.commit()

            # Send notifications last (network I/O)
            self._send_final_notifications_background(event_id, stream_id, stream_name, confidence, 
                                                    timestamp_str, thumbnail_url, clip_url, 
                                                    incident_id, incident_status, should_stitch)

        except Exception as e:
            print(f"Error in background processing for stream {detection_data.get('stream_id', 'unknown')}: {e}")
            import traceback
            traceback.print_exc()

    def _generate_thumbnail_background(self, stream_id, timestamp, frame):
        """Generate thumbnail in background (heavy I/O)"""
        try:
            if frame is None:
                return ""
                
            thumbnail_filename = f"stream_{stream_id}_event_{int(timestamp)}.jpg"
            thumbnail_dir = os.path.join(RESULTS_FOLDER, "stream_thumbnails")
            os.makedirs(thumbnail_dir, exist_ok=True)
            thumbnail_path = os.path.join(thumbnail_dir, thumbnail_filename)
            
            # Resize and save thumbnail
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
            
            return f"/api/results/stream_thumbnails/{thumbnail_filename}"
            
        except Exception as e:
            print(f"Error generating thumbnail for stream {stream_id}: {e}")
            return ""

    def _generate_clip_background(self, stream_id, timestamp, frame):
        """Generate video clip in background (very heavy operation)"""
        try:
            if frame is None:
                return ""
                
            clip_filename = f"stream_{stream_id}_clip_{int(timestamp)}.mp4"
            clips_dir = os.path.join(RESULTS_FOLDER, "stream_clips")
            os.makedirs(clips_dir, exist_ok=True)
            clip_path = os.path.join(clips_dir, clip_filename)
            
            # Create clip frames
            target_width, target_height = 640, 480
            resized_frame = cv2.resize(frame, (target_width, target_height))
            
            # Create 16 frames (4 seconds at 4 FPS)
            clip_frames = [resized_frame.copy() for _ in range(16)]
            
            if len(clip_frames) >= 8:
                # Try H.264 encoding
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                out = cv2.VideoWriter(clip_path, fourcc, 4.0, (target_width, target_height), isColor=True)
                
                if out.isOpened():
                    for frame in clip_frames:
                        out.write(frame)
                    out.release()
                    
                    # Verify file
                    if os.path.exists(clip_path) and os.path.getsize(clip_path) > 2000:
                        test_cap = cv2.VideoCapture(clip_path)
                        if test_cap.isOpened():
                            ret, test_frame = test_cap.read()
                            test_cap.release()
                            
                            if ret and test_frame is not None:
                                return f"/api/results/stream_clips/{clip_filename}"
                            else:
                                os.remove(clip_path)
                        else:
                            os.remove(clip_path)
                    else:
                        if os.path.exists(clip_path):
                            os.remove(clip_path)
                else:
                    out.release()
            
            return ""
            
        except Exception as e:
            print(f"Error generating clip for stream {stream_id}: {e}")
            return ""

    def _send_final_notifications_background(self, event_id, stream_id, stream_name, confidence, 
                                           timestamp_str, thumbnail_url, clip_url, incident_id, 
                                           incident_status, should_stitch):
        """Send final notifications in background (network I/O)"""
        try:
            # Send complete WebSocket notification
            if main_loop and main_loop.is_running():
                payload = {
                    'type': 'violence_detected',
                    'event_id': event_id,
                    'stream_id': stream_id,
                    'stream_name': stream_name,
                    'confidence': confidence,
                    'thumbnail': thumbnail_url,
                    'clip': clip_url,
                    'timestamp': timestamp_str,
                    'incident_id': incident_id,
                    'incident_status': incident_status,
                    'is_ongoing_incident': should_stitch
                }
                
                asyncio.run_coroutine_threadsafe(
                    manager.send_job_update(f"violence_event_{event_id}", payload),
                    main_loop
                )
                print(f"Stream {stream_id}: Sent complete event notification via WebSocket")

            # Send Discord notification
            if discord_notifier and discord_notifier.enabled:
                discord_notifier.send_violence_alert(
                    stream_id=stream_id,
                    stream_name=stream_name,
                    confidence=confidence,
                    timestamp=timestamp_str,
                    thumbnail_path=thumbnail_url,
                    clip_path=clip_url,
                    incident_id=incident_id,
                    is_ongoing=should_stitch
                )
                print(f"Stream {stream_id}: Sent Discord notification")

        except Exception as e:
            print(f"Error sending final notifications for stream {stream_id}: {e}")

    def stop_stream(self):
        """Stop the RTSP stream and cleanup resources"""
        if not self.is_running:
            return
            
        print(f"Stopping stream {self.stream_id}")
        self.is_running = False

        # Finalize any active incidents before stopping
        if stitching_manager:
            try:
                stitching_manager.cleanup_stream_incidents(self.stream_id)
            except Exception as e:
                print(f"Error cleaning up incidents for stream {self.stream_id}: {e}")

        # Signal background thread to stop
        try:
            self.detection_queue.put_nowait(None)  # Shutdown signal
        except:
            pass

        # Clean up capture
        if self.cap:
            try:
                self.cap.release()
                self.cap = None
            except Exception as e:
                print(f"Error releasing capture for stream {self.stream_id}: {e}")

        # Don't join threads during global shutdown
        if not global_shutdown_event.is_set() and not shutdown_in_progress:
            threads_to_join = []
            
            if hasattr(self, 'capture_thread') and self.capture_thread and self.capture_thread.is_alive():
                threads_to_join.append(("capture", self.capture_thread))
            
            if hasattr(self, 'process_thread') and self.process_thread and self.process_thread.is_alive():
                threads_to_join.append(("process", self.process_thread))
            
            if hasattr(self, 'background_thread') and self.background_thread and self.background_thread.is_alive():
                threads_to_join.append(("background", self.background_thread))
            
            for thread_name, thread in threads_to_join:
                try:
                    thread.join(timeout=1.0)
                    if thread.is_alive():
                        print(f"Warning: {thread_name} thread for stream {self.stream_id} did not stop gracefully")
                except Exception as e:
                    print(f"Error joining {thread_name} thread for stream {self.stream_id}: {e}")

        # Clear buffers safely
        try:
            while not self.raw_frame_queue.empty():
                try:
                    self.raw_frame_queue.get_nowait()
                except:
                    break
            
            while not self.detection_queue.empty():
                try:
                    self.detection_queue.get_nowait()
                except:
                    break
            
            self.rgb_frame_buffer.clear()
            
            with self._lock:
                self.last_display_frame = None
                
        except Exception as e:
            print(f"Error clearing buffers for stream {self.stream_id}: {e}")

        # Remove from active streams
        try:
            if self.stream_id in active_streams:
                del active_streams[self.stream_id]
        except Exception as e:
            print(f"Error removing stream {self.stream_id} from active streams: {e}")

        print(f"Stream {self.stream_id} stopped successfully")

    # Keep all other existing methods unchanged...
    def validate_rtsp_url(self, url: str) -> bool:
        """Validate RTSP URL format"""
        try:
            parsed = urlparse(url)
            return parsed.scheme.lower() in ['rtsp', 'rtmp', 'http', 'https'] and parsed.netloc
        except:
            return False

    def preprocess_frame_for_buffer(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess a single frame to match extract_frames() output format"""
        try:
            frame_resized = cv2.resize(frame, self.model_input_size, interpolation=cv2.INTER_LINEAR)

            if len(frame_resized.shape) == 3 and frame_resized.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame_resized.copy()

            if frame_rgb.dtype != np.uint8:
                frame_rgb = frame_rgb.astype(np.uint8)

            return frame_rgb

        except Exception as e:
            print(f"Error preprocessing frame for stream {self.stream_id}: {e}")
            return None

    def _capture_frames(self):
        """Capture frames with better error recovery"""
        consecutive_failures = 0
        max_failures = 10
        reconnect_attempts = 0
        max_reconnect_attempts = 3

        while self.is_running and not global_shutdown_event.is_set():
            try:
                # Check if capture is still valid
                if not self.cap or not self.cap.isOpened():
                    print(f"Stream {self.stream_id}: Reconnecting to {self.rtsp_url}")
                    if self.cap:
                        self.cap.release()
                    
                    self.cap = cv2.VideoCapture(self.rtsp_url)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    if not self.cap.isOpened():
                        reconnect_attempts += 1
                        if reconnect_attempts >= max_reconnect_attempts:
                            print(f"Stream {self.stream_id}: Max reconnection attempts reached")
                            break
                        time.sleep(2.0)  # Wait before retry
                        continue
                    else:
                        reconnect_attempts = 0  # Reset on successful reconnection

                ret, frame = self.cap.read()

                if not ret or frame is None:
                    consecutive_failures += 1
                    print(f"Stream {self.stream_id}: Frame read failed ({consecutive_failures}/{max_failures})")

                    if consecutive_failures >= max_failures:
                        print(f"Stream {self.stream_id}: Too many consecutive failures, will attempt reconnection")
                        if self.cap:
                            self.cap.release()
                            self.cap = None
                        consecutive_failures = 0
                        continue

                    time.sleep(0.1)
                    continue

                consecutive_failures = 0

                with self._lock:
                    self.last_display_frame = frame.copy()
                    self.display_frame_buffer.append(frame.copy())

                try:
                    self.raw_frame_queue.put(frame, block=False)
                except queue.Full:
                    try:
                        self.raw_frame_queue.get_nowait()
                        self.raw_frame_queue.put(frame, block=False)
                    except:
                        pass

                time.sleep(0.033)

            except Exception as e:
                print(f"Exception in capture thread for stream {self.stream_id}: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    break
                time.sleep(0.5)

        print(f"Capture thread for stream {self.stream_id} ended")

    def _process_frames(self):
        """Dedicated thread for processing frames for ML model"""
        last_process_time = 0
        process_interval = 1.0 / self.target_fps
        frame_counter = 0
        last_buffer_add_time = 0
        buffer_interval = 0.2

        while self.is_running:
            try:
                current_time = time.time()

                if current_time - last_process_time < process_interval:
                    time.sleep(0.01)
                    continue

                try:
                    raw_frame = self.raw_frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                frame_counter += 1
                last_process_time = current_time

                if current_time - last_buffer_add_time >= buffer_interval:
                    rgb_frame = self.preprocess_frame_for_buffer(raw_frame)
                    if rgb_frame is not None:
                        self.rgb_frame_buffer.append(rgb_frame)
                        last_buffer_add_time = current_time
                        print(f"Stream {self.stream_id}: Added frame to buffer (size: {len(self.rgb_frame_buffer)}/16)")

                if frame_counter % 60 == 0:
                    self._save_thumbnail()

                if (len(self.rgb_frame_buffer) >= self.model_temporal_length and
                    current_time - self.last_detection_time >= self.detection_interval):
                    self._run_detection()
                    self.last_detection_time = current_time

                if frame_counter % 15 == 0:
                    try:
                        # Don't create async loop, just schedule the coroutine
                        if main_loop and main_loop.is_running():
                            asyncio.run_coroutine_threadsafe(
                                self._send_frame_update(),
                                main_loop
                            )
                    except Exception as e:
                        print(f"Error scheduling frame update for stream {self.stream_id}: {e}")

            except Exception as e:
                print(f"Error in process thread for stream {self.stream_id}: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1.0)

        print(f"Process thread for stream {self.stream_id} ended")

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

            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            return base64.b64encode(buffer).decode('utf-8')

        except Exception as e:
            print(f"Error encoding frame for stream {self.stream_id}: {e}")
            return None

    async def _send_frame_update(self):
        """Send frame update via WebSocket - FIXED"""
        try:
            frame_data = self.get_frame_base64()
            if frame_data and main_loop and main_loop.is_running():
                payload = {
                    'type': 'stream_frame',
                    'stream_id': self.stream_id,
                    'frame': frame_data,
                    'status': 'active',
                    'buffer_size': len(self.rgb_frame_buffer)
                }
                
                # Use the existing main loop instead of creating new one
                asyncio.run_coroutine_threadsafe(
                    manager.send_job_update(f"stream_{self.stream_id}", payload),
                    main_loop
                )
        except Exception as e:
            print(f"Error sending frame update for stream {self.stream_id}: {e}")

    def check_health(self):
        """Check if stream is healthy"""
        try:
            # Check if threads are alive
            capture_alive = self.capture_thread and self.capture_thread.is_alive()
            process_alive = self.process_thread and self.process_thread.is_alive()
            background_alive = hasattr(self, 'background_thread') and self.background_thread and self.background_thread.is_alive()
            
            # Check if we're getting frames
            with self._lock:
                has_recent_frame = self.last_display_frame is not None
            
            return {
                'capture_thread': capture_alive,
                'process_thread': process_alive, 
                'background_thread': background_alive,
                'has_frame': has_recent_frame,
                'queue_size': self.raw_frame_queue.qsize(),
                'buffer_size': len(self.rgb_frame_buffer)
            }
        except Exception as e:
            print(f"Error checking health for stream {self.stream_id}: {e}")
            return {'error': str(e)}

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
            cleanup_old_batches()
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
    global model, event_db, stream_db, stitching_manager, discord_notifier, main_loop
    try:
        # Capture the running event loop
        main_loop = asyncio.get_running_loop()
        print("Main asyncio loop captured.")

        print("Starting up Violence Detection API...")

        # Initialize databases
        print("Initializing databases...")
        event_db = EventDatabase()
        stream_db = StreamDatabase()
        stitching_manager = EventStitchingManager()
        print("Databases and stitching manager initialized successfully")

        # Initialize Discord notifier
        print("Initializing Discord notifications...")
        discord_notifier = DiscordNotifier(DISCORD_WEBHOOK_URL, DISCORD_NOTIFICATIONS_ENABLED)
        if discord_notifier.enabled:
            discord_notifier.send_system_status(
                "Violence Detection System started successfully!", 
                "success"
            )

        # Recover stream states from previous shutdown
        recover_stream_states()

        # Load global model for video upload processing only
        print("Loading global model for video upload processing...")
        try:
            model, _ = load_violence_detection_model(MODEL_PATH)
            print("Global model loaded successfully for video uploads")
        except Exception as e:
            print(f"Failed to load global model: {e}")
            model = None
            print("Video upload processing will be disabled")

        print("Stream models will be loaded individually per-stream as needed")

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
    """Process video using consecutive frame sequences with progress updates"""

    if model is None:
        active_jobs[job_id]['status'] = 'error'
        active_jobs[job_id]['message'] = 'Model not loaded'
        # Send WebSocket update
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(manager.send_job_update(job_id, active_jobs[job_id]))
            loop.close()
        except:
            pass
        return

    if threshold is None:
        threshold = DETECTION_THRESHOLD

    try:
        # Extract metadata
        metadata = get_video_metadata(video_path)
        if metadata is None:
            active_jobs[job_id]['status'] = 'error'
            active_jobs[job_id]['message'] = 'Could not read video file'
            # Send WebSocket update
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(manager.send_job_update(job_id, active_jobs[job_id]))
                loop.close()
            except:
                pass
            return

        active_jobs[job_id]['metadata'] = metadata
        active_jobs[job_id]['status'] = 'processing'
        active_jobs[job_id]['progress'] = 5
        active_jobs[job_id]['message'] = 'Extracting frames'

        # Send WebSocket progress update
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(manager.send_job_update(job_id, active_jobs[job_id]))
            loop.close()
        except:
            pass

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
            # Send WebSocket update
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(manager.send_job_update(job_id, active_jobs[job_id]))
                loop.close()
            except:
                pass
            return

        active_jobs[job_id]['progress'] = 30
        active_jobs[job_id]['message'] = 'Processing sequences'

        # Send WebSocket progress update
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(manager.send_job_update(job_id, active_jobs[job_id]))
            loop.close()
        except:
            pass

        # Determine motion enhancement
        use_motion = hasattr(model, 'use_motion_enhancement') and model.use_motion_enhancement

        # Process each sequence independently with individual timing
        segments = []
        total_sequences = len(sequences)
        first_violence_inference_time = None
        total_inference_time = 0.0
        
        for i, (sequence, (start_time, end_time)) in enumerate(zip(sequences, timestamps)):
            progress = 30 + int(60 * i / total_sequences)
            active_jobs[job_id]['progress'] = progress
            active_jobs[job_id]['message'] = f'Analyzing sequence {i+1}/{total_sequences}'
            
            # Send WebSocket progress update every few sequences to avoid spam
            if i % 3 == 0 or i == total_sequences - 1:
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(manager.send_job_update(job_id, active_jobs[job_id]))
                    loop.close()
                except:
                    pass
            
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
                    'inference_time': inference_time,
                    'start_formatted': f"{int(start_time//60)}:{int(start_time%60):02d}",
                    'end_formatted': f"{int(end_time//60)}:{int(end_time%60):02d}"
                })
                print(f"Violence detected in sequence {i+1}: {start_time:.1f}-{end_time:.1f}s, confidence: {confidence:.3f}, inference: {inference_time:.3f}s")

        active_jobs[job_id]['progress'] = 95
        active_jobs[job_id]['message'] = 'Finalizing results'
        
        # Send WebSocket progress update
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(manager.send_job_update(job_id, active_jobs[job_id]))
            loop.close()
        except:
            pass

        # Merge close segments (within 1 second) while preserving first inference time
        if segments:
            merged_segments = [segments[0]]
            for segment in segments[1:]:
                prev = merged_segments[-1]
                # More intelligent merging: allow gaps up to 3 seconds, or if confidence suggests continuity
                gap = segment['start'] - prev['end']
                should_merge = (gap <= 3.0) or (gap <= 5.0 and min(segment['confidence'], prev['confidence']) > 0.8)
                
                if should_merge:
                    prev['end'] = max(prev['end'], segment['end'])
                    prev['confidence'] = max(prev['confidence'], segment['confidence'])
                    prev['end_formatted'] = f"{int(prev['end']//60)}:{int(prev['end']%60):02d}"
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
                'inference_time': display_inference_time,
                'first_violence_inference_time': first_violence_inference_time,
                'total_processing_time': total_inference_time,
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

        # Send final WebSocket update
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(manager.send_job_update(job_id, active_jobs[job_id]))
            loop.close()
        except:
            pass

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

        # BATCH COMPLETION TRACKING WITH MEDIA DATA
        if 'batch_id' in active_jobs[job_id]:
            batch_result = {
                'has_violence': has_violence,
                'overall_result': {
                    'confidence': float(overall_confidence)
                },
                'job_id': job_id,
                'filename': os.path.basename(video_path),
                'thumbnail': active_jobs[job_id]['thumbnail'],  # Include thumbnail path
                'segments': segments  # Include segments for clip detection
            }
            update_batch_completion(job_id, batch_result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        active_jobs[job_id]['status'] = 'error'
        active_jobs[job_id]['message'] = f'Error: {str(e)}'
        
        # Send WebSocket error update
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(manager.send_job_update(job_id, active_jobs[job_id]))
            loop.close()
        except:
            pass

        # BATCH ERROR TRACKING WITH EMPTY MEDIA
        if job_id in active_jobs and 'batch_id' in active_jobs[job_id]:
            error_result = {
                'has_violence': False,
                'overall_result': {'confidence': 0.0},
                'job_id': job_id,
                'filename': 'Error processing',
                'error': str(e),
                'thumbnail': '',  # Empty thumbnail for errors
                'segments': []    # Empty segments for errors
            }
            update_batch_completion(job_id, error_result)

# API Routes
@app.get("/")
async def root():
    return {"message": "Violence Detection API with Event Storage", "docs": "/docs", "status": "running"}

@app.get("/api/jobs", response_model=List[JobStatus])
async def get_all_jobs():
    """Get all active jobs ordered by most recent first"""
    # Sort jobs by timestamp (most recent first)
    sorted_jobs = sorted(
        active_jobs.values(), 
        key=lambda x: x.get('timestamp', ''), 
        reverse=True
    )
    
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
        for job in sorted_jobs
    ]

@app.get("/api/history")
async def get_history():
    """Get processing history ordered by most recent first"""
    # Reload history from file to ensure consistency
    await load_history_from_file()
    
    # Sort history by timestamp (most recent first)
    sorted_history = sorted(
        results_history.values(),
        key=lambda x: x.get('timestamp', ''),
        reverse=True
    )
    
    return {"history": sorted_history}

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
@limiter.limit("10/minute")  # 5 uploads per minute per IP
async def upload_file(
    request: Request,
    file: Optional[UploadFile] = File(None),
    video_path: Optional[str] = Form(None)
):
    """Handle file upload or local path with rate limiting and resource management"""

    # Check if too many active jobs (limit to 3 like before)
    active_count = sum(1 for job in active_jobs.values()
                       if job['status'] in ['queued', 'processing'])
    if active_count >= 10:
        raise HTTPException(
            status_code=429,
            detail=f"Too many active jobs ({active_count}/10). Please wait."
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

@app.post("/api/upload-batch")
@limiter.limit("5/minute")
async def upload_batch(
    request: Request,
    files: List[UploadFile] = File(...),
):
    """Handle multiple file uploads with batch tracking"""
    
    # Check active jobs limit
    active_count = sum(1 for job in active_jobs.values()
                       if job['status'] in ['queued', 'processing'])
    if active_count >= 10:
        raise HTTPException(
            status_code=429,
            detail=f"Too many active jobs ({active_count}/10). Please wait."
        )
    
    # Validate all files first
    validated_files = []
    for file in files:
        if not file.filename or not allowed_file(file.filename):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type: {file.filename}"
            )
        
        # Check file size
        file.file.seek(0, 2)  # Seek to end
        size = file.file.tell()
        file.file.seek(0)  # Reset
        
        if size > MAX_CONTENT_LENGTH:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large: {file.filename}"
            )
        
        validated_files.append((file, size))
    
    # Create batch tracking
    batch_id = create_batch_id()
    job_ids = []
    filenames = []
    
    try:
        # Process all files
        for file, size in validated_files:
            job_id = str(uuid.uuid4())
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{filename}")
            
            # Save file
            contents = await file.read()
            with open(file_path, "wb") as f:
                f.write(contents)
            
            # Create job
            active_jobs[job_id] = {
                'id': job_id,
                'file_path': file_path,
                'filename': filename,
                'status': 'queued',
                'progress': 0,
                'message': 'Queued for processing',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'batch_id': batch_id  # Add batch tracking
            }
            
            job_ids.append(job_id)
            filenames.append(filename)
        
        # Track batch
        track_batch_upload(batch_id, job_ids, filenames)
        
        # Send Discord batch start notification
        if discord_notifier and discord_notifier.enabled:
            discord_notifier.send_batch_upload_start(batch_id, len(job_ids), filenames)
        
        # Start processing all videos
        for job_id, (_, _) in zip(job_ids, validated_files):
            job_data = active_jobs[job_id]
            threading.Thread(target=process_video_sync, args=(job_id, job_data['file_path'])).start()
        
        return {
            "success": True,
            "message": f"Batch upload successful - {len(job_ids)} videos queued",
            "batch_id": batch_id,
            "job_ids": job_ids
        }
        
    except Exception as e:
        # Cleanup on error
        for job_id in job_ids:
            if job_id in active_jobs:
                del active_jobs[job_id]
        
        raise HTTPException(status_code=500, detail=str(e))

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

@app.get("/api/batch/{batch_id}")
async def get_batch_status(batch_id: str):
    """Get status of a batch upload"""
    try:
        with batch_lock:
            if batch_id not in batch_uploads:
                raise HTTPException(status_code=404, detail="Batch not found")
            
            batch_data = batch_uploads[batch_id]
            
            # Get current status of all jobs
            job_statuses = []
            for job_id in batch_data['job_ids']:
                if job_id in active_jobs:
                    job = active_jobs[job_id]
                    job_statuses.append({
                        'job_id': job_id,
                        'filename': job['filename'],
                        'status': job['status'],
                        'progress': job['progress']
                    })
            
            return {
                'batch_id': batch_id,
                'total_count': batch_data['total_count'],
                'completed_count': batch_data['completed_count'],
                'progress_percent': (batch_data['completed_count'] / batch_data['total_count']) * 100,
                'is_complete': batch_data['completed_count'] >= batch_data['total_count'],
                'job_statuses': job_statuses
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
                    media_type='video/mp4'
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
            
            # ADD THIS DISCORD NOTIFICATION
            if discord_notifier and discord_notifier.enabled:
                discord_notifier.send_system_status(
                    f"Stream **{stream_data['name']}** (ID: {stream_id}) started monitoring",
                    "info"
                )
            
            return {"success": True, "message": "Stream started"}
        else:
            stream_db.update_stream_status(stream_id, 'error')
            
            # ADD THIS DISCORD ERROR NOTIFICATION  
            if discord_notifier and discord_notifier.enabled:
                discord_notifier.send_system_status(
                    f"Failed to start stream **{stream_data['name']}** (ID: {stream_id})",
                    "error"
                )
            
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
        "max_concurrent_jobs": 10,
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
        ''', (yesterday,))
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
                    media_type='video/mp4'
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
                
        # Get incident info (new columns are at end)
        incident_status = event[12] if len(event) > 12 else "completed"
        incident_id = event[13] if len(event) > 13 else ""
        
        # Get stream information
        stream_info = None
        if stream_db:
            streams = stream_db.get_streams()
            stream_info = next((s for s in streams if str(s['id']) == event[3]), None)
        
        stream_name = stream_info['name'] if stream_info else event[4]
        
        # Format duration
        duration = event[7]  # duration column
        duration_formatted = f"{int(duration//60)}:{int(duration%60):02d}"
        
        # Create segments from timeline data (if available from stitched incident)
        segments = []
        timeline_segments = metadata_dict.get('timeline_segments', [])
        
        if timeline_segments and incident_status == 'completed':
            # Use detailed timeline from stitched incident
            for seg in timeline_segments:
                segments.append({
                    'start': seg['start'],
                    'end': seg['end'], 
                    'confidence': seg['confidence'],
                    'inference_time': 3.0,  # detection_interval
                    'start_formatted': f"{int(seg['start']//60)}:{int(seg['start']%60):02d}",
                    'end_formatted': f"{int(seg['end']//60)}:{int(seg['end']%60):02d}"
                })
        else:
            # Fallback for single detection or active incidents
            segments = [{
                'start': 0.0,  # Relative to incident start
                'end': duration,
                'confidence': float(event[8]),
                'inference_time': metadata_dict.get('detection_interval', 3.0),
                'start_formatted': f"0:00",
                'end_formatted': f"{int(duration//60)}:{int(duration%60):02d}"
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
            'clip_path': event[10] if event[10] else None,   # Also provide as clip_path
            'thumbnail': event[9] if event[9] else None,     # thumbnail_path
            'incident_status': incident_status,
            'incident_id': incident_id,
            
            # Overall result info with enhanced timing
            'overall_result': {
                'is_fight': True,
                'confidence': float(event[8]),
                'inference_time': detection_interval,  
                'first_violence_inference_time': detection_interval,
                'total_processing_time': duration,  # Full incident duration
                'sequences_processed': metadata_dict.get('total_detections', 1)
            },
            
            # Segments data
            'segments': segments,
            'violence_duration': duration,
            'violence_percentage': 100.0,  # Assume entire clip is violent
            
            # Metadata for video player
            'metadata': {
                'duration': duration,
                'duration_formatted': duration_formatted,
                'width': metadata_dict.get('frame_width', 640),
                'height': metadata_dict.get('frame_height', 480),  
                'fps': metadata_dict.get('fps', 4.0),
                'frame_count': int(duration * metadata_dict.get('fps', 4.0)),
                'source_type': 'live_stream',
                'timeline_segments': timeline_segments,  # Pass timeline to frontend
                'total_detections': metadata_dict.get('total_detections', 1)
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

@app.get("/api/stitched-incidents")
async def get_stitched_incidents(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    stream_id: Optional[int] = None,
    limit: Optional[int] = 50
):
    """Get stitched incident summaries with filtering"""
    try:
        if not start_date:
            # Default to last 24 hours
            start_date = (datetime.now() - timedelta(hours=24)).strftime('%Y-%m-%d %H:%M:%S')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        incidents = event_db.get_stitched_incidents(start_date, end_date, stream_id, limit)
        
        return {
            'incidents': incidents,
            'count': len(incidents),
            'filters': {
                'start_date': start_date,
                'end_date': end_date,
                'stream_id': stream_id
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stitched-incident/{incident_id}")
async def get_stitched_incident_details(incident_id: str):
    """Get detailed information about a stitched incident"""
    try:
        conn = sqlite3.connect(event_db.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM stitched_incidents WHERE incident_id = ?', (incident_id,))
        incident = cursor.fetchone()
        
        if not incident:
            raise HTTPException(status_code=404, detail="Incident not found")
        
        # Get individual events for this incident
        cursor.execute('''
            SELECT * FROM violence_events 
            WHERE incident_id = ? 
            ORDER BY timestamp ASC
        ''', (incident_id,))
        events = cursor.fetchall()
        
        conn.close()
        
        # Format response
        incident_data = {
            'id': incident[0],
            'incident_id': incident[1],
            'stream_id': incident[2],
            'stream_name': incident[3],
            'start_timestamp': incident[4],
            'end_timestamp': incident[5],
            'total_duration': incident[6],
            'detection_count': incident[7],
            'avg_confidence': incident[8],
            'max_confidence': incident[9],
            'stitched_clip_path': incident[10],
            'timeline_data': json.loads(incident[11]) if incident[11] else [],
            'event_ids': json.loads(incident[12]) if incident[12] else [],
            'individual_events': [
                {
                    'id': row[0],
                    'timestamp': row[1],
                    'start_time': row[5],
                    'end_time': row[6],
                    'duration': row[7],
                    'confidence': row[8],
                    'thumbnail_path': row[9],
                    'clip_path': row[10]
                }
                for row in events
            ]
        }
        
        return incident_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stitched-stats")
async def get_stitched_incident_stats():
    """Get statistics for stitched incidents"""
    try:
        if not event_db:
            return {
                'error': 'Database not initialized',
                'total_incidents': 0,
                'incidents_24h': 0,
                'avg_duration': 0,
                'avg_detections_per_incident': 0
            }

        conn = sqlite3.connect(event_db.db_path)
        cursor = conn.cursor()

        # Total incidents
        cursor.execute("SELECT COUNT(*) FROM stitched_incidents")
        total_incidents = cursor.fetchone()[0]

        # Incidents in last 24 hours
        yesterday = (datetime.now() - timedelta(hours=24)).strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('''
            SELECT COUNT(*) FROM stitched_incidents 
            WHERE start_timestamp >= ?
        ''', (yesterday,))
        incidents_24h = cursor.fetchone()[0]

        # Average duration
        cursor.execute("SELECT AVG(total_duration) FROM stitched_incidents")
        avg_duration_result = cursor.fetchone()[0]
        avg_duration = avg_duration_result if avg_duration_result else 0

        # Average detections per incident
        cursor.execute("SELECT AVG(detection_count) FROM stitched_incidents")
        avg_detections_result = cursor.fetchone()[0]
        avg_detections_per_incident = avg_detections_result if avg_detections_result else 0

        conn.close()

        return {
            'total_incidents': total_incidents,
            'incidents_24h': incidents_24h,
            'avg_duration': round(avg_duration, 2) if avg_duration else 0,
            'avg_detections_per_incident': round(avg_detections_per_incident, 1) if avg_detections_per_incident else 0
        }

    except Exception as e:
        return {
            'error': str(e),
            'total_incidents': 0,
            'incidents_24h': 0,
            'avg_duration': 0,
            'avg_detections_per_incident': 0
        }

@app.get("/api/incident-health")
async def check_incident_health():
    """Check for stuck active incidents"""
    try:
        conn = sqlite3.connect(event_db.db_path)
        cursor = conn.cursor()
        
        # Check for old active incidents (> 1 hour old)
        one_hour_ago = datetime.now() - timedelta(hours=1)
        cursor.execute("""
            SELECT incident_id, COUNT(*) as event_count, MIN(timestamp) as oldest
            FROM violence_events 
            WHERE incident_status = 'active' 
            AND timestamp < ?
            GROUP BY incident_id
        """, (one_hour_ago.strftime('%Y-%m-%d %H:%M:%S'),))
        
        stuck_incidents = cursor.fetchall()
        conn.close()
        
        return {
            'stuck_incidents': len(stuck_incidents),
            'incidents': [
                {
                    'incident_id': row[0],
                    'event_count': row[1], 
                    'oldest_event': row[2]
                } for row in stuck_incidents
            ]
        }
    except Exception as e:
        return {'error': str(e)}

@app.get("/api/incident-result/{incident_id}")
async def get_incident_result(incident_id: str):
    """Get incident data formatted for ResultsViewer compatibility"""
    try:
        if not event_db:
            raise HTTPException(status_code=500, detail="Database not initialized")
            
        conn = sqlite3.connect(event_db.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM stitched_incidents WHERE incident_id = ?', (incident_id,))
        incident = cursor.fetchone()
        conn.close()
        
        if not incident:
            raise HTTPException(status_code=404, detail="Incident not found")
        
        # Parse timeline data
        timeline_data = json.loads(incident[11]) if incident[11] else []
        event_ids = json.loads(incident[12]) if incident[12] else []
        
        # Create segments from timeline data
        segments = []
        for seg in timeline_data:
            segments.append({
                'start': seg['start'],
                'end': seg['end'], 
                'confidence': seg['confidence'],
                'inference_time': 3.0,  # detection_interval
                'start_formatted': f"{int(seg['start']//60)}:{int(seg['start']%60):02d}",
                'end_formatted': f"{int(seg['end']//60)}:{int(seg['end']%60):02d}"
            })
        
        # Format result to match ResultsViewer expectations
        result = {
            'job_id': f"incident_{incident[0]}",
            'filename': f"{incident[3]} - Security Incident",  # stream_name
            'timestamp': incident[4],  # start_timestamp
            'has_violence': True,
            'video_path': incident[10] if incident[10] else None,  # stitched_clip_path
            'clip_path': incident[10] if incident[10] else None,
            'thumbnail': incident[10].replace('.mp4', '_thumb.jpg') if incident[10] else None,
            'incident_id': incident[1],  # incident_id
            'incident_status': 'completed',
            
            # Overall result info
            'overall_result': {
                'is_fight': True,
                'confidence': float(incident[9]),  # max_confidence
                'inference_time': incident[6],  # total_duration
                'first_violence_inference_time': 3.0,
                'total_processing_time': incident[6],
                'sequences_processed': incident[7]  # detection_count
            },
            
            # Segments data
            'segments': segments,
            'violence_duration': incident[6],  # total_duration
            'violence_percentage': 100.0,
            
            # Metadata for video player
            'metadata': {
                'duration': incident[6],  # total_duration
                'duration_formatted': f"{int(incident[6]//60)}:{int(incident[6]%60):02d}",
                'width': 640,
                'height': 480,
                'fps': 4.0,
                'frame_count': int(incident[6] * 4.0),
                'source_type': 'stitched_incident',
                'timeline_segments': timeline_data,
                'total_detections': incident[7]  # detection_count
            },
            
            # Model information
            'model_info': {
                'architecture': 'X3D-M (Stitched Incident)',
                'motion_enhancement': True,
                'input_frames': 16,
                'input_resolution': '336x336',
                'analysis_method': 'incident_stitching',
                'hop_seconds': 3.0,
                'total_sequences_processed': incident[7],
                'source_stream': incident[3],  # stream_name
                'stream_id': incident[2]  # stream_id
            },
            
            # Incident-specific metadata
            'incident_metadata': {
                'incident_id': incident[1],
                'stream_id': incident[2],
                'stream_name': incident[3],
                'start_timestamp': incident[4],
                'end_timestamp': incident[5],
                'detection_count': incident[7],
                'avg_confidence': incident[8],
                'max_confidence': incident[9],
                'individual_events': len(event_ids),
                'event_ids': event_ids
            },
            
            # Processing statistics
            'processing_stats': {
                'total_sequences': incident[7],
                'violent_sequences': incident[7],
                'total_inference_time': incident[6],
                'first_violence_time': 3.0,
                'avg_inference_per_sequence': incident[6] / incident[7] if incident[7] > 0 else 0
            }
        }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting incident result {incident_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.options("/api/uploads/{filename}")
@app.options("/api/results/{filename}")  
@app.options("/api/results/clips/{filename}")
@app.options("/api/results/stream_clips/{filename}")
async def handle_video_options():
    """Handle CORS preflight requests for video endpoints"""
    from fastapi.responses import Response
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