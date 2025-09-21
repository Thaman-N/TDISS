"""
Infrastructure managers for the batch processing system.
This file contains critical infrastructure components to address the issues
identified in the implementation plan.
"""

import os
import time
import gc
import sqlite3
import threading
import queue
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import cv2
import torch
import numpy as np
from urllib.parse import urlparse
from collections import defaultdict


@dataclass
class RTSPConnection:
    """Represents an RTSP connection with retry logic and health monitoring"""
    stream_id: int
    rtsp_url: str
    cap: Optional[cv2.VideoCapture]
    created_at: float
    last_activity: float
    retry_count: int = 0
    max_retries: int = 3
    status: str = 'connecting'  # connecting, active, error, reconnecting
    consecutive_failures: int = 0
    bandwidth_usage: float = 0.0  # MB/s estimate


class RTSPConnectionManager:
    """
    Centralized RTSP connection management with pooling, retry logic,
    and bandwidth throttling across streams.
    
    Addresses: Network bandwidth saturation, connection management problems
    """
    
    def __init__(self, max_connections: int = 12, bandwidth_limit_mbps: float = 50.0):
        self.max_connections = max_connections
        self.bandwidth_limit_mbps = bandwidth_limit_mbps
        self.connections: Dict[int, RTSPConnection] = {}
        self._lock = threading.RLock()
        self.total_bandwidth_usage = 0.0
        self.connection_timeouts = {
            'open_timeout': 10000,  # 10 seconds
            'read_timeout': 5000,   # 5 seconds
        }
        
        # Connection monitoring
        self.health_check_interval = 30.0  # seconds
        self.last_health_check = 0.0
        
        logging.info("RTSPConnectionManager initialized")
    
    def create_connection(self, stream_id: int, rtsp_url: str) -> bool:
        """Create new RTSP connection with proper error handling"""
        with self._lock:
            if len(self.connections) >= self.max_connections:
                logging.warning(f"Max connections ({self.max_connections}) reached")
                return False
            
            if stream_id in self.connections:
                logging.warning(f"Connection for stream {stream_id} already exists")
                return False
            
            if not self._validate_rtsp_url(rtsp_url):
                logging.error(f"Invalid RTSP URL: {rtsp_url}")
                return False
            
            # Check bandwidth availability
            if self.total_bandwidth_usage >= self.bandwidth_limit_mbps:
                logging.warning("Bandwidth limit reached, cannot create new connection")
                return False
            
            connection = RTSPConnection(
                stream_id=stream_id,
                rtsp_url=rtsp_url,
                cap=None,
                created_at=time.time(),
                last_activity=time.time()
            )
            
            # Attempt to establish connection
            if self._establish_connection(connection):
                self.connections[stream_id] = connection
                logging.info(f"RTSP connection established for stream {stream_id}")
                return True
            else:
                logging.error(f"Failed to establish RTSP connection for stream {stream_id}")
                return False
    
    def _validate_rtsp_url(self, url: str) -> bool:
        """Validate RTSP URL format and accessibility"""
        try:
            parsed = urlparse(url)
            return parsed.scheme.lower() in ['rtsp', 'rtmp', 'http', 'https'] and parsed.netloc
        except Exception:
            return False
    
    def _establish_connection(self, connection: RTSPConnection) -> bool:
        """Establish OpenCV VideoCapture connection with timeouts"""
        try:
            connection.status = 'connecting'
            
            # Create VideoCapture with FFMPEG backend for better RTSP support
            cap = cv2.VideoCapture(connection.rtsp_url, cv2.CAP_FFMPEG)
            
            if not cap.isOpened():
                # Fallback to default backend
                cap = cv2.VideoCapture(connection.rtsp_url)
            
            if not cap.isOpened():
                connection.status = 'error'
                return False
            
            # Configure capture properties
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
            
            # Set timeouts if supported
            try:
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.connection_timeouts['open_timeout'])
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.connection_timeouts['read_timeout'])
            except Exception:
                pass  # Ignore if not supported
            
            # Test frame reading
            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                cap.release()
                connection.status = 'error'
                return False
            
            # Estimate bandwidth usage (rough calculation)
            height, width = test_frame.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            connection.bandwidth_usage = (width * height * 3 * fps) / (1024 * 1024 * 8)  # MB/s
            
            connection.cap = cap
            connection.status = 'active'
            connection.last_activity = time.time()
            connection.retry_count = 0
            connection.consecutive_failures = 0
            
            # Update total bandwidth usage
            self.total_bandwidth_usage += connection.bandwidth_usage
            
            return True
            
        except Exception as e:
            logging.error(f"Error establishing connection for stream {connection.stream_id}: {e}")
            connection.status = 'error'
            return False
    
    def get_connection(self, stream_id: int) -> Optional[RTSPConnection]:
        """Get connection for stream with health check"""
        with self._lock:
            if stream_id not in self.connections:
                return None
            
            connection = self.connections[stream_id]
            
            # Perform health check if needed
            current_time = time.time()
            if current_time - self.last_health_check > self.health_check_interval:
                self._health_check_connection(connection)
                self.last_health_check = current_time
            
            if connection.status == 'active':
                connection.last_activity = current_time
                return connection
            
            return None
    
    def _health_check_connection(self, connection: RTSPConnection):
        """Perform health check on connection"""
        if not connection.cap or not connection.cap.isOpened():
            connection.status = 'error'
            return
        
        # Try to read a frame to test connectivity
        try:
            ret, frame = connection.cap.read()
            if ret and frame is not None:
                connection.consecutive_failures = 0
                connection.status = 'active'
            else:
                connection.consecutive_failures += 1
                if connection.consecutive_failures >= 3:
                    connection.status = 'error'
        except Exception:
            connection.consecutive_failures += 1
            connection.status = 'error'
    
    def remove_connection(self, stream_id: int):
        """Remove and cleanup connection"""
        with self._lock:
            if stream_id in self.connections:
                connection = self.connections[stream_id]
                if connection.cap:
                    try:
                        connection.cap.release()
                    except Exception as e:
                        logging.error(f"Error releasing connection for stream {stream_id}: {e}")
                
                # Update bandwidth usage
                self.total_bandwidth_usage = max(0, self.total_bandwidth_usage - connection.bandwidth_usage)
                
                del self.connections[stream_id]
                logging.info(f"Removed connection for stream {stream_id}")
    
    def get_bandwidth_usage(self) -> Dict[str, float]:
        """Get current bandwidth usage statistics"""
        with self._lock:
            return {
                'total_usage_mbps': self.total_bandwidth_usage,
                'limit_mbps': self.bandwidth_limit_mbps,
                'utilization_percent': (self.total_bandwidth_usage / self.bandwidth_limit_mbps) * 100,
                'active_connections': len([c for c in self.connections.values() if c.status == 'active'])
            }


class GPUMemoryManager:
    """
    GPU memory management with allocation tracking, cleanup, and batch size adaptation.
    
    Addresses: GPU memory exhaustion, memory leaks, batch processing memory requirements
    """
    
    def __init__(self, max_gpu_memory_gb: float = 6.0):
        self.max_gpu_memory_gb = max_gpu_memory_gb
        self.max_gpu_memory_bytes = int(max_gpu_memory_gb * 1024 ** 3)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_cuda = self.device.type == 'cuda'
        self._lock = threading.Lock()
        
        # Memory tracking
        self.allocation_history = []
        self.last_cleanup = 0.0
        self.cleanup_interval = 10.0  # seconds
        
        # Adaptive batch sizing
        self.base_batch_size = 8
        self.min_batch_size = 1
        self.max_batch_size = 16
        self.current_batch_size = self.base_batch_size
        
        if self.is_cuda:
            self._initialize_cuda_monitoring()
        
        logging.info(f"GPUMemoryManager initialized for device: {self.device}")
    
    def _initialize_cuda_monitoring(self):
        """Initialize CUDA memory monitoring"""
        try:
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            total_memory = torch.cuda.get_device_properties(0).total_memory
            
            logging.info(f"GPU Total Memory: {total_memory / 1024**3:.2f} GB")
            logging.info(f"GPU Initial Allocated: {initial_memory / 1024**3:.2f} GB")
            
        except Exception as e:
            logging.error(f"Error initializing CUDA monitoring: {e}")
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory information"""
        if not self.is_cuda:
            return {'device': 'cpu', 'allocated_gb': 0, 'available_gb': 0}
        
        try:
            allocated = torch.cuda.memory_allocated()
            cached = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            
            return {
                'device': 'cuda',
                'allocated_gb': allocated / 1024**3,
                'cached_gb': cached / 1024**3,
                'total_gb': total / 1024**3,
                'available_gb': (total - allocated) / 1024**3,
                'utilization_percent': (allocated / total) * 100
            }
        except Exception as e:
            logging.error(f"Error getting GPU memory info: {e}")
            return {'device': 'cuda', 'error': str(e)}
    
    def check_memory_pressure(self) -> bool:
        """Check if GPU memory pressure is high"""
        if not self.is_cuda:
            return False
        
        try:
            memory_info = self.get_memory_info()
            return memory_info.get('utilization_percent', 0) > 85.0
        except Exception:
            return True  # Assume pressure if we can't check
    
    def adaptive_batch_size(self, target_streams: int) -> int:
        """Calculate adaptive batch size based on memory pressure and stream count"""
        with self._lock:
            if not self.is_cuda:
                return min(target_streams, self.base_batch_size)
            
            try:
                memory_info = self.get_memory_info()
                utilization = memory_info.get('utilization_percent', 50)
                
                # Adjust batch size based on memory pressure
                if utilization > 90:
                    self.current_batch_size = max(self.min_batch_size, self.current_batch_size - 2)
                elif utilization > 75:
                    self.current_batch_size = max(self.min_batch_size, self.current_batch_size - 1)
                elif utilization < 50:
                    self.current_batch_size = min(self.max_batch_size, self.current_batch_size + 1)
                
                # Don't exceed number of available streams
                return min(target_streams, self.current_batch_size)
                
            except Exception as e:
                logging.error(f"Error calculating adaptive batch size: {e}")
                return min(target_streams, self.min_batch_size)
    
    def cleanup_gpu_memory(self, force: bool = False):
        """Cleanup GPU memory if needed"""
        current_time = time.time()
        
        if not force and current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        if not self.is_cuda:
            return
        
        try:
            with self._lock:
                # Force garbage collection
                gc.collect()
                
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
                # Synchronize to ensure all operations are complete
                torch.cuda.synchronize()
                
                self.last_cleanup = current_time
                
                logging.debug("GPU memory cleanup completed")
                
        except Exception as e:
            logging.error(f"Error during GPU memory cleanup: {e}")
    
    def allocate_batch_memory(self, batch_size: int, tensor_shape: Tuple) -> bool:
        """Check if we can allocate memory for a batch"""
        if not self.is_cuda:
            return True
        
        try:
            # Estimate memory requirement
            # Each tensor: batch_size * elements * 4 bytes (float32)
            elements_per_item = np.prod(tensor_shape)
            estimated_bytes = batch_size * elements_per_item * 4
            
            memory_info = self.get_memory_info()
            available_bytes = memory_info.get('available_gb', 0) * 1024**3
            
            return estimated_bytes < (available_bytes * 0.8)  # 80% safety margin
            
        except Exception as e:
            logging.error(f"Error checking batch memory allocation: {e}")
            return False


class FileSystemManager:
    """
    File system organization and cleanup with hierarchical structure and rotation policies.
    
    Addresses: File system scaling problems, storage space explosion
    """
    
    def __init__(self, base_results_dir: str = "results", max_storage_gb: float = 10.0):
        self.base_results_dir = Path(base_results_dir)
        self.max_storage_gb = max_storage_gb
        self.max_storage_bytes = int(max_storage_gb * 1024**3)
        
        # Directory structure
        self.directories = {
            'thumbnails': self.base_results_dir / 'thumbnails',
            'clips': self.base_results_dir / 'clips', 
            'stitched_clips': self.base_results_dir / 'stitched_clips',
            'temp': self.base_results_dir / 'temp'
        }
        
        # Cleanup policies
        self.cleanup_policies = {
            'thumbnails': {'max_age_days': 7, 'max_files_per_stream': 100},
            'clips': {'max_age_days': 3, 'max_files_per_stream': 50},
            'stitched_clips': {'max_age_days': 30, 'max_files_per_stream': 20},
            'temp': {'max_age_days': 1, 'max_files_per_stream': 10}
        }
        
        self._lock = threading.Lock()
        self.last_cleanup = 0.0
        self.cleanup_interval = 3600.0  # 1 hour
        
        self._initialize_directories()
        logging.info("FileSystemManager initialized")
    
    def _initialize_directories(self):
        """Initialize directory structure"""
        try:
            for dir_name, dir_path in self.directories.items():
                dir_path.mkdir(parents=True, exist_ok=True)
                logging.info(f"Directory initialized: {dir_path}")
        except Exception as e:
            logging.error(f"Error initializing directories: {e}")
    
    def get_organized_path(self, file_type: str, stream_id: int, filename: str) -> Path:
        """Get organized file path with date-based hierarchy"""
        if file_type not in self.directories:
            raise ValueError(f"Unknown file type: {file_type}")
        
        # Create date-based hierarchy: type/YYYY/MM/DD/stream_id/filename
        now = datetime.now()
        date_path = self.directories[file_type] / f"{now.year}" / f"{now.month:02d}" / f"{now.day:02d}" / f"stream_{stream_id}"
        
        # Ensure directory exists
        date_path.mkdir(parents=True, exist_ok=True)
        
        return date_path / filename
    
    def save_thumbnail(self, stream_id: int, frame: np.ndarray, timestamp: float = None) -> Optional[str]:
        """Save thumbnail with organized path"""
        try:
            timestamp = timestamp or time.time()
            filename = f"thumb_{int(timestamp)}.jpg"
            file_path = self.get_organized_path('thumbnails', stream_id, filename)
            
            # Resize for thumbnail (max 400px)
            height, width = frame.shape[:2]
            max_dim = 400
            if height > width:
                new_height = max_dim
                new_width = int(width * (max_dim / height))
            else:
                new_width = max_dim
                new_height = int(height * (max_dim / width))
            
            thumbnail = cv2.resize(frame, (new_width, new_height))
            
            # Save with good quality
            success = cv2.imwrite(str(file_path), thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            if success:
                # Return relative path for web access
                relative_path = file_path.relative_to(self.base_results_dir)
                return f"/api/results/{relative_path.as_posix()}"
            
            return None
            
        except Exception as e:
            logging.error(f"Error saving thumbnail for stream {stream_id}: {e}")
            return None
    
    def save_clip(self, stream_id: int, frames: List[np.ndarray], timestamp: float = None, 
                  fps: float = 4.0) -> Optional[str]:
        """Save video clip with organized path"""
        try:
            timestamp = timestamp or time.time()
            filename = f"clip_{int(timestamp)}.mp4"
            file_path = self.get_organized_path('clips', stream_id, filename)
            
            if not frames:
                return None
            
            # Standard web resolution (divisible by 16)
            target_width, target_height = 640, 480
            
            # Create video writer with H.264
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(str(file_path), fourcc, fps, (target_width, target_height), isColor=True)
            
            if not out.isOpened():
                return None
            
            # Write frames
            for frame in frames:
                if len(frame.shape) == 3:
                    resized_frame = cv2.resize(frame, (target_width, target_height))
                    out.write(resized_frame)
            
            out.release()
            
            # Verify file was created successfully
            if file_path.exists() and file_path.stat().st_size > 2000:
                relative_path = file_path.relative_to(self.base_results_dir)
                return f"/api/results/{relative_path.as_posix()}"
            
            return None
            
        except Exception as e:
            logging.error(f"Error saving clip for stream {stream_id}: {e}")
            return None
    
    def cleanup_old_files(self, force: bool = False):
        """Cleanup old files based on policies"""
        current_time = time.time()
        
        if not force and current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        try:
            with self._lock:
                total_removed = 0
                total_bytes_freed = 0
                
                for dir_type, dir_path in self.directories.items():
                    if not dir_path.exists():
                        continue
                    
                    policy = self.cleanup_policies.get(dir_type, {})
                    max_age_days = policy.get('max_age_days', 7)
                    max_age_seconds = max_age_days * 24 * 3600
                    
                    # Find old files
                    for file_path in dir_path.rglob('*'):
                        if not file_path.is_file():
                            continue
                        
                        try:
                            file_age = current_time - file_path.stat().st_mtime
                            
                            if file_age > max_age_seconds:
                                file_size = file_path.stat().st_size
                                file_path.unlink()
                                total_removed += 1
                                total_bytes_freed += file_size
                                
                        except Exception as e:
                            logging.error(f"Error removing file {file_path}: {e}")
                
                # Remove empty directories
                for dir_path in self.directories.values():
                    self._remove_empty_dirs(dir_path)
                
                self.last_cleanup = current_time
                
                if total_removed > 0:
                    logging.info(f"Cleanup completed: removed {total_removed} files, "
                               f"freed {total_bytes_freed / 1024**2:.1f} MB")
                
        except Exception as e:
            logging.error(f"Error during file cleanup: {e}")
    
    def _remove_empty_dirs(self, dir_path: Path):
        """Remove empty directories recursively"""
        try:
            if not dir_path.exists() or not dir_path.is_dir():
                return
            
            # Remove empty subdirectories first
            for subdir in dir_path.iterdir():
                if subdir.is_dir():
                    self._remove_empty_dirs(subdir)
            
            # Try to remove this directory if it's empty
            try:
                if not any(dir_path.iterdir()):  # Directory is empty
                    dir_path.rmdir()
            except OSError:
                pass  # Directory not empty or other error
                
        except Exception as e:
            logging.error(f"Error removing empty directories: {e}")
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage usage information"""
        try:
            total_size = 0
            file_counts = {}
            
            for dir_type, dir_path in self.directories.items():
                if not dir_path.exists():
                    file_counts[dir_type] = 0
                    continue
                
                dir_size = 0
                file_count = 0
                
                for file_path in dir_path.rglob('*'):
                    if file_path.is_file():
                        dir_size += file_path.stat().st_size
                        file_count += 1
                
                total_size += dir_size
                file_counts[dir_type] = file_count
            
            return {
                'total_size_gb': total_size / 1024**3,
                'max_size_gb': self.max_storage_gb,
                'utilization_percent': (total_size / self.max_storage_bytes) * 100,
                'file_counts': file_counts
            }
            
        except Exception as e:
            logging.error(f"Error getting storage info: {e}")
            return {'error': str(e)}


# Shared resource pool for database connections
class DatabaseConnectionPool:
    """
    Connection pooling for SQLite with batch operations support.
    
    Addresses: Database bottlenecks, connection overhead, batch operations
    """
    
    def __init__(self, db_path: str, pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = pool_size
        self.connections = queue.Queue(maxsize=pool_size)
        self._lock = threading.Lock()
        self.active_connections = 0
        
        # Initialize connection pool
        for _ in range(pool_size):
            conn = self._create_connection()
            if conn:
                self.connections.put(conn)
        
        logging.info(f"Database connection pool initialized with {pool_size} connections")
    
    def _create_connection(self) -> Optional[sqlite3.Connection]:
        """Create a new database connection"""
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute('PRAGMA journal_mode=WAL')  # Enable WAL mode for better concurrency
            conn.execute('PRAGMA synchronous=NORMAL')  # Balance performance and safety
            conn.execute('PRAGMA cache_size=10000')  # Increase cache size
            conn.execute('PRAGMA temp_store=MEMORY')  # Use memory for temp tables
            return conn
        except Exception as e:
            logging.error(f"Error creating database connection: {e}")
            return None
    
    def get_connection(self) -> Optional[sqlite3.Connection]:
        """Get connection from pool"""
        try:
            return self.connections.get(timeout=5.0)
        except queue.Empty:
            # Create new connection if pool is exhausted
            with self._lock:
                if self.active_connections < self.pool_size * 2:  # Allow some overflow
                    conn = self._create_connection()
                    if conn:
                        self.active_connections += 1
                        return conn
            return None
    
    def return_connection(self, conn: sqlite3.Connection):
        """Return connection to pool"""
        try:
            if self.connections.qsize() < self.pool_size:
                self.connections.put(conn)
            else:
                # Pool is full, close the connection
                conn.close()
                with self._lock:
                    self.active_connections = max(0, self.active_connections - 1)
        except Exception as e:
            logging.error(f"Error returning connection to pool: {e}")
    
    def close_all(self):
        """Close all connections in pool"""
        while not self.connections.empty():
            try:
                conn = self.connections.get_nowait()
                conn.close()
            except Exception as e:
                logging.error(f"Error closing connection: {e}")
        
        with self._lock:
            self.active_connections = 0