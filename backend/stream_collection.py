"""
Stream frame collection components for the batch processing system.
Replaces the old RTSPStreamProcessor with lightweight frame collectors.
"""

import time
import threading
import queue
import logging
from typing import Dict, List, Optional, Callable, Deque
from dataclasses import dataclass
from collections import deque
import cv2
import numpy as np

from batch_processing import FrameData, BatchResult
from infrastructure_managers import RTSPConnection, RTSPConnectionManager


class StreamFrameCollector:
    """
    Lightweight frame collector that replaces RTSPStreamProcessor.
    
    Key changes from old system:
    - Single thread (capture only, no processing)
    - Smaller buffers (no 3 large independent buffers)  
    - Frame queuing to batch system instead of direct model access
    - Simplified lifecycle management
    """
    
    def __init__(
        self,
        stream_id: int,
        rtsp_url: str, 
        stream_name: str,
        connection_manager: RTSPConnectionManager,
        batch_callback: Callable[[FrameData], None]
    ):
        self.stream_id = stream_id
        self.rtsp_url = rtsp_url
        self.stream_name = stream_name
        self.connection_manager = connection_manager
        self.batch_callback = batch_callback  # Callback to submit to batch system
        
        # Frame collection state
        self.is_running = False
        self.capture_thread = None
        
        # Lightweight buffers (much smaller than old system)
        self.frame_sequence_buffer = deque(maxlen=16)  # Just enough for one sequence
        self.display_frame_buffer = deque(maxlen=30)   # For thumbnails/clips (reduced from 50)
        self.last_display_frame = None
        
        # Timing control
        self.target_fps = 8  # Target collection FPS (reduced from processing every frame)
        self.frame_interval = 1.0 / self.target_fps  # ~125ms between frames
        self.last_frame_time = 0.0
        self.last_sequence_submit_time = 0.0
        self.sequence_submit_interval = 3.0  # Submit sequence every 3 seconds
        
        # Model input requirements (from torch_detection.py)  
        self.model_input_size = (336, 336)  # INPUT_SIZE = 336
        self.sequence_length = 16  # NUM_FRAMES = 16
        
        # Thread synchronization (minimal locking)
        self._lock = threading.Lock()
        
        # Status tracking
        self.stats = {
            'frames_collected': 0,
            'sequences_submitted': 0,
            'connection_failures': 0,
            'last_activity': 0.0
        }
        
        logging.info(f"StreamFrameCollector initialized for stream {stream_id}: {stream_name}")
    
    def start_collection(self) -> bool:
        """Start frame collection"""
        if self.is_running:
            logging.warning(f"Stream {self.stream_id} collection already running")
            return False
        
        # Create RTSP connection through manager
        if not self.connection_manager.create_connection(self.stream_id, self.rtsp_url):
            logging.error(f"Failed to create connection for stream {self.stream_id}")
            return False
        
        # Start collection thread
        self.is_running = True
        self.capture_thread = threading.Thread(
            target=self._collection_loop,
            name=f"StreamCollector-{self.stream_id}",
            daemon=True
        )
        
        self.capture_thread.start()
        
        logging.info(f"Started frame collection for stream {self.stream_id}")
        return True
    
    def stop_collection(self):
        """Stop frame collection"""
        if not self.is_running:
            return
        
        logging.info(f"Stopping frame collection for stream {self.stream_id}")
        self.is_running = False
        
        # Wait for thread to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        # Remove connection through manager
        self.connection_manager.remove_connection(self.stream_id)
        
        # Clear buffers
        with self._lock:
            self.frame_sequence_buffer.clear()
            self.display_frame_buffer.clear()
            self.last_display_frame = None
        
        logging.info(f"Stopped frame collection for stream {self.stream_id}")
    
    def _collection_loop(self):
        """Main frame collection loop - single thread, optimized for efficiency"""
        logging.info(f"Frame collection loop started for stream {self.stream_id}")
        
        consecutive_failures = 0
        max_failures = 10
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Rate limiting - don't capture every frame, just what we need
                if current_time - self.last_frame_time < self.frame_interval:
                    time.sleep(0.01)  # Brief sleep
                    continue
                
                # Get connection from manager
                connection = self.connection_manager.get_connection(self.stream_id)
                if not connection or not connection.cap:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        logging.error(f"Stream {self.stream_id}: Connection lost, stopping collection")
                        break
                    time.sleep(0.5)
                    continue
                
                # Read frame
                ret, frame = connection.cap.read()
                if not ret or frame is None:
                    consecutive_failures += 1
                    self.stats['connection_failures'] += 1
                    
                    if consecutive_failures >= max_failures:
                        logging.error(f"Stream {self.stream_id}: Too many read failures, stopping")
                        break
                    
                    time.sleep(0.1)
                    continue
                
                # Reset failure counter on successful read
                consecutive_failures = 0
                self.last_frame_time = current_time
                self.stats['frames_collected'] += 1
                self.stats['last_activity'] = current_time
                
                # Process frame efficiently
                self._process_collected_frame(frame, current_time)
                
                # Check if we should submit a sequence for batch processing
                if (current_time - self.last_sequence_submit_time >= self.sequence_submit_interval and
                    len(self.frame_sequence_buffer) >= self.sequence_length):
                    
                    self._submit_frame_sequence(current_time)
                    self.last_sequence_submit_time = current_time
                
            except Exception as e:
                logging.error(f"Error in collection loop for stream {self.stream_id}: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    break
                time.sleep(0.5)
        
        logging.info(f"Frame collection loop ended for stream {self.stream_id}")
    
    def _process_collected_frame(self, frame: np.ndarray, timestamp: float):
        """Process collected frame efficiently"""
        try:
            # Store display frame for thumbnails/clips (keep original BGR)
            with self._lock:
                self.last_display_frame = frame.copy()
                self.display_frame_buffer.append(frame.copy())
            
            # Preprocess frame for model (convert to RGB and resize)
            rgb_frame = self._preprocess_frame_for_sequence(frame)
            if rgb_frame is not None:
                self.frame_sequence_buffer.append(rgb_frame)
            
        except Exception as e:
            logging.error(f"Error processing frame for stream {self.stream_id}: {e}")
    
    def _preprocess_frame_for_sequence(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Preprocess frame for model sequence (lightweight version)
        Output: RGB uint8 frame in [336, 336, 3] format - matching extract_frames()
        """
        try:
            # Resize to model input size
            frame_resized = cv2.resize(frame, self.model_input_size, interpolation=cv2.INTER_LINEAR)
            
            # Convert BGR to RGB (like extract_frames does)  
            if len(frame_resized.shape) == 3 and frame_resized.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame_resized.copy()
            
            # Ensure uint8 type
            if frame_rgb.dtype != np.uint8:
                frame_rgb = frame_rgb.astype(np.uint8)
            
            return frame_rgb
            
        except Exception as e:
            logging.error(f"Error preprocessing frame for stream {self.stream_id}: {e}")
            return None
    
    def _submit_frame_sequence(self, timestamp: float):
        """Submit frame sequence to batch processing system"""
        try:
            # Get last 16 frames from buffer
            frames_list = list(self.frame_sequence_buffer)[-self.sequence_length:]
            
            if len(frames_list) < self.sequence_length:
                # Pad with last frame if needed
                while len(frames_list) < self.sequence_length:
                    frames_list.append(frames_list[-1].copy() if frames_list else 
                                     np.zeros((self.model_input_size[0], self.model_input_size[1], 3), dtype=np.uint8))
            
            # Convert to numpy array [T, H, W, C] - matching extract_frames output
            frame_sequence = np.array(frames_list[:self.sequence_length])
            
            # Get buffer frames for incident clips
            buffer_frames = []
            with self._lock:
                if self.display_frame_buffer:
                    # Get recent frames for incident context (last 10 frames, ~3 seconds)
                    buffer_frames = list(self.display_frame_buffer)[-10:]
            
            # Create frame data for batch processing
            frame_data = FrameData(
                stream_id=self.stream_id,
                frame_sequence=frame_sequence,
                timestamp=timestamp,
                sequence_start_time=timestamp - (self.sequence_length * self.frame_interval),
                sequence_end_time=timestamp,
                buffer_frames=buffer_frames,
                metadata={
                    'stream_name': self.stream_name,
                    'rtsp_url': self.rtsp_url,
                    'collection_stats': self.stats.copy()
                }
            )
            
            # Submit to batch processing via callback
            self.batch_callback(frame_data)
            
            self.stats['sequences_submitted'] += 1
            
            logging.debug(f"Stream {self.stream_id}: Submitted sequence {self.stats['sequences_submitted']} "
                        f"with {len(frame_sequence)} frames")
            
        except Exception as e:
            logging.error(f"Error submitting frame sequence for stream {self.stream_id}: {e}")
    
    def get_thumbnail_frame(self) -> Optional[np.ndarray]:
        """Get current thumbnail frame"""
        with self._lock:
            return self.last_display_frame.copy() if self.last_display_frame is not None else None
    
    def get_recent_frames_for_clip(self, num_frames: int = 20) -> List[np.ndarray]:
        """Get recent frames for creating incident clips"""
        with self._lock:
            if self.display_frame_buffer:
                return list(self.display_frame_buffer)[-num_frames:]
            return []
    
    def get_statistics(self) -> Dict[str, any]:
        """Get collection statistics"""
        return {
            **self.stats,
            'is_running': self.is_running,
            'buffer_sizes': {
                'sequence_buffer': len(self.frame_sequence_buffer),
                'display_buffer': len(self.display_frame_buffer)
            },
            'timing': {
                'target_fps': self.target_fps,
                'sequence_interval': self.sequence_submit_interval,
                'last_frame_time': self.last_frame_time,
                'last_sequence_submit': self.last_sequence_submit_time
            }
        }


class StreamCollectionManager:
    """
    Manager for multiple stream collectors.
    Coordinates frame collection across streams and interfaces with batch processing.
    """
    
    def __init__(self, batch_manager, connection_manager: Optional[RTSPConnectionManager] = None):
        self.batch_manager = batch_manager
        self.connection_manager = connection_manager or RTSPConnectionManager()
        
        # Active collectors
        self.collectors: Dict[int, StreamFrameCollector] = {}
        self._lock = threading.Lock()
        
        # Result handling
        self.result_callbacks: Dict[int, Callable[[BatchResult], None]] = {}
        
        logging.info("StreamCollectionManager initialized")
    
    def start_stream_collection(
        self, 
        stream_id: int, 
        rtsp_url: str, 
        stream_name: str,
        result_callback: Optional[Callable[[BatchResult], None]] = None
    ) -> bool:
        """Start collection for a stream"""
        
        with self._lock:
            if stream_id in self.collectors:
                logging.warning(f"Stream {stream_id} already has active collector")
                return False
            
            # Create collector
            collector = StreamFrameCollector(
                stream_id=stream_id,
                rtsp_url=rtsp_url,
                stream_name=stream_name,
                connection_manager=self.connection_manager,
                batch_callback=self._on_frame_data_ready
            )
            
            # Start collection
            if not collector.start_collection():
                return False
            
            # Store collector and register callbacks
            self.collectors[stream_id] = collector
            
            if result_callback:
                self.result_callbacks[stream_id] = result_callback
            
            # Register with batch manager for results
            self.batch_manager.register_stream_callback(stream_id, self._on_batch_result)
            
            logging.info(f"Started collection for stream {stream_id}: {stream_name}")
            return True
    
    def stop_stream_collection(self, stream_id: int):
        """Stop collection for a stream"""
        with self._lock:
            if stream_id not in self.collectors:
                logging.warning(f"No active collector for stream {stream_id}")
                return
            
            # Stop collector
            collector = self.collectors[stream_id]
            collector.stop_collection()
            
            # Cleanup
            del self.collectors[stream_id]
            
            if stream_id in self.result_callbacks:
                del self.result_callbacks[stream_id]
            
            # Unregister from batch manager
            self.batch_manager.unregister_stream_callback(stream_id)
            
            logging.info(f"Stopped collection for stream {stream_id}")
    
    def stop_all_collections(self):
        """Stop all active collections"""
        stream_ids = list(self.collectors.keys())
        for stream_id in stream_ids:
            self.stop_stream_collection(stream_id)
    
    def _on_frame_data_ready(self, frame_data: FrameData):
        """Handle frame data ready for batch processing"""
        # Submit to batch manager with normal priority
        self.batch_manager.submit_frame_data(frame_data, priority=0)
    
    def _on_batch_result(self, result: BatchResult):
        """Handle batch processing result"""
        # Forward to stream-specific callback if registered
        callback = self.result_callbacks.get(result.stream_id)
        if callback:
            try:
                callback(result)
            except Exception as e:
                logging.error(f"Error in result callback for stream {result.stream_id}: {e}")
        else:
            # Default handling if no specific callback
            logging.debug(f"Stream {result.stream_id}: Violence={result.is_violent}, "
                        f"Confidence={result.confidence:.3f}")
    
    def get_stream_thumbnail(self, stream_id: int) -> Optional[np.ndarray]:
        """Get thumbnail for a stream"""
        collector = self.collectors.get(stream_id)
        if collector:
            return collector.get_thumbnail_frame()
        return None
    
    def get_stream_statistics(self, stream_id: int) -> Optional[Dict]:
        """Get statistics for a stream"""
        collector = self.collectors.get(stream_id)
        if collector:
            return collector.get_statistics()
        return None
    
    def get_all_statistics(self) -> Dict[str, any]:
        """Get statistics for all streams"""
        with self._lock:
            stats = {
                'active_streams': len(self.collectors),
                'connection_manager': self.connection_manager.get_bandwidth_usage(),
                'streams': {}
            }
            
            for stream_id, collector in self.collectors.items():
                stats['streams'][stream_id] = collector.get_statistics()
            
            return stats