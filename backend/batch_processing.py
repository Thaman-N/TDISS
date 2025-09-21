"""
Batch processing infrastructure for multi-stream violence detection.
This implements the core batch processing system to replace per-stream threading.
"""

import time
import threading
import queue
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
import torch
from enum import Enum

# Import our infrastructure components
from infrastructure_managers import GPUMemoryManager, FileSystemManager


class BatchTrigger(Enum):
    """Triggers for batch processing"""
    SIZE_BASED = "size"      # Process when batch reaches target size
    TIME_BASED = "time"      # Process after timeout
    HYBRID = "hybrid"        # Process on size OR timeout
    IMMEDIATE = "immediate"  # Process immediately (for testing)


@dataclass
class FrameData:
    """Container for frame data with metadata"""
    stream_id: int
    frame_sequence: np.ndarray  # Shape: [16, 336, 336, 3] RGB uint8
    timestamp: float
    sequence_start_time: float
    sequence_end_time: float
    buffer_frames: List[np.ndarray] = field(default_factory=list)  # For incident clips
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class BatchResult:
    """Result from batch processing"""
    stream_id: int
    is_violent: bool
    confidence: float
    inference_time: float
    timestamp: float
    frame_data: FrameData


@dataclass
class BatchJob:
    """Container for a batch processing job"""
    job_id: str
    frame_data_list: List[FrameData]
    created_at: float
    priority: int = 0  # Higher priority = processed first
    callback: Optional[Callable] = None


class BatchInferenceManager:
    """
    Central batch inference manager that coordinates model access and processing.
    Replaces individual stream processing with efficient batch operations.
    
    Key features:
    - Centralized model access with proper locking
    - Adaptive batch sizing based on GPU memory
    - Multiple batch triggers (size, time, hybrid)  
    - Priority-based processing queue
    - Result distribution back to streams
    """
    
    def __init__(
        self,
        model,
        detection_threshold: float = 0.6,
        base_batch_size: int = 8,
        batch_timeout: float = 3.0,
        trigger_mode: BatchTrigger = BatchTrigger.HYBRID,
        gpu_memory_manager: Optional[GPUMemoryManager] = None
    ):
        self.model = model
        self.detection_threshold = detection_threshold
        self.base_batch_size = base_batch_size
        self.batch_timeout = batch_timeout
        self.trigger_mode = trigger_mode
        
        # GPU memory management
        self.gpu_manager = gpu_memory_manager or GPUMemoryManager()
        
        # Batch processing queues
        self.frame_queue = queue.PriorityQueue()  # (priority, timestamp, FrameData)
        self.batch_queue = queue.Queue()  # BatchJob objects
        self.result_callbacks = {}  # stream_id -> callback function
        
        # Processing control
        self.is_running = False
        self.batch_collector_thread = None
        self.batch_processor_thread = None
        self._model_lock = threading.RLock()  # Reentrant lock for model access
        
        # Batch collection state
        self.current_batch = []
        self.last_batch_time = 0.0
        self.batch_stats = {
            'total_batches': 0,
            'total_frames': 0,
            'avg_batch_size': 0.0,
            'avg_processing_time': 0.0,
            'gpu_memory_errors': 0
        }
        
        # Performance monitoring
        self.processing_times = deque(maxlen=100)
        self.batch_sizes = deque(maxlen=100)
        
        logging.info(f"BatchInferenceManager initialized - batch_size: {base_batch_size}, "
                    f"timeout: {batch_timeout}s, trigger: {trigger_mode.value}")
    
    def start(self):
        """Start batch processing threads"""
        if self.is_running:
            logging.warning("BatchInferenceManager already running")
            return
        
        self.is_running = True
        self.last_batch_time = time.time()
        
        # Start batch collection thread
        self.batch_collector_thread = threading.Thread(
            target=self._batch_collection_loop, 
            name="BatchCollector",
            daemon=True
        )
        
        # Start batch processing thread  
        self.batch_processor_thread = threading.Thread(
            target=self._batch_processing_loop,
            name="BatchProcessor", 
            daemon=True
        )
        
        self.batch_collector_thread.start()
        self.batch_processor_thread.start()
        
        logging.info("BatchInferenceManager started")
    
    def stop(self):
        """Stop batch processing threads"""
        if not self.is_running:
            return
        
        logging.info("Stopping BatchInferenceManager...")
        self.is_running = False
        
        # Wait for threads to finish
        if self.batch_collector_thread and self.batch_collector_thread.is_alive():
            self.batch_collector_thread.join(timeout=2.0)
        
        if self.batch_processor_thread and self.batch_processor_thread.is_alive():
            self.batch_processor_thread.join(timeout=2.0)
        
        logging.info("BatchInferenceManager stopped")
    
    def register_stream_callback(self, stream_id: int, callback: Callable[[BatchResult], None]):
        """Register callback for stream results"""
        self.result_callbacks[stream_id] = callback
        logging.debug(f"Registered callback for stream {stream_id}")
    
    def unregister_stream_callback(self, stream_id: int):
        """Unregister stream callback"""
        if stream_id in self.result_callbacks:
            del self.result_callbacks[stream_id]
            logging.debug(f"Unregistered callback for stream {stream_id}")
    
    def submit_frame_data(self, frame_data: FrameData, priority: int = 0):
        """Submit frame data for batch processing"""
        if not self.is_running:
            logging.warning("BatchInferenceManager not running, dropping frame data")
            return
        
        try:
            # Use negative priority so higher priority items are processed first
            self.frame_queue.put((-priority, frame_data.timestamp, frame_data), timeout=1.0)
            logging.debug(f"Submitted frame data for stream {frame_data.stream_id}")
        except queue.Full:
            logging.warning(f"Frame queue full, dropping frame data for stream {frame_data.stream_id}")
    
    def _batch_collection_loop(self):
        """Main batch collection loop - collects frames into batches"""
        logging.info("Batch collection loop started")
        
        while self.is_running:
            try:
                current_time = time.time()
                batch_ready = False
                
                # Try to get frame data with timeout
                try:
                    _, timestamp, frame_data = self.frame_queue.get(timeout=0.5)
                    self.current_batch.append(frame_data)
                    
                    logging.debug(f"Added frame from stream {frame_data.stream_id} to batch "
                                f"(size: {len(self.current_batch)})")
                    
                except queue.Empty:
                    # No new frames, check if we should process current batch based on timeout
                    pass
                
                # Determine if batch is ready based on trigger mode
                batch_ready = self._should_process_batch(current_time)
                
                if batch_ready and self.current_batch:
                    # Adaptive batch sizing based on GPU memory
                    target_batch_size = self.gpu_manager.adaptive_batch_size(len(self.current_batch))
                    
                    if len(self.current_batch) > target_batch_size:
                        # Split into smaller batches if needed
                        batch_to_process = self.current_batch[:target_batch_size]
                        self.current_batch = self.current_batch[target_batch_size:]
                    else:
                        batch_to_process = self.current_batch
                        self.current_batch = []
                    
                    # Create batch job
                    job_id = f"batch_{int(current_time * 1000)}"
                    batch_job = BatchJob(
                        job_id=job_id,
                        frame_data_list=batch_to_process,
                        created_at=current_time
                    )
                    
                    # Submit for processing
                    try:
                        self.batch_queue.put(batch_job, timeout=1.0)
                        self.last_batch_time = current_time
                        
                        logging.info(f"Created batch job {job_id} with {len(batch_to_process)} frames")
                        
                    except queue.Full:
                        logging.error("Batch queue full, dropping batch!")
                        # Put frames back in current batch
                        self.current_batch = batch_to_process + self.current_batch
                
                # Brief sleep to prevent CPU spinning
                time.sleep(0.01)
                
            except Exception as e:
                logging.error(f"Error in batch collection loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
        
        logging.info("Batch collection loop ended")
    
    def _should_process_batch(self, current_time: float) -> bool:
        """Determine if current batch should be processed"""
        if not self.current_batch:
            return False
        
        batch_size = len(self.current_batch)
        time_since_last = current_time - self.last_batch_time
        
        if self.trigger_mode == BatchTrigger.SIZE_BASED:
            return batch_size >= self.base_batch_size
        
        elif self.trigger_mode == BatchTrigger.TIME_BASED:
            return time_since_last >= self.batch_timeout
        
        elif self.trigger_mode == BatchTrigger.HYBRID:
            return (batch_size >= self.base_batch_size or 
                   time_since_last >= self.batch_timeout)
        
        elif self.trigger_mode == BatchTrigger.IMMEDIATE:
            return True
        
        return False
    
    def _batch_processing_loop(self):
        """Main batch processing loop - processes batches through model"""
        logging.info("Batch processing loop started")
        
        while self.is_running:
            try:
                # Get batch job with timeout
                try:
                    batch_job = self.batch_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process the batch
                start_time = time.time()
                results = self._process_batch(batch_job)
                processing_time = time.time() - start_time
                
                # Update statistics
                self.processing_times.append(processing_time)
                self.batch_sizes.append(len(batch_job.frame_data_list))
                self.batch_stats['total_batches'] += 1
                self.batch_stats['total_frames'] += len(batch_job.frame_data_list)
                self.batch_stats['avg_processing_time'] = np.mean(self.processing_times)
                self.batch_stats['avg_batch_size'] = np.mean(self.batch_sizes)
                
                # Distribute results
                self._distribute_results(results)
                
                logging.info(f"Processed batch {batch_job.job_id}: {len(results)} results "
                           f"in {processing_time:.3f}s")
                
                # Cleanup GPU memory periodically
                if self.batch_stats['total_batches'] % 10 == 0:
                    self.gpu_manager.cleanup_gpu_memory()
                
            except Exception as e:
                logging.error(f"Error in batch processing loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
        
        logging.info("Batch processing loop ended")
    
    def _process_batch(self, batch_job: BatchJob) -> List[BatchResult]:
        """Process a batch of frame data through the model"""
        results = []
        
        if not batch_job.frame_data_list:
            return results
        
        try:
            # Prepare batch data for model
            batch_data = self._prepare_batch_data(batch_job.frame_data_list)
            
            if not batch_data:
                logging.error("Failed to prepare batch data")
                return results
            
            # Run inference with model lock
            with self._model_lock:
                # Ensure model is in eval mode
                self.model.eval()
                
                # Check GPU memory before processing
                if not self.gpu_manager.allocate_batch_memory(
                    len(batch_job.frame_data_list), 
                    batch_data['rgb'].shape[1:]  # Shape without batch dimension
                ):
                    logging.warning("Insufficient GPU memory for batch, reducing size")
                    self.batch_stats['gpu_memory_errors'] += 1
                    
                    # Process in smaller sub-batches
                    return self._process_batch_split(batch_job)
                
                # Run model inference
                start_inference = time.time()
                
                with torch.no_grad():
                    model_outputs = self.model(batch_data)
                
                inference_time = time.time() - start_inference
                
                # Process outputs for each item in batch
                results = self._process_batch_outputs(
                    batch_job.frame_data_list, 
                    model_outputs, 
                    inference_time
                )
        
        except Exception as e:
            logging.error(f"Error processing batch {batch_job.job_id}: {e}")
            import traceback
            traceback.print_exc()
            
            # Create error results
            for frame_data in batch_job.frame_data_list:
                results.append(BatchResult(
                    stream_id=frame_data.stream_id,
                    is_violent=False,
                    confidence=0.0,
                    inference_time=0.0,
                    timestamp=frame_data.timestamp,
                    frame_data=frame_data
                ))
        
        return results
    
    def _prepare_batch_data(self, frame_data_list: List[FrameData]) -> Optional[Dict[str, torch.Tensor]]:
        """Prepare batch data for model input"""
        try:
            # Import preprocessing function
            from torch_detection import preprocess_frames
            
            batch_rgb_tensors = []
            batch_flow_tensors = []
            
            # Check if model uses motion enhancement
            use_motion = hasattr(self.model, 'use_motion_enhancement') and self.model.use_motion_enhancement
            
            for frame_data in frame_data_list:
                # Preprocess frame sequence (same as torch_detection.py)
                processed_data = preprocess_frames(frame_data.frame_sequence, compute_flow=use_motion)
                
                batch_rgb_tensors.append(processed_data['rgb'])
                
                if use_motion and 'flow' in processed_data:
                    batch_flow_tensors.append(processed_data['flow'])
            
            # Stack into batch tensors
            batch_data = {
                'rgb': torch.stack(batch_rgb_tensors, dim=0)  # [B, C, T, H, W]
            }
            
            if use_motion and batch_flow_tensors:
                batch_data['flow'] = torch.stack(batch_flow_tensors, dim=0)
            
            # Move to GPU if available
            device = next(self.model.parameters()).device
            for key in batch_data:
                batch_data[key] = batch_data[key].to(device)
            
            return batch_data
            
        except Exception as e:
            logging.error(f"Error preparing batch data: {e}")
            return None
    
    def _process_batch_outputs(self, frame_data_list: List[FrameData], 
                              model_outputs: torch.Tensor, inference_time: float) -> List[BatchResult]:
        """Process model outputs into individual results"""
        results = []
        
        try:
            # Convert outputs to numpy for processing
            outputs_np = model_outputs.cpu().numpy()  # Shape: [B, num_classes]
            
            per_item_inference_time = inference_time / len(frame_data_list)
            
            for i, frame_data in enumerate(frame_data_list):
                # Process individual prediction (same logic as predict_violence)
                if outputs_np.shape[1] >= 2:
                    # Two-class output: Apply softmax
                    exp_logits = np.exp(outputs_np[i] - np.max(outputs_np[i]))
                    probs = exp_logits / np.sum(exp_logits)
                    
                    # Label mapping: 0=NonFight, 1=Fight
                    fight_prob = float(probs[1])
                    is_violent = fight_prob > self.detection_threshold
                    
                else:
                    # Single output (fallback)
                    fight_prob = float(outputs_np[i][0])
                    if abs(fight_prob) > 5:
                        fight_prob = 1.0 / (1.0 + np.exp(-fight_prob))
                    is_violent = fight_prob > self.detection_threshold
                
                # Create result
                result = BatchResult(
                    stream_id=frame_data.stream_id,
                    is_violent=is_violent,
                    confidence=fight_prob,
                    inference_time=per_item_inference_time,
                    timestamp=frame_data.timestamp,
                    frame_data=frame_data
                )
                
                results.append(result)
                
                logging.debug(f"Stream {frame_data.stream_id}: Violence={is_violent}, "
                            f"Confidence={fight_prob:.3f}")
        
        except Exception as e:
            logging.error(f"Error processing batch outputs: {e}")
            # Create default results on error
            for frame_data in frame_data_list:
                results.append(BatchResult(
                    stream_id=frame_data.stream_id,
                    is_violent=False,
                    confidence=0.0,
                    inference_time=0.0,
                    timestamp=frame_data.timestamp,
                    frame_data=frame_data
                ))
        
        return results
    
    def _process_batch_split(self, batch_job: BatchJob) -> List[BatchResult]:
        """Process batch in smaller sub-batches due to memory constraints"""
        logging.info(f"Splitting batch {batch_job.job_id} due to memory constraints")
        
        results = []
        batch_size = max(1, len(batch_job.frame_data_list) // 2)  # Split in half
        
        for i in range(0, len(batch_job.frame_data_list), batch_size):
            sub_batch_data = batch_job.frame_data_list[i:i+batch_size]
            
            sub_job = BatchJob(
                job_id=f"{batch_job.job_id}_split_{i//batch_size}",
                frame_data_list=sub_batch_data,
                created_at=batch_job.created_at
            )
            
            sub_results = self._process_batch(sub_job)
            results.extend(sub_results)
        
        return results
    
    def _distribute_results(self, results: List[BatchResult]):
        """Distribute results to registered stream callbacks"""
        for result in results:
            callback = self.result_callbacks.get(result.stream_id)
            if callback:
                try:
                    callback(result)
                except Exception as e:
                    logging.error(f"Error in callback for stream {result.stream_id}: {e}")
            else:
                logging.debug(f"No callback registered for stream {result.stream_id}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get batch processing statistics"""
        return {
            **self.batch_stats,
            'is_running': self.is_running,
            'current_batch_size': len(self.current_batch),
            'queue_sizes': {
                'frame_queue': self.frame_queue.qsize(),
                'batch_queue': self.batch_queue.qsize()
            },
            'gpu_info': self.gpu_manager.get_memory_info(),
            'registered_streams': len(self.result_callbacks)
        }


class SharedResourcePool:
    """
    Shared resource pool for batch processing system.
    Manages shared resources like database connections, file operations, etc.
    """
    
    def __init__(self):
        self.file_manager = FileSystemManager()
        self.notification_queue = queue.Queue()
        self._lock = threading.Lock()
        
        # Notification processing
        self.notification_processor_thread = None
        self.is_running = False
        
        logging.info("SharedResourcePool initialized")
    
    def start(self):
        """Start shared resource processing"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start notification processor
        self.notification_processor_thread = threading.Thread(
            target=self._notification_processing_loop,
            name="NotificationProcessor",
            daemon=True
        )
        self.notification_processor_thread.start()
        
        logging.info("SharedResourcePool started")
    
    def stop(self):
        """Stop shared resource processing"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.notification_processor_thread and self.notification_processor_thread.is_alive():
            self.notification_processor_thread.join(timeout=2.0)
        
        logging.info("SharedResourcePool stopped")
    
    def save_thumbnail(self, stream_id: int, frame: np.ndarray, timestamp: float = None) -> Optional[str]:
        """Save thumbnail using shared file manager"""
        return self.file_manager.save_thumbnail(stream_id, frame, timestamp)
    
    def save_clip(self, stream_id: int, frames: List[np.ndarray], timestamp: float = None) -> Optional[str]:
        """Save clip using shared file manager"""
        return self.file_manager.save_clip(stream_id, frames, timestamp)
    
    def queue_notification(self, notification_data: Dict[str, Any]):
        """Queue notification for processing"""
        try:
            self.notification_queue.put(notification_data, timeout=1.0)
        except queue.Full:
            logging.warning("Notification queue full, dropping notification")
    
    def _notification_processing_loop(self):
        """Process queued notifications"""
        logging.info("Notification processing loop started")
        
        while self.is_running:
            try:
                notification = self.notification_queue.get(timeout=1.0)
                self._process_notification(notification)
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error processing notification: {e}")
        
        logging.info("Notification processing loop ended")
    
    def _process_notification(self, notification: Dict[str, Any]):
        """Process individual notification"""
        # Implementation depends on notification type
        # For now, just log
        logging.info(f"Processing notification: {notification.get('type', 'unknown')}")
    
    def cleanup_resources(self):
        """Cleanup shared resources"""
        self.file_manager.cleanup_old_files(force=True)
    
    def get_resource_info(self) -> Dict[str, Any]:
        """Get resource usage information"""
        return {
            'storage_info': self.file_manager.get_storage_info(),
            'notification_queue_size': self.notification_queue.qsize(),
            'is_running': self.is_running
        }