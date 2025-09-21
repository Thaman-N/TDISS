"""
Integration System for TDISS Violence Detection
============================================

This module provides the complete integration layer that orchestrates all the new
batch processing components with the existing FastAPI system.

Key Features:
- Seamless integration with existing API endpoints
- Backward compatibility with legacy stream management
- Complete error handling and recovery mechanisms
- Performance monitoring and optimization
"""

import asyncio
import logging
import traceback
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

from infrastructure_managers import (
    RTSPConnectionManager, GPUMemoryManager, 
    FileSystemManager, DatabaseConnectionPool
)
from batch_processing import (
    BatchInferenceManager, SharedResourcePool,
    FrameData, BatchResult
)
from stream_collection import StreamFrameCollector, StreamCollectionManager
from smart_notifications import SmartNotificationManager
from enhanced_database import EnhancedDatabaseManager

logger = logging.getLogger(__name__)

@dataclass
class StreamConfiguration:
    """Configuration for a single RTSP stream."""
    stream_id: str
    rtsp_url: str
    stream_name: str
    detection_threshold: float = 0.5
    priority_level: int = 1  # 1=high, 2=medium, 3=low
    
@dataclass
class SystemStats:
    """System performance statistics."""
    active_streams: int
    total_frames_processed: int
    average_batch_size: float
    gpu_memory_usage: float
    detection_rate: float
    notification_rate: float
    database_queue_size: int
    error_count: int
    uptime: timedelta

class IntegratedViolenceDetectionSystem:
    """
    Main integration system that orchestrates all components.
    
    This class provides the single interface for the FastAPI backend
    to interact with the new batch processing architecture.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_initialized = False
        self.start_time = datetime.now()
        
        # Core managers
        self.connection_manager: Optional[RTSPConnectionManager] = None
        self.gpu_manager: Optional[GPUMemoryManager] = None
        self.file_manager: Optional[FileSystemManager] = None
        self.db_pool: Optional[DatabaseConnectionPool] = None
        
        # Processing components
        self.batch_manager: Optional[BatchInferenceManager] = None
        self.resource_pool: Optional[SharedResourcePool] = None
        self.stream_manager: Optional[StreamCollectionManager] = None
        
        # Service components
        self.notification_manager: Optional[SmartNotificationManager] = None
        self.enhanced_db: Optional[EnhancedDatabaseManager] = None
        
        # State tracking
        self.active_streams: Dict[str, StreamFrameCollector] = {}
        self.stream_configs: Dict[str, StreamConfiguration] = {}
        self.performance_stats = SystemStats(
            active_streams=0,
            total_frames_processed=0,
            average_batch_size=0.0,
            gpu_memory_usage=0.0,
            detection_rate=0.0,
            notification_rate=0.0,
            database_queue_size=0,
            error_count=0,
            uptime=timedelta()
        )
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._stats_task: Optional[asyncio.Task] = None
        
    async def initialize(self) -> bool:
        """
        Initialize all system components.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing Integrated Violence Detection System...")
            
            # Initialize infrastructure managers
            await self._init_infrastructure()
            
            # Initialize processing components
            await self._init_processing()
            
            # Initialize service components
            await self._init_services()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.is_initialized = True
            logger.info("System initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            logger.error(traceback.format_exc())
            await self.cleanup()
            return False
    
    async def _init_infrastructure(self):
        """Initialize infrastructure managers."""
        # Connection manager
        self.connection_manager = RTSPConnectionManager(
            max_connections=self.config.get('max_rtsp_connections', 12),
            bandwidth_limit_mbps=self.config.get('bandwidth_limit_mbps', 50.0)
        )
        
        # GPU manager
        max_gpu_memory_gb = self.config.get('max_gpu_memory_gb', 6.0)
        self.gpu_manager = GPUMemoryManager(
            max_gpu_memory_gb=max_gpu_memory_gb
        )
        
        # File system manager
        self.file_manager = FileSystemManager(
            base_results_dir=self.config.get('storage_path', './data'),
            max_storage_gb=self.config.get('max_storage_gb', 50)
        )
        
        # Database connection pool
        self.db_pool = DatabaseConnectionPool(
            db_path=self.config.get('db_path', './violence_detection.db'),
            pool_size=self.config.get('max_db_connections', 10)
        )
        
        logger.info("Infrastructure managers initialized")
    
    async def _init_processing(self):
        """Initialize processing components."""
        # Shared resource pool
        self.resource_pool = SharedResourcePool()
        
        # Load model for batch manager
        from torch_detection import load_violence_detection_model
        model_path = self.config.get('model_path', './models/rwf9425.pth')
        model, _ = load_violence_detection_model(model_path)
        
        # Batch inference manager
        self.batch_manager = BatchInferenceManager(
            model=model,
            detection_threshold=self.config.get('detection_threshold', 0.6),
            base_batch_size=self.config.get('default_batch_size', 4),
            batch_timeout=self.config.get('batch_timeout', 3.0),
            gpu_memory_manager=self.gpu_manager
        )
        
        # Stream collection manager
        self.stream_manager = StreamCollectionManager(
            batch_manager=self.batch_manager,
            connection_manager=self.connection_manager
        )
        
        logger.info("Processing components initialized")
    
    async def _init_services(self):
        """Initialize service components."""
        # Enhanced database manager
        self.enhanced_db = EnhancedDatabaseManager(
            db_path=self.config.get('db_path', './violence_detection.db'),
            pool_size=self.config.get('max_db_connections', 8)
        )
        
        # Smart notification manager
        self.notification_manager = SmartNotificationManager()
        
        logger.info("Service components initialized")
    
    async def _start_background_tasks(self):
        """Start background monitoring and maintenance tasks."""
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._stats_task = asyncio.create_task(self._stats_collection_loop())
        
        logger.info("Background tasks started")
    
    async def start_stream(self, stream_config: StreamConfiguration) -> bool:
        """
        Start processing a new RTSP stream.
        
        Args:
            stream_config: Configuration for the stream
            
        Returns:
            bool: True if stream started successfully
        """
        if not self.is_initialized:
            logger.error("System not initialized")
            return False
        
        try:
            stream_id = stream_config.stream_id
            
            if stream_id in self.active_streams:
                logger.warning(f"Stream {stream_id} already active")
                return True
            
            # Create stream collector
            collector = await self.stream_manager.create_stream_collector(
                stream_id=stream_id,
                rtsp_url=stream_config.rtsp_url,
                priority=stream_config.priority_level,
                result_callback=self._handle_detection_result
            )
            
            if collector:
                self.active_streams[stream_id] = collector
                self.stream_configs[stream_id] = stream_config
                
                # Update performance stats
                self.performance_stats.active_streams = len(self.active_streams)
                
                logger.info(f"Stream {stream_id} started successfully")
                return True
            else:
                logger.error(f"Failed to create collector for stream {stream_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error starting stream {stream_config.stream_id}: {e}")
            return False
    
    async def stop_stream(self, stream_id: str) -> bool:
        """
        Stop processing an RTSP stream.
        
        Args:
            stream_id: ID of the stream to stop
            
        Returns:
            bool: True if stream stopped successfully
        """
        if not self.is_initialized:
            logger.error("System not initialized")
            return False
        
        try:
            if stream_id not in self.active_streams:
                logger.warning(f"Stream {stream_id} not active")
                return True
            
            # Stop the collector
            collector = self.active_streams[stream_id]
            await collector.stop()
            
            # Clean up
            del self.active_streams[stream_id]
            if stream_id in self.stream_configs:
                del self.stream_configs[stream_id]
            
            # Update performance stats
            self.performance_stats.active_streams = len(self.active_streams)
            
            logger.info(f"Stream {stream_id} stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping stream {stream_id}: {e}")
            return False
    
    async def _handle_detection_result(self, result: BatchResult):
        """
        Handle detection results from the batch processing system.
        
        Args:
            result: Batch processing result containing detections
        """
        try:
            for detection in result.detections:
                stream_id = detection.get('stream_id')
                confidence = detection.get('confidence', 0.0)
                
                if stream_id not in self.stream_configs:
                    continue
                
                config = self.stream_configs[stream_id]
                
                # Check if detection exceeds threshold
                if confidence >= config.detection_threshold:
                    # Save to database
                    await self.enhanced_db.save_detection(
                        stream_id=stream_id,
                        confidence=confidence,
                        timestamp=detection.get('timestamp'),
                        frame_data=detection.get('frame_data'),
                        metadata=detection.get('metadata', {})
                    )
                    
                    # Send notification
                    await self.notification_manager.send_detection_notification(
                        stream_id=stream_id,
                        stream_name=config.stream_name,
                        confidence=confidence,
                        timestamp=detection.get('timestamp'),
                        thumbnail_data=detection.get('thumbnail')
                    )
                    
                    # Update stats
                    self.performance_stats.total_frames_processed += 1
                    self.performance_stats.detection_rate = (
                        self.performance_stats.detection_rate * 0.9 + confidence * 0.1
                    )
            
        except Exception as e:
            logger.error(f"Error handling detection result: {e}")
            self.performance_stats.error_count += 1
    
    async def get_stream_status(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """Get status information for a specific stream."""
        if stream_id not in self.active_streams:
            return None
        
        collector = self.active_streams[stream_id]
        return {
            'stream_id': stream_id,
            'status': 'active',
            'frames_collected': collector.frames_collected,
            'frames_processed': collector.frames_processed,
            'last_frame_time': collector.last_frame_time,
            'connection_status': await self.connection_manager.get_connection_status(stream_id),
            'queue_size': collector.frame_queue.qsize() if hasattr(collector, 'frame_queue') else 0
        }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        uptime = datetime.now() - self.start_time
        
        gpu_memory_info = self.gpu_manager.get_memory_info() if self.gpu_manager else {}
        gpu_memory_usage = gpu_memory_info.get('memory_used_percent', 0.0) / 100.0 if gpu_memory_info else 0.0
        
        # Get queue sizes safely
        db_queue_size = 0
        if self.enhanced_db and hasattr(self.enhanced_db, 'pending_events'):
            db_queue_size = len(getattr(self.enhanced_db, 'pending_events', []))
        
        notification_queue_size = 0
        if self.notification_manager and hasattr(self.notification_manager, 'notification_queue'):
            notification_queue_size = self.notification_manager.notification_queue.qsize()
        
        # Get batch stats safely
        batch_stats = {}
        if self.batch_manager:
            if hasattr(self.batch_manager, 'frame_queue'):
                batch_stats['queue_size'] = self.batch_manager.frame_queue.qsize()
            if hasattr(self.batch_manager, 'current_batch_size'):
                batch_stats['current_batch_size'] = self.batch_manager.current_batch_size
        
        return {
            'system_initialized': self.is_initialized,
            'uptime_seconds': int(uptime.total_seconds()),
            'active_streams': len(self.active_streams),
            'total_frames_processed': self.performance_stats.total_frames_processed,
            'gpu_memory_usage': gpu_memory_usage,
            'database_queue_size': db_queue_size,
            'notification_queue_size': notification_queue_size,
            'error_count': self.performance_stats.error_count,
            'batch_processing_stats': batch_stats
        }
    
    async def _monitoring_loop(self):
        """Background monitoring loop for system health."""
        while self.is_initialized:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
                # Check GPU memory usage
                if self.gpu_manager:
                    memory_info = self.gpu_manager.get_memory_info()
                    memory_usage = memory_info.get('memory_used_percent', 0.0) / 100.0 if memory_info else 0.0
                    if memory_usage > 0.9:
                        logger.warning(f"High GPU memory usage: {memory_usage:.2%}")
                
                # Check database queue
                if self.enhanced_db and hasattr(self.enhanced_db, 'pending_events'):
                    queue_size = len(getattr(self.enhanced_db, 'pending_events', []))
                    if queue_size > 100:
                        logger.warning(f"Large database queue: {queue_size} items")
                
                # Check failed streams
                failed_streams = []
                for stream_id, collector in self.active_streams.items():
                    if hasattr(collector, 'is_failed') and collector.is_failed:
                        failed_streams.append(stream_id)
                
                # Restart failed streams
                for stream_id in failed_streams:
                    logger.info(f"Restarting failed stream: {stream_id}")
                    config = self.stream_configs.get(stream_id)
                    if config:
                        await self.stop_stream(stream_id)
                        await asyncio.sleep(5)  # Brief delay
                        await self.start_stream(config)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                self.performance_stats.error_count += 1
    
    async def _stats_collection_loop(self):
        """Background statistics collection loop."""
        while self.is_initialized:
            try:
                await asyncio.sleep(60)  # Collect stats every minute
                
                # Update system stats
                if self.batch_manager and hasattr(self.batch_manager, 'current_batch_size'):
                    self.performance_stats.average_batch_size = getattr(self.batch_manager, 'current_batch_size', 0.0)
                
                if self.gpu_manager:
                    memory_info = self.gpu_manager.get_memory_info()
                    self.performance_stats.gpu_memory_usage = memory_info.get('memory_used_percent', 0.0) / 100.0 if memory_info else 0.0
                
                if self.enhanced_db and hasattr(self.enhanced_db, 'pending_events'):
                    self.performance_stats.database_queue_size = len(getattr(self.enhanced_db, 'pending_events', []))
                
                # Log periodic stats
                logger.info(f"System Stats - Streams: {self.performance_stats.active_streams}, "
                           f"Frames: {self.performance_stats.total_frames_processed}, "
                           f"GPU: {self.performance_stats.gpu_memory_usage:.1%}, "
                           f"Errors: {self.performance_stats.error_count}")
                
            except Exception as e:
                logger.error(f"Error in stats collection: {e}")
    
    async def cleanup(self):
        """Clean up all system resources."""
        logger.info("Starting system cleanup...")
        
        try:
            # Stop background tasks
            if self._monitoring_task:
                self._monitoring_task.cancel()
            if self._stats_task:
                self._stats_task.cancel()
            
            # Stop all active streams
            for stream_id in list(self.active_streams.keys()):
                await self.stop_stream(stream_id)
            
            # Clean up components in reverse order
            if self.notification_manager:
                self.notification_manager.stop()
            
            if self.enhanced_db:
                if hasattr(self.enhanced_db, 'stop'):
                    await self.enhanced_db.stop()
                elif hasattr(self.enhanced_db, 'cleanup'):
                    self.enhanced_db.cleanup()
            
            if self.stream_manager:
                self.stream_manager.stop_all_collections()
            
            if self.batch_manager:
                self.batch_manager.stop()
                if hasattr(self.batch_manager, 'cleanup_resources'):
                    self.batch_manager.cleanup_resources()
            
            if self.connection_manager:
                await self.connection_manager.cleanup_all()
            
            if self.gpu_manager:
                if hasattr(self.gpu_manager, 'cleanup'):
                    self.gpu_manager.cleanup()
            
            if self.file_manager:
                if hasattr(self.file_manager, 'cleanup'):
                    await self.file_manager.cleanup_old_files()
            
            if self.db_pool:
                if hasattr(self.db_pool, 'close_all'):
                    self.db_pool.close_all()
            
            self.is_initialized = False
            logger.info("System cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Global system instance
violence_detection_system: Optional[IntegratedViolenceDetectionSystem] = None

async def initialize_system(config: Dict[str, Any]) -> bool:
    """Initialize the global violence detection system."""
    global violence_detection_system
    
    if violence_detection_system is None:
        violence_detection_system = IntegratedViolenceDetectionSystem(config)
    
    return await violence_detection_system.initialize()

async def get_system() -> Optional[IntegratedViolenceDetectionSystem]:
    """Get the global violence detection system instance."""
    return violence_detection_system

async def cleanup_system():
    """Clean up the global violence detection system."""
    global violence_detection_system
    
    if violence_detection_system:
        await violence_detection_system.cleanup()
        violence_detection_system = None