"""
Smart notification system with rate limiting, deduplication, and aggregation.
Addresses notification spam prevention and WebSocket optimization.
"""

import time
import threading
import queue
import asyncio
import logging
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum


class NotificationType(Enum):
    """Types of notifications"""
    VIOLENCE_ALERT = "violence_alert"
    INCIDENT_FINALIZED = "incident_finalized" 
    SYSTEM_STATUS = "system_status"
    STREAM_STATUS = "stream_status"
    PERFORMANCE_WARNING = "performance_warning"


class NotificationPriority(Enum):
    """Notification priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class NotificationEvent:
    """Individual notification event"""
    id: str
    type: NotificationType
    priority: NotificationPriority
    stream_id: Optional[int] = None
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    retry_count: int = 0
    max_retries: int = 3


class NotificationRateLimiter:
    """Rate limiter for notifications to prevent spam"""
    
    def __init__(self, max_notifications: int = 10, time_window: float = 60.0):
        self.max_notifications = max_notifications
        self.time_window = time_window
        self.notification_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._lock = threading.Lock()
    
    def can_send(self, notification_key: str) -> bool:
        """Check if notification can be sent based on rate limits"""
        with self._lock:
            current_time = time.time()
            history = self.notification_history[notification_key]
            
            # Remove old entries outside time window
            while history and current_time - history[0] > self.time_window:
                history.popleft()
            
            # Check if under rate limit
            if len(history) < self.max_notifications:
                history.append(current_time)
                return True
            
            return False
    
    def get_rate_status(self, notification_key: str) -> Dict[str, Any]:
        """Get current rate limiting status"""
        with self._lock:
            current_time = time.time()
            history = self.notification_history[notification_key]
            
            # Clean old entries
            while history and current_time - history[0] > self.time_window:
                history.popleft()
            
            return {
                'current_count': len(history),
                'max_count': self.max_notifications,
                'time_window': self.time_window,
                'can_send': len(history) < self.max_notifications,
                'next_available': history[0] + self.time_window if history else current_time
            }


class NotificationAggregator:
    """Aggregates related notifications to reduce spam"""
    
    def __init__(self, aggregation_window: float = 30.0):
        self.aggregation_window = aggregation_window
        self.pending_aggregations: Dict[str, List[NotificationEvent]] = {}
        self.aggregation_timers: Dict[str, threading.Timer] = {}
        self._lock = threading.Lock()
        self.aggregate_callback: Optional[callable] = None
    
    def set_aggregate_callback(self, callback: callable):
        """Set callback for aggregated notifications"""
        self.aggregate_callback = callback
    
    def add_notification(self, notification: NotificationEvent) -> bool:
        """Add notification for potential aggregation. Returns True if aggregated, False if should send immediately"""
        
        # Don't aggregate critical notifications
        if notification.priority == NotificationPriority.CRITICAL:
            return False
        
        # Don't aggregate system status notifications
        if notification.type == NotificationType.SYSTEM_STATUS:
            return False
        
        # Create aggregation key based on type and stream
        agg_key = f"{notification.type.value}_{notification.stream_id or 'global'}"
        
        with self._lock:
            # Add to pending aggregations
            if agg_key not in self.pending_aggregations:
                self.pending_aggregations[agg_key] = []
            
            self.pending_aggregations[agg_key].append(notification)
            
            # Cancel existing timer and start new one
            if agg_key in self.aggregation_timers:
                self.aggregation_timers[agg_key].cancel()
            
            timer = threading.Timer(
                self.aggregation_window, 
                lambda: self._flush_aggregation(agg_key)
            )
            self.aggregation_timers[agg_key] = timer
            timer.start()
            
            # If we have multiple notifications, aggregate immediately for certain types
            if (len(self.pending_aggregations[agg_key]) >= 3 and 
                notification.type == NotificationType.VIOLENCE_ALERT):
                self._flush_aggregation(agg_key)
            
            return True  # Notification was aggregated
    
    def _flush_aggregation(self, agg_key: str):
        """Flush aggregated notifications"""
        with self._lock:
            if agg_key not in self.pending_aggregations:
                return
            
            notifications = self.pending_aggregations[agg_key]
            if not notifications:
                return
            
            # Clean up
            del self.pending_aggregations[agg_key]
            if agg_key in self.aggregation_timers:
                del self.aggregation_timers[agg_key]
        
        # Create aggregated notification
        if len(notifications) == 1:
            # Single notification, send as-is
            aggregated = notifications[0]
        else:
            # Multiple notifications, create summary
            aggregated = self._create_aggregated_notification(notifications)
        
        # Send via callback
        if self.aggregate_callback:
            try:
                self.aggregate_callback(aggregated)
            except Exception as e:
                logging.error(f"Error in aggregation callback: {e}")
    
    def _create_aggregated_notification(self, notifications: List[NotificationEvent]) -> NotificationEvent:
        """Create aggregated notification from multiple events"""
        first = notifications[0]
        count = len(notifications)
        
        # Get unique stream IDs
        stream_ids = list(set(n.stream_id for n in notifications if n.stream_id is not None))
        
        # Create aggregated message based on type
        if first.type == NotificationType.VIOLENCE_ALERT:
            if len(stream_ids) == 1:
                message = f"🚨 {count} violence alerts detected in stream {stream_ids[0]}"
            else:
                message = f"🚨 {count} violence alerts detected across {len(stream_ids)} streams"
        
        elif first.type == NotificationType.INCIDENT_FINALIZED:
            message = f"📋 {count} incidents finalized"
        
        else:
            message = f"{count} {first.type.value} notifications"
        
        # Aggregate confidence scores for violence alerts
        confidences = [n.data.get('confidence', 0) for n in notifications if 'confidence' in n.data]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return NotificationEvent(
            id=f"aggregated_{int(time.time())}",
            type=first.type,
            priority=max(n.priority for n in notifications),
            stream_id=stream_ids[0] if len(stream_ids) == 1 else None,
            message=message,
            data={
                'aggregated_count': count,
                'stream_ids': stream_ids,
                'avg_confidence': avg_confidence,
                'time_span': max(n.timestamp for n in notifications) - min(n.timestamp for n in notifications),
                'individual_notifications': [n.id for n in notifications]
            },
            timestamp=time.time()
        )


class SmartNotificationManager:
    """
    Smart notification manager with rate limiting, deduplication, and aggregation.
    
    Features:
    - Rate limiting to prevent spam
    - Notification aggregation for related events
    - Priority-based processing
    - Deduplication of similar notifications
    - WebSocket subscription management
    """
    
    def __init__(self):
        # Core components
        self.rate_limiter = NotificationRateLimiter(max_notifications=5, time_window=60.0)
        self.aggregator = NotificationAggregator(aggregation_window=10.0)
        
        # Processing queues
        self.notification_queue = queue.PriorityQueue()
        self.failed_notifications = queue.Queue()
        
        # WebSocket subscription management
        self.websocket_subscriptions: Dict[str, Set[str]] = defaultdict(set)  # topic -> set of connection_ids
        self.connection_topics: Dict[str, Set[str]] = defaultdict(set)        # connection_id -> set of topics
        
        # Processing control
        self.is_running = False
        self.processor_thread = None
        self.retry_thread = None
        
        # Deduplication
        self.recent_notifications: deque = deque(maxlen=1000)
        self.duplicate_window = 30.0  # seconds
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'rate_limited': 0,
            'aggregated': 0,
            'deduplicated': 0,
            'failed': 0,
            'websocket_sends': 0,
            'discord_sends': 0
        }
        
        # Set up aggregator callback
        self.aggregator.set_aggregate_callback(self._send_notification_internal)
        
        logging.info("SmartNotificationManager initialized")
    
    def start(self):
        """Start notification processing"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start processing threads
        self.processor_thread = threading.Thread(
            target=self._notification_processing_loop,
            name="NotificationProcessor",
            daemon=True
        )
        
        self.retry_thread = threading.Thread(
            target=self._retry_processing_loop, 
            name="NotificationRetry",
            daemon=True
        )
        
        self.processor_thread.start()
        self.retry_thread.start()
        
        logging.info("SmartNotificationManager started")
    
    def stop(self):
        """Stop notification processing"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Wait for threads
        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=2.0)
        
        if self.retry_thread and self.retry_thread.is_alive():
            self.retry_thread.join(timeout=2.0)
        
        logging.info("SmartNotificationManager stopped")
    
    def send_notification(
        self,
        notification_type: NotificationType,
        priority: NotificationPriority,
        message: str,
        stream_id: Optional[int] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send notification through the smart system"""
        
        notification = NotificationEvent(
            id=f"{notification_type.value}_{int(time.time() * 1000)}",
            type=notification_type,
            priority=priority,
            stream_id=stream_id,
            message=message,
            data=data or {},
            timestamp=time.time()
        )
        
        # Check for duplicates
        if self._is_duplicate(notification):
            self.stats['deduplicated'] += 1
            logging.debug(f"Duplicate notification filtered: {notification.id}")
            return False
        
        # Add to recent notifications
        self.recent_notifications.append((notification.timestamp, notification.id, notification.message))
        
        # Check rate limiting
        rate_key = f"{notification.type.value}_{notification.stream_id or 'global'}"
        if not self.rate_limiter.can_send(rate_key):
            self.stats['rate_limited'] += 1
            logging.warning(f"Rate limited notification: {notification.id}")
            return False
        
        # Try aggregation (returns True if aggregated, False if should send immediately)
        if self.aggregator.add_notification(notification):
            self.stats['aggregated'] += 1
            logging.debug(f"Notification aggregated: {notification.id}")
            return True
        
        # Send immediately
        return self._queue_notification(notification)
    
    def _is_duplicate(self, notification: NotificationEvent) -> bool:
        """Check if notification is a duplicate of recent ones"""
        current_time = notification.timestamp
        
        # Check recent notifications within duplicate window
        for timestamp, notif_id, message in self.recent_notifications:
            if current_time - timestamp > self.duplicate_window:
                continue
            
            # Check for similar messages (simple string similarity)
            if (notification.message == message and 
                abs(current_time - timestamp) < 5.0):  # Within 5 seconds
                return True
        
        return False
    
    def _queue_notification(self, notification: NotificationEvent) -> bool:
        """Queue notification for processing"""
        try:
            # Use negative priority so higher priority items are processed first
            priority_value = -notification.priority.value
            self.notification_queue.put((priority_value, notification.timestamp, notification), timeout=1.0)
            return True
        except queue.Full:
            logging.error("Notification queue full, dropping notification")
            return False
    
    def _send_notification_internal(self, notification: NotificationEvent):
        """Internal method to send notification (called by aggregator or directly)"""
        if not self._queue_notification(notification):
            # If queuing fails, try to send directly
            self._process_notification(notification)
    
    def _notification_processing_loop(self):
        """Main notification processing loop"""
        logging.info("Notification processing loop started")
        
        while self.is_running:
            try:
                # Get notification with timeout
                _, _, notification = self.notification_queue.get(timeout=1.0)
                self._process_notification(notification)
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in notification processing loop: {e}")
        
        logging.info("Notification processing loop ended")
    
    def _process_notification(self, notification: NotificationEvent):
        """Process individual notification"""
        try:
            success = False
            
            # Send via WebSocket
            if self._send_websocket_notification(notification):
                success = True
                self.stats['websocket_sends'] += 1
            
            # Send via Discord (for certain types)
            if notification.type in [NotificationType.VIOLENCE_ALERT, NotificationType.SYSTEM_STATUS]:
                if self._send_discord_notification(notification):
                    success = True
                    self.stats['discord_sends'] += 1
            
            if success:
                self.stats['total_processed'] += 1
                logging.debug(f"Processed notification: {notification.id}")
            else:
                # Queue for retry
                self._queue_for_retry(notification)
            
        except Exception as e:
            logging.error(f"Error processing notification {notification.id}: {e}")
            self._queue_for_retry(notification)
    
    def _send_websocket_notification(self, notification: NotificationEvent) -> bool:
        """Send notification via WebSocket"""
        try:
            # This would integrate with your existing WebSocket manager
            # For now, just log
            logging.info(f"WebSocket: {notification.message}")
            return True
        except Exception as e:
            logging.error(f"WebSocket send failed: {e}")
            return False
    
    def _send_discord_notification(self, notification: NotificationEvent) -> bool:
        """Send notification via Discord"""
        try:
            # This would integrate with your existing Discord notifier
            # For now, just log
            logging.info(f"Discord: {notification.message}")
            return True
        except Exception as e:
            logging.error(f"Discord send failed: {e}")
            return False
    
    def _queue_for_retry(self, notification: NotificationEvent):
        """Queue notification for retry"""
        notification.retry_count += 1
        
        if notification.retry_count <= notification.max_retries:
            try:
                self.failed_notifications.put(notification, timeout=1.0)
            except queue.Full:
                logging.error("Failed notification queue full")
                self.stats['failed'] += 1
        else:
            logging.error(f"Notification {notification.id} exceeded max retries")
            self.stats['failed'] += 1
    
    def _retry_processing_loop(self):
        """Process failed notifications for retry"""
        while self.is_running:
            try:
                notification = self.failed_notifications.get(timeout=5.0)
                
                # Wait before retry (exponential backoff)
                wait_time = min(2 ** notification.retry_count, 60)
                time.sleep(wait_time)
                
                # Retry processing
                self._process_notification(notification)
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in retry processing loop: {e}")
    
    def subscribe_websocket(self, connection_id: str, topics: List[str]):
        """Subscribe WebSocket connection to topics"""
        for topic in topics:
            self.websocket_subscriptions[topic].add(connection_id)
            self.connection_topics[connection_id].add(topic)
        
        logging.debug(f"WebSocket {connection_id} subscribed to {topics}")
    
    def unsubscribe_websocket(self, connection_id: str):
        """Unsubscribe WebSocket connection from all topics"""
        topics_to_remove = self.connection_topics.get(connection_id, set())
        
        for topic in topics_to_remove:
            self.websocket_subscriptions[topic].discard(connection_id)
        
        if connection_id in self.connection_topics:
            del self.connection_topics[connection_id]
        
        logging.debug(f"WebSocket {connection_id} unsubscribed from all topics")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get notification system statistics"""
        return {
            **self.stats,
            'is_running': self.is_running,
            'queue_sizes': {
                'notification_queue': self.notification_queue.qsize(),
                'failed_queue': self.failed_notifications.qsize()
            },
            'subscriptions': {
                'total_connections': len(self.connection_topics),
                'total_topics': len(self.websocket_subscriptions)
            },
            'rate_limiting': {
                'active_limits': len(self.rate_limiter.notification_history)
            }
        }


# Global instance (to be initialized in main.py)
smart_notification_manager: Optional[SmartNotificationManager] = None