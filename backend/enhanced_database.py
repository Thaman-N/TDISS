"""
Enhanced database management system with connection pooling and batch operations.
Addresses database bottlenecks and concurrent access issues.
"""

import sqlite3
import threading
import queue
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class BatchDatabaseOperation:
    """Container for batch database operations"""
    operation_type: str  # 'insert', 'update', 'delete'
    table_name: str
    queries: List[Tuple[str, tuple]]  # [(sql, params), ...]
    callback: Optional[callable] = None


class EnhancedDatabaseManager:
    """
    Enhanced database manager with connection pooling and batch operations.
    
    Features:
    - Connection pooling with WAL mode
    - Batch insert/update operations
    - Transaction management
    - Automatic retry logic
    - Performance monitoring
    """
    
    def __init__(self, db_path: str, pool_size: int = 8):
        self.db_path = Path(db_path)
        self.pool_size = pool_size
        
        # Connection pool
        self.connection_pool = queue.Queue(maxsize=pool_size)
        self.active_connections = 0
        self._pool_lock = threading.Lock()
        
        # Batch operations
        self.batch_queue = queue.Queue()
        self.is_running = False
        self.batch_processor_thread = None
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'batch_operations': 0,
            'connection_checkouts': 0,
            'connection_errors': 0,
            'avg_query_time': 0.0,
            'active_connections': 0
        }
        
        self.query_times = []
        
        # Initialize pool
        self._initialize_connection_pool()
        
        logging.info(f"Enhanced database manager initialized: {db_path}")
    
    def _initialize_connection_pool(self):
        """Initialize connection pool"""
        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create initial connections
        for _ in range(self.pool_size):
            conn = self._create_connection()
            if conn:
                self.connection_pool.put(conn)
        
        logging.info(f"Database connection pool initialized with {self.pool_size} connections")
    
    def _create_connection(self) -> Optional[sqlite3.Connection]:
        """Create optimized SQLite connection"""
        try:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            
            # Optimize connection settings
            conn.execute('PRAGMA journal_mode=WAL')     # Better concurrency
            conn.execute('PRAGMA synchronous=NORMAL')   # Balance safety/performance  
            conn.execute('PRAGMA cache_size=10000')     # 10MB cache
            conn.execute('PRAGMA temp_store=MEMORY')    # Use memory for temp tables
            conn.execute('PRAGMA mmap_size=268435456')  # 256MB memory-mapped I/O
            conn.execute('PRAGMA page_size=4096')       # Optimize page size
            
            # Enable foreign key constraints
            conn.execute('PRAGMA foreign_keys=ON')
            
            # Set row factory for dict-like access
            conn.row_factory = sqlite3.Row
            
            with self._pool_lock:
                self.active_connections += 1
            
            return conn
            
        except Exception as e:
            logging.error(f"Error creating database connection: {e}")
            return None
    
    def get_connection(self) -> Optional[sqlite3.Connection]:
        """Get connection from pool"""
        try:
            # Try to get from pool with timeout
            conn = self.connection_pool.get(timeout=5.0)
            self.stats['connection_checkouts'] += 1
            return conn
            
        except queue.Empty:
            # Pool exhausted, create new connection if under limit
            with self._pool_lock:
                if self.active_connections < self.pool_size * 2:  # Allow some overflow
                    conn = self._create_connection()
                    if conn:
                        return conn
            
            self.stats['connection_errors'] += 1
            logging.warning("Database connection pool exhausted")
            return None
    
    def return_connection(self, conn: sqlite3.Connection):
        """Return connection to pool"""
        if conn is None:
            return
        
        try:
            # Check if connection is still valid
            conn.execute('SELECT 1')
            
            # Return to pool if space available
            if self.connection_pool.qsize() < self.pool_size:
                self.connection_pool.put(conn)
            else:
                # Pool full, close connection
                conn.close()
                with self._pool_lock:
                    self.active_connections = max(0, self.active_connections - 1)
                    
        except Exception as e:
            # Connection is bad, close it
            try:
                conn.close()
            except:
                pass
            with self._pool_lock:
                self.active_connections = max(0, self.active_connections - 1)
            logging.warning(f"Closed bad database connection: {e}")
    
    def execute_query(self, sql: str, params: tuple = None, fetch: bool = False) -> Any:
        """Execute single query with connection pooling"""
        conn = self.get_connection()
        if not conn:
            raise Exception("Could not get database connection")
        
        try:
            start_time = time.time()
            
            cursor = conn.cursor()
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            
            result = None
            if fetch:
                if sql.strip().lower().startswith('select'):
                    result = cursor.fetchall()
                else:
                    result = cursor.fetchone()
            else:
                conn.commit()
                result = cursor.lastrowid if cursor.lastrowid else cursor.rowcount
            
            # Update statistics
            query_time = time.time() - start_time
            self.query_times.append(query_time)
            if len(self.query_times) > 1000:
                self.query_times.pop(0)
            
            self.stats['total_queries'] += 1
            self.stats['avg_query_time'] = sum(self.query_times) / len(self.query_times)
            
            return result
            
        finally:
            self.return_connection(conn)
    
    def execute_batch(self, operations: List[Tuple[str, tuple]]) -> bool:
        """Execute batch operations in single transaction"""
        if not operations:
            return True
        
        conn = self.get_connection()
        if not conn:
            return False
        
        try:
            start_time = time.time()
            
            # Start transaction
            conn.execute('BEGIN TRANSACTION')
            
            cursor = conn.cursor()
            for sql, params in operations:
                cursor.execute(sql, params)
            
            # Commit transaction
            conn.commit()
            
            # Update statistics
            batch_time = time.time() - start_time
            self.stats['batch_operations'] += 1
            self.stats['total_queries'] += len(operations)
            
            logging.debug(f"Executed batch of {len(operations)} operations in {batch_time:.3f}s")
            return True
            
        except Exception as e:
            # Rollback on error
            try:
                conn.rollback()
            except:
                pass
            logging.error(f"Batch operation failed: {e}")
            return False
            
        finally:
            self.return_connection(conn)
    
    def start_batch_processing(self):
        """Start batch processing thread"""
        if self.is_running:
            return
        
        self.is_running = True
        self.batch_processor_thread = threading.Thread(
            target=self._batch_processing_loop,
            name="DatabaseBatchProcessor",
            daemon=True
        )
        self.batch_processor_thread.start()
        
        logging.info("Database batch processing started")
    
    def stop_batch_processing(self):
        """Stop batch processing"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.batch_processor_thread and self.batch_processor_thread.is_alive():
            self.batch_processor_thread.join(timeout=5.0)
        
        logging.info("Database batch processing stopped")
    
    def queue_batch_operation(self, operation: BatchDatabaseOperation):
        """Queue batch operation for processing"""
        try:
            self.batch_queue.put(operation, timeout=1.0)
        except queue.Full:
            logging.warning("Database batch queue full, executing immediately")
            # Execute immediately if queue is full
            self._process_batch_operation(operation)
    
    def _batch_processing_loop(self):
        """Process batch operations"""
        while self.is_running:
            try:
                operation = self.batch_queue.get(timeout=1.0)
                self._process_batch_operation(operation)
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in batch processing loop: {e}")
    
    def _process_batch_operation(self, operation: BatchDatabaseOperation):
        """Process individual batch operation"""
        try:
            success = self.execute_batch(operation.queries)
            
            if operation.callback:
                operation.callback(success)
            
        except Exception as e:
            logging.error(f"Error processing batch operation: {e}")
            if operation.callback:
                operation.callback(False)
    
    def close_all_connections(self):
        """Close all connections in pool"""
        # Stop batch processing first
        self.stop_batch_processing()
        
        # Close all connections
        while not self.connection_pool.empty():
            try:
                conn = self.connection_pool.get_nowait()
                conn.close()
            except Exception as e:
                logging.error(f"Error closing connection: {e}")
        
        with self._pool_lock:
            self.active_connections = 0
        
        logging.info("All database connections closed")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database performance statistics"""
        with self._pool_lock:
            current_active = self.active_connections
        
        return {
            **self.stats,
            'active_connections': current_active,
            'pool_size': self.connection_pool.qsize(),
            'batch_queue_size': self.batch_queue.qsize(),
            'is_running': self.is_running
        }


class BatchEventSaver:
    """
    Specialized class for batch saving violence detection events.
    Optimizes the common case of saving multiple events from batch processing.
    """
    
    def __init__(self, db_manager: EnhancedDatabaseManager):
        self.db_manager = db_manager
        self.pending_events = []
        self.pending_incidents = []
        self._lock = threading.Lock()
        
        # Batch settings
        self.batch_size = 20
        self.batch_timeout = 5.0  # seconds
        self.last_flush = time.time()
    
    def save_event(self, event_data: Dict[str, Any]) -> Optional[int]:
        """Save single event (may be batched)"""
        with self._lock:
            self.pending_events.append(event_data)
            
            # Check if we should flush
            if (len(self.pending_events) >= self.batch_size or 
                time.time() - self.last_flush >= self.batch_timeout):
                return self._flush_events()
        
        return None  # Will be batched
    
    def save_incident(self, incident_data: Dict[str, Any]) -> Optional[int]:
        """Save incident data (may be batched)"""
        with self._lock:
            self.pending_incidents.append(incident_data)
            
            # Incidents are less frequent, use smaller batch
            if len(self.pending_incidents) >= 5:
                return self._flush_incidents()
        
        return None
    
    def _flush_events(self) -> Optional[int]:
        """Flush pending events to database"""
        if not self.pending_events:
            return None
        
        events_to_save = self.pending_events.copy()
        self.pending_events.clear()
        self.last_flush = time.time()
        
        # Create batch operations
        operations = []
        for event in events_to_save:
            sql = '''
                INSERT INTO violence_events 
                (timestamp, source_type, source_id, filename, start_time, end_time, 
                 duration, confidence, thumbnail_path, clip_path, metadata, 
                 incident_status, incident_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            params = (
                event.get('timestamp'),
                event.get('source_type'),
                event.get('source_id'),
                event.get('filename'),
                event.get('start_time'),
                event.get('end_time'),
                event.get('duration'),
                event.get('confidence'),
                event.get('thumbnail_path'),
                event.get('clip_path'),
                event.get('metadata'),
                event.get('incident_status'),
                event.get('incident_id')
            )
            operations.append((sql, params))
        
        # Execute batch
        batch_op = BatchDatabaseOperation(
            operation_type='insert',
            table_name='violence_events',
            queries=operations
        )
        
        self.db_manager.queue_batch_operation(batch_op)
        
        logging.info(f"Batched {len(events_to_save)} violence events for database save")
        return len(events_to_save)
    
    def _flush_incidents(self) -> Optional[int]:
        """Flush pending incidents to database"""
        if not self.pending_incidents:
            return None
        
        incidents_to_save = self.pending_incidents.copy()
        self.pending_incidents.clear()
        
        # Create batch operations for incidents
        operations = []
        for incident in incidents_to_save:
            sql = '''
                INSERT INTO stitched_incidents 
                (incident_id, stream_id, stream_name, start_timestamp, end_timestamp,
                 total_duration, detection_count, avg_confidence, max_confidence,
                 stitched_clip_path, timeline_data, event_ids)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            params = (
                incident.get('incident_id'),
                incident.get('stream_id'),
                incident.get('stream_name'),
                incident.get('start_timestamp'),
                incident.get('end_timestamp'),
                incident.get('total_duration'),
                incident.get('detection_count'),
                incident.get('avg_confidence'),
                incident.get('max_confidence'),
                incident.get('stitched_clip_path'),
                incident.get('timeline_data'),
                incident.get('event_ids')
            )
            operations.append((sql, params))
        
        # Execute batch
        batch_op = BatchDatabaseOperation(
            operation_type='insert',
            table_name='stitched_incidents',
            queries=operations
        )
        
        self.db_manager.queue_batch_operation(batch_op)
        
        logging.info(f"Batched {len(incidents_to_save)} incidents for database save")
        return len(incidents_to_save)
    
    def force_flush(self):
        """Force flush all pending data"""
        with self._lock:
            self._flush_events()
            self._flush_incidents()


# Global instances (to be initialized in main.py)
enhanced_db_manager: Optional[EnhancedDatabaseManager] = None
batch_event_saver: Optional[BatchEventSaver] = None