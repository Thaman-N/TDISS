"""
Integration Test Suite for TDISS Batch Processing System
=====================================================

This script provides comprehensive testing for the new integrated batch
processing architecture, ensuring all components work together correctly.

Run this script to validate:
- System initialization and startup
- Stream management and processing
- Database operations and event stitching
- Notification system functionality
- Performance characteristics
- Error handling and recovery
"""

import asyncio
import sys
import os
import tempfile
import shutil
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional

# Add backend to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from integration_system import (
    IntegratedViolenceDetectionSystem, 
    StreamConfiguration, SystemStats
)

class IntegrationTestSuite:
    """Comprehensive test suite for the integrated system."""
    
    def __init__(self):
        self.test_dir = None
        self.system: Optional[IntegratedViolenceDetectionSystem] = None
        self.test_results: Dict[str, bool] = {}
        
    async def setup(self):
        """Set up test environment."""
        print("Setting up test environment...")
        
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp(prefix="tdiss_test_")
        print(f"Test directory: {self.test_dir}")
        
        # Test system configuration
        config = {
            'model_path': './models/rwf9425.pth',  # Adjust path as needed
            'device': 'cpu',  # Use CPU for testing to avoid GPU conflicts
            'max_streams': 4,  # Smaller limit for testing
            'max_batch_size': 2,
            'detection_threshold': 0.5,
            'max_rtsp_connections': 4,
            'bandwidth_limit_mbps': 10.0,
            'max_gpu_memory': 0.5,
            'min_batch_size': 1,
            'default_batch_size': 1,
            'storage_path': os.path.join(self.test_dir, 'data'),
            'max_storage_gb': 1,  # Small limit for testing
            'db_path': os.path.join(self.test_dir, 'test_violence_events.db'),
            'max_db_connections': 3,
            'db_timeout': 10,
            'discord_webhook_url': None,  # Disable Discord for testing
            'notifications_per_stream': 5,
            'notifications_per_hour': 50,
            'critical_burst_limit': 10,
            'notification_window': 60,
            'queue_timeout': 2.0,
            'db_batch_size': 10,
            'db_flush_interval': 1.0
        }
        
        # Initialize system
        self.system = IntegratedViolenceDetectionSystem(config)
        return await self.system.initialize()
    
    async def cleanup(self):
        """Clean up test environment."""
        print("Cleaning up test environment...")
        
        if self.system:
            await self.system.cleanup()
        
        if self.test_dir and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    async def test_system_initialization(self) -> bool:
        """Test system initialization."""
        print("\n🔧 Testing System Initialization...")
        
        try:
            # Check if system is initialized
            if not self.system.is_initialized:
                print("❌ System not initialized")
                return False
            
            # Check if all managers are created
            managers = [
                'connection_manager', 'gpu_manager', 'file_manager', 
                'db_pool', 'batch_manager', 'resource_pool',
                'stream_manager', 'notification_manager', 'enhanced_db'
            ]
            
            for manager in managers:
                if not hasattr(self.system, manager) or getattr(self.system, manager) is None:
                    print(f"❌ Manager {manager} not initialized")
                    return False
            
            print("✅ All system components initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error in system initialization test: {e}")
            return False
    
    async def test_stream_lifecycle(self) -> bool:
        """Test stream start/stop lifecycle."""
        print("\n📺 Testing Stream Lifecycle...")
        
        try:
            # Create test stream configuration
            stream_config = StreamConfiguration(
                stream_id="test_stream_1",
                rtsp_url="rtsp://localhost:8554/test",  # Mock RTSP URL
                stream_name="Test Stream 1",
                detection_threshold=0.5,
                priority_level=1
            )
            
            # Test stream start
            print("  Starting test stream...")
            success = await self.system.start_stream(stream_config)
            if not success:
                print("❌ Failed to start test stream")
                return False
            
            # Check if stream is active
            await asyncio.sleep(1)  # Give system time to process
            status = await self.system.get_stream_status("test_stream_1")
            if not status:
                print("❌ Stream status not available")
                return False
            
            print(f"  Stream status: {status['status']}")
            
            # Test stream stop
            print("  Stopping test stream...")
            success = await self.system.stop_stream("test_stream_1")
            if not success:
                print("❌ Failed to stop test stream")
                return False
            
            # Verify stream is stopped
            await asyncio.sleep(1)
            status = await self.system.get_stream_status("test_stream_1")
            if status:
                print("❌ Stream still active after stop")
                return False
            
            print("✅ Stream lifecycle test passed")
            return True
            
        except Exception as e:
            print(f"❌ Error in stream lifecycle test: {e}")
            traceback.print_exc()
            return False
    
    async def test_multiple_streams(self) -> bool:
        """Test multiple stream handling."""
        print("\n📡 Testing Multiple Streams...")
        
        try:
            # Create multiple test streams
            stream_configs = [
                StreamConfiguration(
                    stream_id=f"test_stream_{i}",
                    rtsp_url=f"rtsp://localhost:8554/test{i}",
                    stream_name=f"Test Stream {i}",
                    detection_threshold=0.5,
                    priority_level=i % 3 + 1  # Different priorities
                )
                for i in range(1, 4)  # 3 streams
            ]
            
            # Start all streams
            for config in stream_configs:
                print(f"  Starting {config.stream_name}...")
                success = await self.system.start_stream(config)
                if not success:
                    print(f"❌ Failed to start {config.stream_name}")
                    return False
            
            await asyncio.sleep(2)  # Let streams initialize
            
            # Check system status
            system_status = await self.system.get_system_status()
            print(f"  Active streams: {system_status['active_streams']}")
            
            if system_status['active_streams'] != 3:
                print(f"❌ Expected 3 active streams, got {system_status['active_streams']}")
                return False
            
            # Stop all streams
            for config in stream_configs:
                print(f"  Stopping {config.stream_name}...")
                success = await self.system.stop_stream(config.stream_id)
                if not success:
                    print(f"❌ Failed to stop {config.stream_name}")
                    return False
            
            await asyncio.sleep(1)
            
            # Verify all stopped
            final_status = await self.system.get_system_status()
            if final_status['active_streams'] != 0:
                print(f"❌ Expected 0 active streams, got {final_status['active_streams']}")
                return False
            
            print("✅ Multiple streams test passed")
            return True
            
        except Exception as e:
            print(f"❌ Error in multiple streams test: {e}")
            traceback.print_exc()
            return False
    
    async def test_system_monitoring(self) -> bool:
        """Test system monitoring and statistics."""
        print("\n📊 Testing System Monitoring...")
        
        try:
            # Get initial system status
            status = await self.system.get_system_status()
            
            required_fields = [
                'system_initialized', 'uptime_seconds', 'active_streams',
                'total_frames_processed', 'gpu_memory_usage', 'error_count'
            ]
            
            for field in required_fields:
                if field not in status:
                    print(f"❌ Missing status field: {field}")
                    return False
            
            print(f"  System initialized: {status['system_initialized']}")
            print(f"  Uptime: {status['uptime_seconds']} seconds")
            print(f"  Active streams: {status['active_streams']}")
            print(f"  GPU memory usage: {status['gpu_memory_usage']:.1%}")
            print(f"  Error count: {status['error_count']}")
            
            if not status['system_initialized']:
                print("❌ System not showing as initialized")
                return False
            
            print("✅ System monitoring test passed")
            return True
            
        except Exception as e:
            print(f"❌ Error in system monitoring test: {e}")
            return False
    
    async def test_error_handling(self) -> bool:
        """Test error handling and recovery."""
        print("\n🚨 Testing Error Handling...")
        
        try:
            # Test invalid stream configuration
            invalid_config = StreamConfiguration(
                stream_id="invalid_stream",
                rtsp_url="invalid://url",
                stream_name="Invalid Stream",
                detection_threshold=0.5
            )
            
            print("  Testing invalid stream URL...")
            success = await self.system.start_stream(invalid_config)
            
            # This should fail gracefully without crashing
            if success:
                print("❌ Invalid stream unexpectedly succeeded")
                # Clean up if it somehow worked
                await self.system.stop_stream("invalid_stream")
                return False
            
            print("  ✓ Invalid stream correctly rejected")
            
            # Test stopping non-existent stream
            print("  Testing stop of non-existent stream...")
            success = await self.system.stop_stream("nonexistent_stream")
            
            # This should handle gracefully
            print(f"  ✓ Non-existent stream stop handled: {success}")
            
            # Test getting status of non-existent stream
            print("  Testing status of non-existent stream...")
            status = await self.system.get_stream_status("nonexistent_stream")
            
            if status is not None:
                print("❌ Non-existent stream returned status")
                return False
            
            print("  ✓ Non-existent stream status correctly returned None")
            
            print("✅ Error handling test passed")
            return True
            
        except Exception as e:
            print(f"❌ Error in error handling test: {e}")
            return False
    
    async def test_performance_characteristics(self) -> bool:
        """Test basic performance characteristics."""
        print("\n⚡ Testing Performance Characteristics...")
        
        try:
            # Test system response time
            start_time = time.time()
            status = await self.system.get_system_status()
            response_time = time.time() - start_time
            
            print(f"  System status response time: {response_time:.3f}s")
            
            if response_time > 1.0:  # Should be very fast
                print("❌ System status response too slow")
                return False
            
            # Test stream start performance
            config = StreamConfiguration(
                stream_id="perf_test_stream",
                rtsp_url="rtsp://localhost:8554/perf",
                stream_name="Performance Test Stream"
            )
            
            start_time = time.time()
            success = await self.system.start_stream(config)
            start_duration = time.time() - start_time
            
            print(f"  Stream start time: {start_duration:.3f}s")
            
            if success:
                # Test stream stop performance
                start_time = time.time()
                await self.system.stop_stream("perf_test_stream")
                stop_duration = time.time() - start_time
                
                print(f"  Stream stop time: {stop_duration:.3f}s")
                
                # Basic performance thresholds
                if start_duration > 5.0:  # Should start within 5 seconds
                    print("❌ Stream start too slow")
                    return False
                
                if stop_duration > 2.0:  # Should stop within 2 seconds
                    print("❌ Stream stop too slow")
                    return False
            
            print("✅ Performance characteristics test passed")
            return True
            
        except Exception as e:
            print(f"❌ Error in performance test: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """Run all integration tests."""
        print("🧪 Starting TDISS Integration Test Suite")
        print("=" * 60)
        
        tests = [
            ("System Initialization", self.test_system_initialization),
            ("Stream Lifecycle", self.test_stream_lifecycle),
            ("Multiple Streams", self.test_multiple_streams),
            ("System Monitoring", self.test_system_monitoring),
            ("Error Handling", self.test_error_handling),
            ("Performance Characteristics", self.test_performance_characteristics)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            try:
                result = await test_func()
                results[test_name] = result
                self.test_results[test_name] = result
                
            except Exception as e:
                print(f"❌ Test '{test_name}' crashed: {e}")
                traceback.print_exc()
                results[test_name] = False
                self.test_results[test_name] = False
        
        return results
    
    def print_summary(self):
        """Print test results summary."""
        print("\n" + "=" * 60)
        print("🧪 INTEGRATION TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for result in self.test_results.values() if result)
        total = len(self.test_results)
        
        print(f"Tests Passed: {passed}/{total}")
        print(f"Success Rate: {passed/total*100:.1f}%")
        print()
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {status} - {test_name}")
        
        print()
        if passed == total:
            print("🎉 All integration tests passed! System is ready for production.")
        else:
            print(f"⚠️  {total - passed} test(s) failed. Please review and fix issues.")
        
        return passed == total

async def main():
    """Main test runner."""
    test_suite = IntegrationTestSuite()
    
    try:
        # Setup test environment
        setup_success = await test_suite.setup()
        
        if not setup_success:
            print("❌ Failed to set up test environment")
            return False
        
        print("✅ Test environment set up successfully")
        
        # Run all tests
        results = await test_suite.run_all_tests()
        
        # Print summary
        all_passed = test_suite.print_summary()
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Critical error in test runner: {e}")
        traceback.print_exc()
        return False
        
    finally:
        # Always cleanup
        await test_suite.cleanup()

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)