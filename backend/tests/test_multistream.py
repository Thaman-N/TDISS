"""
Multi-Stream Performance Test Script
==================================

This script helps test multiple RTSP streams to validate the batch processing
system's performance under load.

Usage:
1. Start with 2-3 test streams
2. Gradually increase to 6-8 streams
3. Monitor system performance and resource usage
"""

import asyncio
import aiohttp
import json
import time
from typing import List, Dict

class MultiStreamTester:
    """Test multiple RTSP streams simultaneously"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_streams = []
        
    def create_test_streams(self, count: int) -> List[Dict]:
        """Create test stream configurations"""
        # You can replace these with real RTSP URLs for testing
        test_streams = []
        
        for i in range(1, count + 1):
            stream = {
                "name": f"Test Stream {i}",
                "rtsp_url": f"rtsp://your-test-server.com:8554/stream{i}",  # Replace with real URLs
                "description": f"Performance test stream {i}"
            }
            test_streams.append(stream)
        
        return test_streams
    
    async def add_stream(self, session: aiohttp.ClientSession, stream_config: Dict) -> Dict:
        """Add a single stream via API"""
        try:
            async with session.post(f"{self.base_url}/api/streams", json=stream_config) as response:
                result = await response.json()
                print(f"  ✅ Added stream: {stream_config['name']} - ID: {result.get('id', 'Unknown')}")
                return result
        except Exception as e:
            print(f"  ❌ Failed to add stream {stream_config['name']}: {e}")
            return {"error": str(e)}
    
    async def start_stream(self, session: aiohttp.ClientSession, stream_id: int) -> bool:
        """Start a stream via API"""
        try:
            async with session.post(f"{self.base_url}/api/streams/{stream_id}/start") as response:
                result = await response.json()
                success = result.get("success", False)
                if success:
                    print(f"  ✅ Started stream ID: {stream_id}")
                else:
                    print(f"  ❌ Failed to start stream ID: {stream_id} - {result.get('message', 'Unknown error')}")
                return success
        except Exception as e:
            print(f"  ❌ Error starting stream {stream_id}: {e}")
            return False
    
    async def stop_stream(self, session: aiohttp.ClientSession, stream_id: int) -> bool:
        """Stop a stream via API"""
        try:
            async with session.post(f"{self.base_url}/api/streams/{stream_id}/stop") as response:
                result = await response.json()
                success = result.get("success", False)
                if success:
                    print(f"  ✅ Stopped stream ID: {stream_id}")
                else:
                    print(f"  ❌ Failed to stop stream ID: {stream_id} - {result.get('message', 'Unknown error')}")
                return success
        except Exception as e:
            print(f"  ❌ Error stopping stream {stream_id}: {e}")
            return False
    
    async def get_system_status(self, session: aiohttp.ClientSession) -> Dict:
        """Get system performance status"""
        try:
            async with session.get(f"{self.base_url}/api/status") as response:
                return await response.json()
        except Exception as e:
            print(f"  ❌ Error getting system status: {e}")
            return {}
    
    async def get_streams_status(self, session: aiohttp.ClientSession) -> List[Dict]:
        """Get all streams status"""
        try:
            async with session.get(f"{self.base_url}/api/streams") as response:
                return await response.json()
        except Exception as e:
            print(f"  ❌ Error getting streams status: {e}")
            return []
    
    async def run_progressive_test(self, max_streams: int = 6):
        """Run progressive multi-stream test"""
        print("🧪 Multi-Stream Performance Test")
        print("=" * 50)
        
        async with aiohttp.ClientSession() as session:
            # Test connectivity first
            print("📡 Testing API connectivity...")
            status = await self.get_system_status(session)
            if not status:
                print("❌ Cannot connect to TDISS API. Make sure the server is running.")
                return False
            
            print(f"✅ Connected to TDISS API - System Status: {status.get('system_status', 'Unknown')}")
            
            # Progressive testing
            for stream_count in range(1, max_streams + 1):
                print(f"\n🔄 Testing with {stream_count} stream(s)...")
                
                # Create test stream configurations
                test_configs = self.create_test_streams(stream_count)
                stream_ids = []
                
                # Add streams to system
                print(f"  Adding {stream_count} stream(s)...")
                for config in test_configs:
                    result = await self.add_stream(session, config)
                    if "id" in result:
                        stream_ids.append(result["id"])
                
                if not stream_ids:
                    print(f"  ❌ No streams added successfully for count {stream_count}")
                    continue
                
                # Start all streams
                print(f"  Starting {len(stream_ids)} stream(s)...")
                start_time = time.time()
                
                start_results = []
                for stream_id in stream_ids:
                    success = await self.start_stream(session, stream_id)
                    start_results.append(success)
                
                startup_time = time.time() - start_time
                successful_starts = sum(start_results)
                
                print(f"  📊 Startup Results: {successful_starts}/{len(stream_ids)} streams started in {startup_time:.2f}s")
                
                if successful_starts > 0:
                    # Monitor for a bit
                    print(f"  📈 Monitoring performance for 30 seconds...")
                    await asyncio.sleep(30)
                    
                    # Get performance metrics
                    status = await self.get_system_status(session)
                    streams_status = await self.get_streams_status(session)
                    
                    print(f"  📊 Performance Metrics:")
                    print(f"    Active Streams: {len([s for s in streams_status if s.get('status') == 'active'])}")
                    print(f"    System Status: {status.get('system_status', 'Unknown')}")
                    print(f"    Total Jobs: {status.get('total_jobs', 0)}")
                    print(f"    Active Jobs: {status.get('active_jobs', 0)}")
                    
                    # Stop all streams
                    print(f"  🛑 Stopping all streams...")
                    for stream_id in stream_ids:
                        await self.stop_stream(session, stream_id)
                    
                    # Brief pause between tests
                    await asyncio.sleep(5)
                
                print(f"  ✅ Test with {stream_count} stream(s) completed")
            
            print(f"\n🎉 Progressive testing completed! Tested up to {max_streams} streams.")
            
            # Final system status
            final_status = await self.get_system_status(session)
            print(f"\n📊 Final System Status: {json.dumps(final_status, indent=2)}")
            
            return True

async def main():
    """Main test function"""
    tester = MultiStreamTester()
    
    print("⚠️  Important: This test requires real RTSP stream URLs.")
    print("   Update the test_streams configurations with your actual RTSP endpoints.")
    print("   Press Ctrl+C to cancel if you need to configure URLs first.\n")
    
    try:
        await asyncio.sleep(3)  # Give user time to read warning
        success = await tester.run_progressive_test(max_streams=6)
        
        if success:
            print("\n✅ Multi-stream testing completed successfully!")
            print("   Your batch processing system can handle multiple streams efficiently.")
        else:
            print("\n⚠️ Some tests failed. Check the output above for details.")
            
    except KeyboardInterrupt:
        print("\n🛑 Testing cancelled by user.")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())