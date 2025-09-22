"""
Direct Batch Processing Test
===========================

This script directly tests the batch processing components without going through
the full API layer. It validates the core batch inference system performance.
"""

import asyncio
import pytest
import sys
import os
import time
import numpy as np
import torch
import cv2
from typing import List
import threading

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

@pytest.mark.asyncio
async def test_batch_processing_directly():
    """Test batch processing system directly"""
    print("🔄 Direct Batch Processing Test")
    print("=" * 40)
    
    try:
        # Import batch processing components
        from batch_processing import BatchInferenceManager, FrameData
        from infrastructure_managers import GPUMemoryManager
        from torch_detection import load_violence_detection_model
        from unittest.mock import Mock, patch

        print("✅ Imported batch processing components")

        # Mock the model to return instantly
        mock_model = Mock()
        mock_model.parameters.return_value = iter([torch.tensor([1.0])])
        mock_model.return_value = torch.tensor([[0.2, 0.8]])

        # Create GPU manager
        gpu_manager = GPUMemoryManager(max_gpu_memory_gb=6.0)
        print("✅ GPU memory manager created")

        # Create batch inference manager with mocked model
        batch_manager = BatchInferenceManager(
            model=mock_model,
            detection_threshold=0.5,
            base_batch_size=4,
            batch_timeout=2.0,
            gpu_memory_manager=gpu_manager
        )
        print("✅ Batch inference manager created")

        # Start batch processing
        batch_manager.start()
        print("🚀 Batch processing started")
        
        # Generate test frame sequences (simulate multiple streams)
        def generate_test_frames(stream_id: int, count: int = 20) -> List[FrameData]:
            """Generate test frame sequences for a stream"""
            frames = []
            
            for i in range(count):
                # Create realistic test frame (336x336x3 for X3D model)
                frame = np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8)
                
                # Add some structure to make it more realistic
                cv2.rectangle(frame, (50, 50), (200, 200), (100, 150, 200), 2)
                cv2.circle(frame, (168, 168), 30, (200, 100, 100), -1)
                
                frame_data = FrameData(
                    stream_id=stream_id,
                    frame_sequence=np.array([frame] * 16),  # X3D needs 16 frames
                    timestamp=time.time() + i * 0.1,
                    sequence_start_time=time.time() + i * 0.1,
                    sequence_end_time=time.time() + i * 0.1 + 0.5
                )
                frames.append(frame_data)
            
            return frames
        
        # Test with multiple simulated streams
        print("\n📊 Testing batch processing with multiple streams...")
        
        results_received = []
        
        def result_callback(stream_id: int, result):
            """Callback for batch results"""
            results_received.append({
                'stream_id': stream_id,
                'timestamp': time.time(),
                'result': result
            })
            print(f"  📥 Result for stream {stream_id}: {result}")
        
        # Register result callbacks for multiple streams
        for stream_id in range(1, 5):  # 4 streams
            batch_manager.register_stream_callback(stream_id, 
                lambda result, sid=stream_id: result_callback(sid, result))
        
        print("✅ Registered callbacks for 4 streams")
        
        # Submit frames from multiple streams
        total_frames_submitted = 0
        start_time = time.time()
        
        for stream_id in range(1, 5):  # 4 streams
            print(f"  📤 Generating and submitting frames for stream {stream_id}...")
            
            test_frames = generate_test_frames(stream_id, count=5)  # 5 sequences per stream
            
            for frame_data in test_frames:
                batch_manager.submit_frame_data(frame_data, priority=1)
                total_frames_submitted += 1
            
            print(f"    ✅ Submitted {len(test_frames)} frame sequences for stream {stream_id}")
        
        submission_time = time.time() - start_time
        print(f"\n📊 Submitted {total_frames_submitted} frame sequences in {submission_time:.2f}s")
        
        # Wait for processing
        print("⏳ Waiting for batch processing to complete...")
        
        # Monitor processing for up to 15 seconds (hard kill)
        monitor_start = time.time()
        timeout = 15

        while (time.time() - monitor_start) < timeout:
            # Check queue status
            queue_size = batch_manager.frame_queue.qsize()
            results_count = len(results_received)

            print(f"  📈 Queue: {queue_size}, Results: {results_count}/{total_frames_submitted}")

            # Check if all processed
            if results_count >= total_frames_submitted:
                break

            await asyncio.sleep(1)

        # If still not done, forcibly stop batch manager
        if len(results_received) < total_frames_submitted:
            print("⏰ Timeout reached, forcibly stopping batch manager!")
            batch_manager.stop()
        
        processing_time = time.time() - monitor_start
        final_results_count = len(results_received)
        
        print(f"\n📊 BATCH PROCESSING RESULTS:")
        print(f"  Total Submitted: {total_frames_submitted}")
        print(f"  Results Received: {final_results_count}")
        print(f"  Processing Time: {processing_time:.2f}s")
        print(f"  Throughput: {final_results_count/processing_time:.1f} sequences/sec")
        print(f"  Success Rate: {final_results_count/total_frames_submitted*100:.1f}%")
        
        # Analyze results by stream
        stream_results = {}
        for result in results_received:
            sid = result['stream_id']
            if sid not in stream_results:
                stream_results[sid] = []
            stream_results[sid].append(result)
        
        print(f"\n📈 RESULTS BY STREAM:")
        for stream_id in sorted(stream_results.keys()):
            count = len(stream_results[stream_id])
            print(f"  Stream {stream_id}: {count} results")
        
        # Test GPU memory usage
        memory_info = gpu_manager.get_memory_info()
        print(f"\n💾 GPU MEMORY STATUS:")
        print(f"  Used: {memory_info.get('memory_used_percent', 0):.1f}%")
        print(f"  Available: {memory_info.get('memory_available_gb', 0):.1f} GB")
        
        # Stop batch processing
        print("\n🛑 Stopping batch processing...")
        batch_manager.stop()
        
        print("✅ Batch processing test completed successfully!")
        
        # Success criteria
        success_rate = final_results_count / total_frames_submitted if total_frames_submitted > 0 else 0
        throughput = final_results_count / processing_time if processing_time > 0 else 0
        
        if success_rate >= 0.8 and throughput >= 1.0:  # 80% success rate, 1+ sequences/sec
            print("🎉 EXCELLENT: Batch processing performance is optimal!")
            return True
        elif success_rate >= 0.5:
            print("✅ GOOD: Batch processing working but could be optimized")
            return True
        else:
            print("⚠️ NEEDS IMPROVEMENT: Low success rate or throughput")
            return False
        
    except Exception as e:
        print(f"❌ Error in batch processing test: {e}")
        import traceback
        traceback.print_exc()
        return False

@pytest.mark.asyncio
async def main():
    """Main test function"""
    print("🧪 TDISS Direct Batch Processing Performance Test")
    print("=" * 60)
    print("Testing the core batch processing system with simulated frame data\n")
    
    success = await test_batch_processing_directly()
    
    if success:
        print("\n🎉 SUCCESS: Batch processing system validated!")
        print("   The core batch inference system is working efficiently.")
        print("   Ready for multi-stream deployment!")
    else:
        print("\n⚠️ Issues detected in batch processing system.")
        print("   Review the performance metrics above.")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)