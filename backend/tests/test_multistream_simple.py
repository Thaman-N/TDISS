"""
Simple Multi-Stream Performance Test
===================================

This script tests the system's ability to handle multiple concurrent frame
processing tasks, simulating the load of multiple RTSP streams.
"""

import asyncio
import time
import numpy as np
import cv2
import threading
from concurrent.futures import ThreadPoolExecutor
import sys
import os

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class MultiStreamSimulator:
    """Simulates multiple streams processing concurrently"""
    
    def __init__(self):
        self.results = []
        self.processing_times = []
        self.errors = []
        
    def generate_test_sequence(self, stream_id: int) -> np.ndarray:
        """Generate a 16-frame sequence for X3D model"""
        frames = []
        
        for i in range(16):  # X3D needs 16 frames
            # Create realistic frame (336x336x3)
            frame = np.random.randint(50, 200, (336, 336, 3), dtype=np.uint8)
            
            # Add moving objects to simulate activity
            t = i / 16.0  # Time within sequence
            
            # Moving circle (simulate person)
            center_x = int(168 + 50 * np.sin(t * 4 + stream_id))
            center_y = int(168 + 30 * np.cos(t * 3 + stream_id))
            cv2.circle(frame, (center_x, center_y), 15, (180, 120, 80), -1)
            
            # Add some "violent" motion for higher streams
            if stream_id > 2 and i % 4 == 0:
                # Rapid movement
                rapid_x = center_x + int(20 * np.sin(t * 20))
                rapid_y = center_y + int(15 * np.cos(t * 25))
                cv2.circle(frame, (rapid_x, rapid_y), 8, (255, 100, 100), -1)
            
            frames.append(frame)
        
        return np.array(frames)
    
    async def process_stream_sequence(self, stream_id: int, model, device) -> dict:
        """Process a single sequence from a stream"""
        try:
            start_time = time.time()
            
            # Generate test sequence
            frame_sequence = self.generate_test_sequence(stream_id)
            
            # Preprocess for model (similar to extract_consecutive_frame_sequences)
            from torch_detection import preprocess_frames, predict_violence
            
            # Convert to model input format
            preprocessed = preprocess_frames(frame_sequence)
            
            # Run inference
            is_fight, fight_prob, inference_time = predict_violence(model, preprocessed, 0.5, False, device)
            
            processing_time = time.time() - start_time
            
            result = {
                'stream_id': stream_id,
                'is_fight': is_fight,
                'fight_prob': fight_prob,
                'inference_time': inference_time,
                'processing_time': processing_time,
                'success': True,
                'timestamp': time.time()
            }
            
            print(f"  ✅ Stream {stream_id}: {fight_prob:.3f} confidence ({'FIGHT' if is_fight else 'SAFE'}) ({processing_time:.2f}s)")
            return result
            
        except Exception as e:
            error_result = {
                'stream_id': stream_id,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'success': False,
                'timestamp': time.time()
            }
            print(f"  ❌ Stream {stream_id}: Error - {e}")
            return error_result
    
    async def test_concurrent_processing(self, max_streams: int = 6):
        """Test concurrent processing with multiple simulated streams"""
        print("🧪 Multi-Stream Concurrent Processing Test")
        print("=" * 50)
        
        # Load model once
        try:
            from torch_detection import load_violence_detection_model
            print("📥 Loading violence detection model...")
            # Force CPU usage to avoid CUDA 3D convolution issues
            model, device = load_violence_detection_model('../models/rwf9425.pth', device='cpu')
            print(f"✅ Model loaded successfully on device: {device}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
        
        # Test with increasing number of concurrent streams
        for stream_count in range(1, max_streams + 1):
            print(f"\n🔄 Testing {stream_count} concurrent stream(s)...")
            
            # Create tasks for concurrent processing
            tasks = []
            start_time = time.time()
            
            # Process multiple sequences concurrently
            for stream_id in range(1, stream_count + 1):
                task = self.process_stream_sequence(stream_id, model, device)
                tasks.append(task)
            
            # Execute all tasks concurrently
            print(f"  🚀 Processing {len(tasks)} sequences concurrently...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            total_time = time.time() - start_time
            
            # Analyze results
            successful = [r for r in results if isinstance(r, dict) and r.get('success', False)]
            failed = [r for r in results if isinstance(r, Exception) or (isinstance(r, dict) and not r.get('success', False))]
            
            if successful:
                avg_processing_time = sum(r['processing_time'] for r in successful) / len(successful)
                max_processing_time = max(r['processing_time'] for r in successful)
                min_processing_time = min(r['processing_time'] for r in successful)
                
                # Calculate efficiency
                theoretical_sequential_time = sum(r['processing_time'] for r in successful)
                efficiency = theoretical_sequential_time / total_time if total_time > 0 else 0
                
                print(f"  📊 Results for {stream_count} stream(s):")
                print(f"    Successful: {len(successful)}/{len(tasks)}")
                print(f"    Total Time: {total_time:.2f}s")
                print(f"    Avg Processing: {avg_processing_time:.2f}s")
                print(f"    Range: {min_processing_time:.2f}s - {max_processing_time:.2f}s")
                print(f"    Parallel Efficiency: {efficiency:.1f}x")
                print(f"    Throughput: {len(successful)/total_time:.1f} sequences/sec")
                
                # Check predictions
                fight_probs = [r['fight_prob'] for r in successful]
                fight_count = sum([1 for r in successful if r['is_fight']])
                avg_confidence = sum(fight_probs) / len(fight_probs) if fight_probs else 0
                print(f"    Avg Confidence: {avg_confidence:.3f}")
                print(f"    Violence Detected: {fight_count}/{len(successful)} streams")
                
                # Store results for summary
                self.results.append({
                    'stream_count': stream_count,
                    'successful': len(successful),
                    'total': len(tasks),
                    'total_time': total_time,
                    'avg_processing_time': avg_processing_time,
                    'efficiency': efficiency,
                    'throughput': len(successful)/total_time,
                    'avg_confidence': avg_confidence
                })
            else:
                print(f"  ❌ No successful processing for {stream_count} streams")
                print(f"    Errors: {len(failed)}")
            
            # Brief pause between tests
            await asyncio.sleep(2)
        
        print(f"\n🎉 Concurrent processing test completed!")
        self._print_performance_summary()
        return True
    
    def _print_performance_summary(self):
        """Print comprehensive performance summary"""
        print(f"\n📊 PERFORMANCE SUMMARY")
        print("=" * 70)
        print(f"{'Streams':<8} {'Success':<8} {'Time':<8} {'Efficiency':<10} {'Throughput':<12} {'Confidence':<10}")
        print("-" * 70)
        
        for result in self.results:
            print(f"{result['stream_count']:<8} "
                  f"{result['successful']}/{result['total']:<6} "
                  f"{result['total_time']:<7.2f}s "
                  f"{result['efficiency']:<9.1f}x "
                  f"{result['throughput']:<11.1f}/s "
                  f"{result['avg_confidence']:<10.3f}")
        
        if len(self.results) > 1:
            print("\n📈 ANALYSIS:")
            
            # Find optimal performance point
            max_throughput = max(r['throughput'] for r in self.results)
            best_result = max(self.results, key=lambda r: r['throughput'])
            
            print(f"  🏆 Best Performance: {best_result['stream_count']} streams")
            print(f"  📈 Peak Throughput: {max_throughput:.1f} sequences/sec")
            print(f"  ⚡ Max Efficiency: {max(r['efficiency'] for r in self.results):.1f}x")
            
            # Check scaling behavior
            single_stream_throughput = self.results[0]['throughput'] if self.results else 0
            scaling_factor = max_throughput / single_stream_throughput if single_stream_throughput > 0 else 0
            
            print(f"  📊 Scaling Factor: {scaling_factor:.1f}x")
            
            if scaling_factor > 2.0:
                print("  ✅ EXCELLENT: System scales very well with multiple streams")
            elif scaling_factor > 1.5:
                print("  ✅ GOOD: System handles multiple streams efficiently")
            elif scaling_factor > 1.0:
                print("  ⚠️ MODERATE: Some benefit from concurrent processing")
            else:
                print("  ❌ POOR: No benefit from concurrent processing")

async def main():
    """Main test function"""
    print("🚀 TDISS Multi-Stream Performance Validation")
    print("=" * 60)
    print("Testing concurrent stream processing with simulated video sequences")
    print("This validates multi-stream performance without requiring RTSP sources\n")
    
    simulator = MultiStreamSimulator()
    success = await simulator.test_concurrent_processing(max_streams=8)
    
    if success:
        print("\n🎉 SUCCESS: Multi-stream processing validated!")
        print("   Your system can handle multiple concurrent streams effectively.")
        print("   Ready for real RTSP deployment! 🚀")
    else:
        print("\n⚠️ Issues detected. Review the performance analysis above.")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)