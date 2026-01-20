"""
Quick FPS Analysis Script
Tests if low FPS is real or due to frame padding

Usage:
    python fps_test.py
"""

import cv2
import numpy as np
from pathlib import Path

def analyze_video_fps(video_path):
    """Analyze true vs reported FPS of a video"""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return None
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / reported_fps if reported_fps > 0 else 0
    
    print(f"\nAnalyzing: {video_path.name}")
    print(f"Reported FPS: {reported_fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")
    
    # Read all frames and detect identical ones
    frames = []
    frame_differences = []
    identical_count = 0
    
    prev_frame = None
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale for comparison
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is not None:
            # Calculate difference between consecutive frames
            diff = np.mean(np.abs(gray.astype(float) - prev_frame.astype(float)))
            frame_differences.append(diff)
            
            # If difference is very small, frames are nearly identical
            if diff < 1.0:  # Very low threshold for identical frames
                identical_count += 1
                print(f"  Frame {frame_idx} identical to previous (diff: {diff:.3f})")
        
        prev_frame = gray.copy()
        frame_idx += 1
        
        # Only process first 100 frames to save time
        if frame_idx > 100:
            break
    
    cap.release()
    
    if len(frame_differences) > 0:
        repetition_ratio = identical_count / len(frame_differences)
        unique_frames = len(frame_differences) - identical_count + 1
        effective_fps = unique_frames / duration if duration > 0 else 0
        avg_frame_diff = np.mean(frame_differences)
        
        print(f"Consecutive identical frames: {identical_count}/{len(frame_differences)}")
        print(f"Repetition ratio: {repetition_ratio:.3f}")
        print(f"Effective FPS: {effective_fps:.2f}")
        print(f"Average frame difference: {avg_frame_diff:.2f}")
        
        # Diagnosis
        if repetition_ratio > 0.8:
            print("🔴 DIAGNOSIS: Heavy frame padding detected")
        elif repetition_ratio > 0.5:
            print("🟡 DIAGNOSIS: Moderate frame repetition")
        elif effective_fps < 10:
            print("🔵 DIAGNOSIS: Genuinely low FPS video")
        else:
            print("🟢 DIAGNOSIS: Normal video")
        
        return {
            'video_name': video_path.name,
            'reported_fps': reported_fps,
            'effective_fps': effective_fps,
            'repetition_ratio': repetition_ratio,
            'avg_frame_diff': avg_frame_diff,
            'total_frames_analyzed': len(frame_differences) + 1
        }
    
    return None

def main():
    # Low FPS videos from your analysis (most problematic ones)
    low_fps_videos = [
        "D:/Thaman/archive/RWF-2000/val/Fight/1Kbw1bUw_3.avi",      # 0.95 FPS
        "D:/Thaman/archive/RWF-2000/val/Fight/6Rl7q_kXYbg_4.avi",   # 2.38 FPS  
        "D:/Thaman/archive/RWF-2000/val/NonFight/7gLKFV5voOg_0.avi", # 4.29 FPS
        "D:/Thaman/archive/RWF-2000/val/NonFight/IMA4zYs83Lo_0.avi", # 7.14 FPS
        "D:/Thaman/archive/RWF-2000/val/NonFight/Fds7C6sp_0.avi",    # 9.05 FPS
    ]
    
    print("="*60)
    print("VIDEO FPS ANALYSIS - TESTING FOR REAL vs PADDED LOW FPS")
    print("="*60)
    
    results = []
    
    for video_path_str in low_fps_videos:
        video_path = Path(video_path_str)
        
        if video_path.exists():
            result = analyze_video_fps(video_path)
            if result:
                results.append(result)
        else:
            print(f"\n❌ Video not found: {video_path}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for result in results:
        print(f"{result['video_name']:25} | "
              f"Reported: {result['reported_fps']:5.1f} | "
              f"Effective: {result['effective_fps']:5.1f} | "
              f"Repeat: {result['repetition_ratio']:4.1%}")
    
    print(f"\nAnalyzed {len(results)} videos")
    
    if results:
        avg_repetition = np.mean([r['repetition_ratio'] for r in results])
        print(f"Average repetition ratio: {avg_repetition:.1%}")
        
        if avg_repetition > 0.7:
            print("\n🔴 CONCLUSION: Videos appear to be heavily padded")
            print("   - Likely dataset processing artifact")
            print("   - Consider frame deduplication preprocessing")
        elif avg_repetition > 0.3:
            print("\n🟡 CONCLUSION: Mixed - some real low FPS, some padding") 
            print("   - Combination of low FPS cameras + some padding")
            print("   - Consider adaptive processing based on repetition ratio")
        else:
            print("\n🔵 CONCLUSION: Genuinely low FPS surveillance videos")
            print("   - Real world surveillance camera limitations")
            print("   - Focus on sparse temporal processing methods")

if __name__ == "__main__":
    main()