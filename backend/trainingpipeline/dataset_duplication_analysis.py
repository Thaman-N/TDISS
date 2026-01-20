"""
Full Dataset Frame Duplication Analysis
Analyzes every video in the dataset and outputs duplication percentages

Usage:
    python dataset_duplication_analysis.py
"""

import cv2
import numpy as np
from pathlib import Path
import csv
import time

def analyze_video_duplication(video_path):
    """Analyze frame duplication percentage for a single video"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return None
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        reported_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Read frames and detect duplicates
        identical_count = 0
        prev_frame = None
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to grayscale for comparison
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                # Calculate difference between consecutive frames
                diff = np.mean(np.abs(gray.astype(float) - prev_frame.astype(float)))
                
                # If difference is very small, frames are nearly identical
                if diff < 1.0:  # Same threshold as before
                    identical_count += 1
            
            prev_frame = gray.copy()
            frame_count += 1
        
        cap.release()
        
        if frame_count > 1:
            duplication_percentage = (identical_count / (frame_count - 1)) * 100
            effective_fps = ((frame_count - 1 - identical_count) / 5.0) if frame_count > 1 else 0
            
            return {
                'video_name': video_path.name,
                'video_path': str(video_path),
                'class': video_path.parent.name,  # Fight or NonFight
                'split': video_path.parent.parent.name,  # train or val
                'total_frames': frame_count,
                'reported_fps': reported_fps,
                'identical_frames': identical_count,
                'duplication_percentage': duplication_percentage,
                'effective_fps': effective_fps
            }
    
    except Exception as e:
        print(f"Error analyzing {video_path.name}: {e}")
        return {
            'video_name': video_path.name,
            'video_path': str(video_path),
            'class': video_path.parent.name if video_path.parent else 'unknown',
            'split': video_path.parent.parent.name if video_path.parent.parent else 'unknown',
            'error': str(e)
        }
    
    return None

def analyze_full_dataset(dataset_path):
    """Analyze the entire dataset"""
    dataset_root = Path(dataset_path)
    
    if not dataset_root.exists():
        print(f"Dataset path not found: {dataset_root}")
        return
    
    # Find all video files
    video_files = []
    for split in ['train', 'val']:
        for class_name in ['Fight', 'NonFight']:
            class_dir = dataset_root / split / class_name
            if class_dir.exists():
                video_files.extend(list(class_dir.glob('*.avi')))
                video_files.extend(list(class_dir.glob('*.mp4')))
    
    print(f"Found {len(video_files)} videos to analyze")
    print("="*80)
    
    results = []
    processed = 0
    start_time = time.time()
    
    for video_path in video_files:
        result = analyze_video_duplication(video_path)
        if result:
            results.append(result)
        
        processed += 1
        
        # Progress update every 100 videos
        if processed % 100 == 0:
            elapsed = time.time() - start_time
            rate = processed / elapsed
            eta = (len(video_files) - processed) / rate if rate > 0 else 0
            print(f"Processed {processed}/{len(video_files)} videos "
                  f"({processed/len(video_files)*100:.1f}%) - "
                  f"ETA: {eta/60:.1f} minutes")
    
    # Save results to CSV
    output_file = 'vf_duplication_analysis.csv'
    
    fieldnames = ['video_name', 'video_path', 'class', 'split', 'total_frames', 
                  'reported_fps', 'identical_frames', 'duplication_percentage', 
                  'effective_fps', 'error']
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nAnalysis complete! Results saved to: {output_file}")
    
    # Generate summary statistics
    valid_results = [r for r in results if 'error' not in r]
    
    if valid_results:
        duplications = [r['duplication_percentage'] for r in valid_results]
        effective_fps_values = [r['effective_fps'] for r in valid_results]
        
        print("\n" + "="*80)
        print("DATASET SUMMARY")
        print("="*80)
        print(f"Total videos analyzed: {len(valid_results)}")
        print(f"Videos with errors: {len(results) - len(valid_results)}")
        
        print(f"\nFrame Duplication Statistics:")
        print(f"  Average duplication: {np.mean(duplications):.1f}%")
        print(f"  Median duplication: {np.median(duplications):.1f}%")
        print(f"  Min duplication: {np.min(duplications):.1f}%")
        print(f"  Max duplication: {np.max(duplications):.1f}%")
        
        print(f"\nEffective FPS Statistics:")
        print(f"  Average effective FPS: {np.mean(effective_fps_values):.1f}")
        print(f"  Median effective FPS: {np.median(effective_fps_values):.1f}")
        print(f"  Min effective FPS: {np.min(effective_fps_values):.1f}")
        print(f"  Max effective FPS: {np.max(effective_fps_values):.1f}")
        
        # Categorize videos by duplication level
        high_dup = len([r for r in valid_results if r['duplication_percentage'] > 80])
        med_dup = len([r for r in valid_results if 50 <= r['duplication_percentage'] <= 80])
        low_dup = len([r for r in valid_results if r['duplication_percentage'] < 50])
        
        print(f"\nDuplication Categories:")
        print(f"  High duplication (>80%): {high_dup} videos ({high_dup/len(valid_results)*100:.1f}%)")
        print(f"  Medium duplication (50-80%): {med_dup} videos ({med_dup/len(valid_results)*100:.1f}%)")
        print(f"  Low duplication (<50%): {low_dup} videos ({low_dup/len(valid_results)*100:.1f}%)")
        
        # Analysis by class
        fight_results = [r for r in valid_results if r['class'] == 'Fight']
        nonfight_results = [r for r in valid_results if r['class'] == 'NonFight']
        
        if fight_results and nonfight_results:
            fight_dup_avg = np.mean([r['duplication_percentage'] for r in fight_results])
            nonfight_dup_avg = np.mean([r['duplication_percentage'] for r in nonfight_results])
            
            print(f"\nBy Class:")
            print(f"  Fight videos - Avg duplication: {fight_dup_avg:.1f}%")
            print(f"  NonFight videos - Avg duplication: {nonfight_dup_avg:.1f}%")
        
        # Analysis by split
        train_results = [r for r in valid_results if r['split'] == 'train']
        val_results = [r for r in valid_results if r['split'] == 'val']
        
        if train_results and val_results:
            train_dup_avg = np.mean([r['duplication_percentage'] for r in train_results])
            val_dup_avg = np.mean([r['duplication_percentage'] for r in val_results])
            
            print(f"\nBy Split:")
            print(f"  Training videos - Avg duplication: {train_dup_avg:.1f}%")
            print(f"  Validation videos - Avg duplication: {val_dup_avg:.1f}%")

def main():

    import sys
    if len(sys.argv) < 2:
        print("Usage: python dataset_duplication_analysis.py <dataset_path>")
        return
    dataset_path = sys.argv[1]

    print("Full Dataset Frame Duplication Analysis")
    print("="*80)
    print(f"Dataset path: {dataset_path}")
    print()

    analyze_full_dataset(dataset_path)

if __name__ == "__main__":
    main()