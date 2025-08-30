"""
Split RLVS dataset into RWF-2000 structure with comprehensive debugging
"""

import os
import shutil
import random
import cv2
from pathlib import Path
import argparse
from typing import List, Tuple, Dict
from collections import defaultdict
import time


def analyze_file(file_path: Path) -> Dict:
    """Analyze a single file"""
    info = {
        'path': file_path,
        'name': file_path.name,
        'extension': file_path.suffix.lower(),
        'size': file_path.stat().st_size if file_path.exists() else 0,
        'is_video': False,
        'can_open': False,
        'frame_count': 0,
        'duration': 0,
        'resolution': None,
        'error': None
    }
    
    # Check if it's a video file
    video_extensions = {'.avi', '.mp4', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}
    info['is_video'] = info['extension'] in video_extensions
    
    if info['is_video']:
        try:
            cap = cv2.VideoCapture(str(file_path))
            if cap.isOpened():
                info['can_open'] = True
                info['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps > 0:
                    info['duration'] = info['frame_count'] / fps
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                info['resolution'] = (width, height)
            else:
                info['error'] = "Cannot open with OpenCV"
            cap.release()
        except Exception as e:
            info['error'] = str(e)
    
    return info


def analyze_directory(directory: Path, max_files: int = None) -> Dict:
    """Comprehensively analyze a directory"""
    print(f"\nAnalyzing directory: {directory}")
    
    if not directory.exists():
        print(f"ERROR: Directory doesn't exist: {directory}")
        return {}
    
    # Get all files
    all_files = [f for f in directory.iterdir() if f.is_file()]
    subdirs = [d for d in directory.iterdir() if d.is_dir()]
    
    print(f"Found {len(all_files)} files and {len(subdirs)} subdirectories")
    if subdirs:
        print(f"Subdirectories: {[d.name for d in subdirs]}")
    
    # Analyze files (limit for performance)
    files_to_analyze = all_files[:max_files] if max_files else all_files
    if max_files and len(all_files) > max_files:
        print(f"Analyzing first {max_files} files out of {len(all_files)}")
    
    file_info = []
    extensions = defaultdict(int)
    video_files = []
    valid_videos = []
    
    for i, file_path in enumerate(files_to_analyze):
        if i % 100 == 0 and i > 0:
            print(f"  Analyzed {i}/{len(files_to_analyze)} files...")
        
        info = analyze_file(file_path)
        file_info.append(info)
        extensions[info['extension']] += 1
        
        if info['is_video']:
            video_files.append(info)
            if info['can_open'] and info['frame_count'] > 0:
                valid_videos.append(info)
    
    # Summary statistics
    total_size = sum(info['size'] for info in file_info)
    video_count = len(video_files)
    valid_video_count = len(valid_videos)
    
    analysis = {
        'directory': directory,
        'total_files': len(all_files),
        'analyzed_files': len(files_to_analyze),
        'total_size_mb': total_size / (1024 * 1024),
        'extensions': dict(extensions),
        'video_files': video_count,
        'valid_videos': valid_video_count,
        'file_details': file_info,
        'video_details': valid_videos
    }
    
    # Print summary
    print(f"Summary:")
    print(f"  Total files: {len(all_files)}")
    print(f"  Total size: {total_size / (1024 * 1024):.1f} MB")
    print(f"  Extensions: {dict(extensions)}")
    print(f"  Video files: {video_count}")
    print(f"  Valid videos (can open + has frames): {valid_video_count}")
    
    if video_files and valid_video_count < video_count:
        invalid_videos = [v for v in video_files if not (v['can_open'] and v['frame_count'] > 0)]
        print(f"  Invalid videos: {len(invalid_videos)}")
        for vid in invalid_videos[:5]:  # Show first 5
            print(f"    {vid['name']}: {vid['error'] or 'No frames'}")
        if len(invalid_videos) > 5:
            print(f"    ... and {len(invalid_videos) - 5} more")
    
    return analysis


def detect_rlvs_structure(rlvs_path: Path) -> Tuple[List[Path], List[Path]]:
    """Detect RLVS structure and find violence/non-violence videos"""
    print(f"\n{'='*60}")
    print(f"ANALYZING RLVS DATASET STRUCTURE")
    print(f"{'='*60}")
    print(f"Dataset path: {rlvs_path}")
    
    if not rlvs_path.exists():
        print(f"ERROR: RLVS path doesn't exist: {rlvs_path}")
        return [], []
    
    # Get top-level structure
    subdirs = [d for d in rlvs_path.iterdir() if d.is_dir()]
    top_files = [f for f in rlvs_path.iterdir() if f.is_file()]
    
    print(f"\nTop-level structure:")
    print(f"  Subdirectories ({len(subdirs)}): {[d.name for d in subdirs]}")
    print(f"  Files ({len(top_files)}): {len(top_files)} files")
    
    violence_videos = []
    nonviolence_videos = []
    
    # Pattern matching for directories
    violence_patterns = ['violence', 'violent', 'fight', 'fighting', 'aggressive']
    nonviolence_patterns = ['nonviolence', 'non-violence', 'normal', 'non-violent', 'nonviolent']
    
    violence_dirs = []
    nonviolence_dirs = []
    
    # Analyze each subdirectory
    for subdir in subdirs:
        dir_name_lower = subdir.name.lower()
        
        # Analyze directory contents
        analysis = analyze_directory(subdir, max_files=50)  # Sample first 50 files
        
        # Classify directory - check NON-VIOLENCE first to avoid substring conflicts
        is_nonviolence = any(pattern in dir_name_lower for pattern in nonviolence_patterns)
        is_violence = any(pattern in dir_name_lower for pattern in violence_patterns) and not is_nonviolence
        
        if is_nonviolence:
            nonviolence_dirs.append(subdir)
            print(f"  -> NON-VIOLENCE directory: {subdir.name}")
        elif is_violence:
            violence_dirs.append(subdir)
            print(f"  -> VIOLENCE directory: {subdir.name}")
        else:
            print(f"  -> UNCLEAR directory: {subdir.name}")
    
    # Collect videos from classified directories
    print(f"\nCollecting videos from classified directories...")
    
    for violence_dir in violence_dirs:
        print(f"\nProcessing violence directory: {violence_dir.name}")
        analysis = analyze_directory(violence_dir)
        dir_videos = [Path(info['path']) for info in analysis['video_details']]
        violence_videos.extend(dir_videos)
        print(f"  Added {len(dir_videos)} valid videos")
    
    for nonviolence_dir in nonviolence_dirs:
        print(f"\nProcessing non-violence directory: {nonviolence_dir.name}")
        analysis = analyze_directory(nonviolence_dir)
        dir_videos = [Path(info['path']) for info in analysis['video_details']]
        nonviolence_videos.extend(dir_videos)
        print(f"  Added {len(dir_videos)} valid videos")
    
    # If no clear structure, try flat approach
    if not violence_dirs and not nonviolence_dirs:
        print(f"\nNo clear directory structure found. Analyzing all files...")
        analysis = analyze_directory(rlvs_path, max_files=200)
        
        # Try filename-based classification
        for video_info in analysis['video_details']:
            filename_lower = video_info['name'].lower()
            
            if any(pattern in filename_lower for pattern in violence_patterns):
                violence_videos.append(Path(video_info['path']))
            elif any(pattern in filename_lower for pattern in nonviolence_patterns):
                nonviolence_videos.append(Path(video_info['path']))
    
    print(f"\n{'='*60}")
    print(f"CLASSIFICATION RESULTS")
    print(f"{'='*60}")
    print(f"Violence videos found: {len(violence_videos)}")
    print(f"Non-violence videos found: {len(nonviolence_videos)}")
    print(f"Total videos: {len(violence_videos) + len(nonviolence_videos)}")
    
    if len(violence_videos) == 0 or len(nonviolence_videos) == 0:
        print(f"\nERROR: Unbalanced or missing classes!")
        print(f"Violence: {len(violence_videos)}, Non-violence: {len(nonviolence_videos)}")
        
        # Show some file examples to help debugging
        print(f"\nSample violence directory contents:")
        for vid in violence_videos[:5]:
            print(f"  {vid.name}")
        
        print(f"\nSample non-violence directory contents:")
        for vid in nonviolence_videos[:5]:
            print(f"  {vid.name}")
        
        return [], []
    
    return violence_videos, nonviolence_videos


def split_videos(violence_videos: List[Path], nonviolence_videos: List[Path], 
                train_ratio: float = 0.8) -> Tuple[List[Path], List[Path], List[Path], List[Path]]:
    """Split videos with detailed reporting"""
    print(f"\n{'='*60}")
    print(f"SPLITTING VIDEOS INTO TRAIN/VAL")
    print(f"{'='*60}")
    
    # Shuffle videos
    random.shuffle(violence_videos)
    random.shuffle(nonviolence_videos)
    
    # Calculate split points
    violence_split = int(len(violence_videos) * train_ratio)
    nonviolence_split = int(len(nonviolence_videos) * train_ratio)
    
    # Split violence videos
    train_violence = violence_videos[:violence_split]
    val_violence = violence_videos[violence_split:]
    
    # Split non-violence videos
    train_nonviolence = nonviolence_videos[:nonviolence_split]
    val_nonviolence = nonviolence_videos[nonviolence_split:]
    
    print(f"Split configuration:")
    print(f"  Train ratio: {train_ratio}")
    print(f"  Random seed: {random.getstate()[1][0]}")  # Get first element of random state
    
    print(f"\nSplit results:")
    print(f"  TRAIN:")
    print(f"    Violence: {len(train_violence)}")
    print(f"    Non-violence: {len(train_nonviolence)}")
    print(f"    Total: {len(train_violence) + len(train_nonviolence)}")
    
    print(f"  VALIDATION:")
    print(f"    Violence: {len(val_violence)}")
    print(f"    Non-violence: {len(val_nonviolence)}")
    print(f"    Total: {len(val_violence) + len(val_nonviolence)}")
    
    print(f"  GRAND TOTAL: {len(violence_videos) + len(nonviolence_videos)}")
    
    return train_violence, val_violence, train_nonviolence, val_nonviolence


def create_rwf_structure(output_path: Path, train_violence: List[Path], val_violence: List[Path],
                        train_nonviolence: List[Path], val_nonviolence: List[Path],
                        copy_files: bool = True):
    """Create RWF-2000 structure with detailed progress tracking"""
    print(f"\n{'='*60}")
    print(f"CREATING RWF-2000 DIRECTORY STRUCTURE")
    print(f"{'='*60}")
    
    # Create directory structure
    directories = [
        output_path / "train" / "Fight",
        output_path / "train" / "NonFight", 
        output_path / "val" / "Fight",
        output_path / "val" / "NonFight"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # File operations with detailed tracking
    file_operations = [
        (train_violence, output_path / "train" / "Fight", "Train Violence"),
        (train_nonviolence, output_path / "train" / "NonFight", "Train Non-Violence"),
        (val_violence, output_path / "val" / "Fight", "Val Violence"),
        (val_nonviolence, output_path / "val" / "NonFight", "Val Non-Violence")
    ]
    
    total_files = sum(len(files) for files, _, _ in file_operations)
    processed = 0
    failed = 0
    
    print(f"\nFile operations:")
    print(f"  Operation: {'Copy' if copy_files else 'Move'}")
    print(f"  Total files to process: {total_files}")
    
    start_time = time.time()
    
    for video_list, dest_dir, description in file_operations:
        print(f"\nProcessing {description}: {len(video_list)} files")
        operation_start = time.time()
        operation_failed = 0
        
        for i, video_path in enumerate(video_list):
            dest_path = dest_dir / video_path.name
            
            try:
                if copy_files:
                    shutil.copy2(video_path, dest_path)
                else:
                    shutil.move(str(video_path), str(dest_path))
                
                processed += 1
                
                # Progress reporting
                if processed % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed
                    eta = (total_files - processed) / rate if rate > 0 else 0
                    print(f"  Progress: {processed}/{total_files} ({processed/total_files*100:.1f}%) "
                          f"Rate: {rate:.1f} files/sec, ETA: {eta:.0f}s")
                    
            except Exception as e:
                print(f"ERROR processing {video_path.name}: {e}")
                failed += 1
                operation_failed += 1
        
        operation_time = time.time() - operation_start
        success_count = len(video_list) - operation_failed
        print(f"  Completed: {success_count}/{len(video_list)} files in {operation_time:.1f}s")
        if operation_failed > 0:
            print(f"  Failed: {operation_failed} files")
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"FILE OPERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Successfully processed: {processed}/{total_files}")
    print(f"Failed: {failed}")
    print(f"Average rate: {processed/total_time:.1f} files/second")
    
    # Verify final structure
    print(f"\nVerifying final structure:")
    for directory in directories:
        file_count = len(list(directory.glob("*")))
        print(f"  {directory.relative_to(output_path)}: {file_count} files")


def main():
    parser = argparse.ArgumentParser(description="Split RLVS dataset with comprehensive analysis")
    
    parser.add_argument("--rlvs_path", type=str, required=True,
                       help="Path to RLVS dataset directory")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Output directory for RWF-2000 structure")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="Ratio of data for training (default: 0.8)")
    parser.add_argument("--copy", action="store_true", default=False,
                       help="Copy files instead of moving them")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible splits")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Validate paths
    rlvs_path = Path(args.rlvs_path)
    output_path = Path(args.output_path)
    
    if not rlvs_path.exists():
        print(f"ERROR: RLVS path does not exist: {rlvs_path}")
        return
    
    print("="*60)
    print("RLVS TO RWF-2000 STRUCTURE CONVERTER WITH ANALYSIS")
    print("="*60)
    print(f"RLVS path: {rlvs_path}")
    print(f"Output path: {output_path}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Operation: {'Copy' if args.copy else 'Move'}")
    print(f"Random seed: {args.seed}")
    print("="*60)
    
    try:
        # Step 1: Detect RLVS structure and classify videos
        violence_videos, nonviolence_videos = detect_rlvs_structure(rlvs_path)
        
        if not violence_videos or not nonviolence_videos:
            print("ABORTING: Could not properly classify videos")
            return
        
        # Step 2: Split into train/val
        train_violence, val_violence, train_nonviolence, val_nonviolence = split_videos(
            violence_videos, nonviolence_videos, args.train_ratio
        )
        
        # Step 3: Create RWF-2000 structure
        create_rwf_structure(output_path, train_violence, val_violence, 
                           train_nonviolence, val_nonviolence, args.copy)
        
        print(f"\n{'='*60}")
        print(f"CONVERSION COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Dataset ready at: {output_path}")
        print(f"Final structure verification:")
        
        # Final verification
        for split in ["train", "val"]:
            for class_name in ["Fight", "NonFight"]:
                folder = output_path / split / class_name
                if folder.exists():
                    file_count = len(list(folder.glob("*")))
                    print(f"  {split}/{class_name}: {file_count} files")
        
        print(f"\nYou can now use this dataset with your X3D training script!")
        print(f"Command example:")
        print(f'python train_x3d_violence.py --dataset_path "{output_path}"')
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: Conversion failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()