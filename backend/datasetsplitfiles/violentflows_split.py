import os
import shutil
import random
import argparse
from pathlib import Path

def reorganize_violentflows(source_dir, target_dir, train_split=0.8, seed=42):
    """
    Reorganize ViolentFlows dataset to match RWF-2000 structure.
    
    Args:
        source_dir: Path to ViolentFlows dataset (e.g., "D:/Thaman/archive/ViolentFlows")
        target_dir: Where to create the new structure (e.g., "D:/Thaman/archive/ViolentFlows_RWF_Format")
        train_split: Fraction of data for training (default 0.8)
        seed: Random seed for reproducible splits
    """
    
    random.seed(seed)
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create target directory structure
    train_fight_dir = target_path / "train" / "Fight"
    train_nonfight_dir = target_path / "train" / "NonFight"
    val_fight_dir = target_path / "val" / "Fight"
    val_nonfight_dir = target_path / "val" / "NonFight"
    
    # Create all directories
    for dir_path in [train_fight_dir, train_nonfight_dir, val_fight_dir, val_nonfight_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all violent and non-violent videos
    violent_videos = []
    nonviolent_videos = []
    
    print("Collecting videos from ViolentFlows dataset...")
    
    # Scan through movies/1, movies/2, movies/3, movies/4, movies/5
    movies_dir = source_path / "movies"
    for i in range(1, 6):
        folder_num = str(i)
        violence_dir = movies_dir / folder_num / "Violence"
        nonviolence_dir = movies_dir / folder_num / "NonViolence"
        
        # Collect violent videos
        if violence_dir.exists():
            for video_file in violence_dir.glob("*"):
                if video_file.is_file() and video_file.suffix.lower() in ['.avi', '.mp4', '.mov', '.mkv']:
                    violent_videos.append(video_file)
        
        # Collect non-violent videos
        if nonviolence_dir.exists():
            for video_file in nonviolence_dir.glob("*"):
                if video_file.is_file() and video_file.suffix.lower() in ['.avi', '.mp4', '.mov', '.mkv']:
                    nonviolent_videos.append(video_file)
    
    print(f"Found {len(violent_videos)} violent videos")
    print(f"Found {len(nonviolent_videos)} non-violent videos")
    
    # Shuffle and split
    random.shuffle(violent_videos)
    random.shuffle(nonviolent_videos)
    
    # Calculate split indices
    violent_train_count = int(len(violent_videos) * train_split)
    nonviolent_train_count = int(len(nonviolent_videos) * train_split)
    
    # Split violent videos
    violent_train = violent_videos[:violent_train_count]
    violent_val = violent_videos[violent_train_count:]
    
    # Split non-violent videos
    nonviolent_train = nonviolent_videos[:nonviolent_train_count]
    nonviolent_val = nonviolent_videos[nonviolent_train_count:]
    
    print(f"\nSplit breakdown:")
    print(f"Training: {len(violent_train)} violent, {len(nonviolent_train)} non-violent")
    print(f"Validation: {len(violent_val)} violent, {len(nonviolent_val)} non-violent")
    
    # Copy files to new structure
    print(f"\nCopying files to {target_dir}...")
    
    # Copy training violent videos
    for i, video in enumerate(violent_train):
        new_name = f"fight_{i+1:03d}{video.suffix}"
        shutil.copy2(video, train_fight_dir / new_name)
    
    # Copy training non-violent videos
    for i, video in enumerate(nonviolent_train):
        new_name = f"nonfight_{i+1:03d}{video.suffix}"
        shutil.copy2(video, train_nonfight_dir / new_name)
    
    # Copy validation violent videos
    for i, video in enumerate(violent_val):
        new_name = f"fight_{i+1:03d}{video.suffix}"
        shutil.copy2(video, val_fight_dir / new_name)
    
    # Copy validation non-violent videos
    for i, video in enumerate(nonviolent_val):
        new_name = f"nonfight_{i+1:03d}{video.suffix}"
        shutil.copy2(video, val_nonfight_dir / new_name)
    
    print("Dataset reorganization completed!")
    print(f"\nNew structure created at: {target_dir}")
    print("├── train/")
    print("│   ├── Fight/")
    print("│   └── NonFight/")
    print("└── val/")
    print("    ├── Fight/")
    print("    └── NonFight/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reorganize ViolentFlows dataset to match RWF-2000 structure"
    )
    
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to ViolentFlows dataset (e.g., D:/Thaman/archive/ViolentFlows)"
    )
    
    parser.add_argument(
        "output_dir", 
        type=str,
        help="Output directory for reorganized dataset (e.g., D:/Thaman/archive/ViolentFlows_RWF_Format)"
    )
    
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Fraction of data for training (default: 0.8)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Run reorganization
    reorganize_violentflows(
        source_dir=args.input_dir,
        target_dir=args.output_dir,
        train_split=args.train_split,
        seed=args.seed
    )
    
    print("\nYou can now use this dataset with your existing RWF-2000 training pipeline!")
    print(f"Update your dataset_path to: {args.output_dir}")