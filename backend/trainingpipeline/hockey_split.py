import os
import shutil
import re
import argparse
from pathlib import Path
import random

def split_hockey_dataset(source_dir, output_dir, train_ratio=0.8):
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if output_path.exists():
        shutil.rmtree(output_path)
    
    output_path.mkdir(parents=True)
    (output_path / "train" / "Fight").mkdir(parents=True)
    (output_path / "train" / "NonFight").mkdir(parents=True)
    (output_path / "val" / "Fight").mkdir(parents=True)
    (output_path / "val" / "NonFight").mkdir(parents=True)
    
    all_files = [f for f in source_path.iterdir() if f.is_file() and f.suffix.lower() in ['.avi', '.mp4', '.mov']]
    
    fight_files = [f for f in all_files if re.match(r'^fi', f.name.lower())]
    nonfight_files = [f for f in all_files if re.match(r'^no', f.name.lower())]
    
    random.shuffle(fight_files)
    random.shuffle(nonfight_files)
    
    fight_train_count = int(len(fight_files) * train_ratio)
    nonfight_train_count = int(len(nonfight_files) * train_ratio)
    
    for i, file in enumerate(fight_files):
        if i < fight_train_count:
            dest = output_path / "train" / "Fight" / file.name
        else:
            dest = output_path / "val" / "Fight" / file.name
        shutil.copy2(file, dest)
    
    for i, file in enumerate(nonfight_files):
        if i < nonfight_train_count:
            dest = output_path / "train" / "NonFight" / file.name
        else:
            dest = output_path / "val" / "NonFight" / file.name
        shutil.copy2(file, dest)
    
    print(f"Fight videos: {len(fight_files)} ({fight_train_count} train, {len(fight_files)-fight_train_count} val)")
    print(f"NonFight videos: {len(nonfight_files)} ({nonfight_train_count} train, {len(nonfight_files)-nonfight_train_count} val)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input directory path")
    parser.add_argument("--output", required=True, help="Output directory path")
    args = parser.parse_args()
    
    split_hockey_dataset(args.input, args.output)