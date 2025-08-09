"""
X3D Violence Detection Training Script

This script trains an X3D model with motion enhancement for real-time violence detection.
Optimized for RTX 5090 with CUDA 12.8 and 24GB VRAM.

Usage:
    python train_x3d_violence.py --dataset_path /path/to/RWF-2000 --batch_size 16 --num_epochs 50
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import random
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from x3d_dataset import create_dataloaders
from x3d_model import create_model, FocalLoss
from x3d_trainer import X3DTrainer, create_optimizer_and_scheduler


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train X3D model for violence detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="C:\\archive\\RWF-2000",
        help="Path to RWF-2000 dataset"
    )
    parser.add_argument(
        "--max_videos_per_class", 
        type=int, 
        default=None,
        help="Maximum videos per class (for testing with smaller dataset)"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="x3d_s",
        choices=["x3d_xs", "x3d_s", "x3d_m", "x3d_l"],
        help="X3D model variant"
    )
    parser.add_argument(
        "--use_motion_enhancement", 
        action="store_true", 
        default=True,
        help="Use motion enhancement with optical flow"
    )
    parser.add_argument(
        "--clip_len", 
        type=int, 
        default=16,
        help="Number of frames per clip"
    )
    parser.add_argument(
        "--spatial_size", 
        type=int, 
        default=224,
        help="Spatial resolution of frames"
    )
    
    # Training arguments
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=12,
        help="Batch size (optimized for RTX 5090 24GB)"
    )
    parser.add_argument(
        "--num_epochs", 
        type=int, 
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=1e-4,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=1e-4,
        help="Weight decay for regularization"
    )
    parser.add_argument(
        "--scheduler", 
        type=str, 
        default="cosine",
        choices=["cosine", "step", "plateau", "none"],
        help="Learning rate scheduler"
    )
    parser.add_argument(
        "--patience", 
        type=int, 
        default=10,
        help="Early stopping patience"
    )
    
    # Loss function arguments
    parser.add_argument(
        "--loss_type", 
        type=str, 
        default="focal",
        choices=["cross_entropy", "focal"],
        help="Loss function type"
    )
    parser.add_argument(
        "--focal_alpha", 
        type=float, 
        default=0.25,
        help="Focal loss alpha parameter"
    )
    parser.add_argument(
        "--focal_gamma", 
        type=float, 
        default=2.0,
        help="Focal loss gamma parameter"
    )
    parser.add_argument(
        "--label_smoothing", 
        type=float, 
        default=0.1,
        help="Label smoothing factor"
    )
    
    # System arguments
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=8,
        help="Number of data loader workers"
    )
    parser.add_argument(
        "--mixed_precision", 
        action="store_true", 
        default=True,
        help="Use mixed precision training"
    )
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        default="checkpoints",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--resume_from", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()


def print_system_info():
    """Print system and environment information"""
    print("="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    
    # PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # CUDA info
    if torch.cuda.is_available():
        print(f"CUDA available: Yes")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Current GPU memory
        if torch.cuda.device_count() > 0:
            current_memory = torch.cuda.memory_allocated(0) / 1e9
            cached_memory = torch.cuda.memory_reserved(0) / 1e9
            print(f"Current GPU memory: {current_memory:.2f} GB allocated, {cached_memory:.2f} GB cached")
    else:
        print("CUDA available: No")
    
    # Memory info
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"System RAM: {memory.total / 1e9:.1f} GB total, {memory.available / 1e9:.1f} GB available")
    except ImportError:
        print("System RAM: psutil not available")
    
    print("="*60)


def validate_args(args):
    """Validate command line arguments"""
    # Check dataset path
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    train_dir = dataset_path / "train"
    val_dir = dataset_path / "val"
    
    if not train_dir.exists():
        raise ValueError(f"Training directory does not exist: {train_dir}")
    if not val_dir.exists():
        raise ValueError(f"Validation directory does not exist: {val_dir}")
    
    # Check for required subdirectories
    for split in ["train", "val"]:
        split_dir = dataset_path / split
        fight_dir = split_dir / "Fight"
        nonfight_dir = split_dir / "NonFight"
        
        if not fight_dir.exists():
            raise ValueError(f"Fight directory does not exist: {fight_dir}")
        if not nonfight_dir.exists():
            raise ValueError(f"NonFight directory does not exist: {nonfight_dir}")
        
        # Count videos
        fight_videos = len(list(fight_dir.glob("*.avi")))
        nonfight_videos = len(list(nonfight_dir.glob("*.avi")))
        
        print(f"{split.capitalize()} split: {fight_videos} fight videos, {nonfight_videos} non-fight videos")
        
        if fight_videos == 0 or nonfight_videos == 0:
            raise ValueError(f"No videos found in {split} split")
    
    # Validate batch size for GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if args.batch_size > 16 and gpu_memory < 20:
            print(f"WARNING: Batch size {args.batch_size} might be too large for {gpu_memory:.1f}GB GPU")
            print("Consider reducing batch size if you encounter out-of-memory errors")
    
    print("Arguments validated successfully!")


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Print system info
    print_system_info()
    
    # Validate arguments
    validate_args(args)
    
    # Print training configuration
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    for key, value in vars(args).items():
        print(f"{key:25}: {value}")
    print("="*60)
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    if device == "cuda":
        # Enable optimizations for RTX 5090
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("Enabled CUDA optimizations for RTX 5090")
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, train_dataset, val_dataset = create_dataloaders(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        clip_len=args.clip_len,
        spatial_size=args.spatial_size,
        max_videos_per_class=args.max_videos_per_class
    )
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        model_name=args.model_name,
        num_classes=2,
        use_motion_enhancement=args.use_motion_enhancement,
        device=device
    )
    
    # Create loss function
    print("\nCreating loss function...")
    if args.loss_type == "focal":
        criterion = FocalLoss(
            alpha=args.focal_alpha,
            gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing
        )
        print(f"Using Focal Loss (alpha={args.focal_alpha}, gamma={args.focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        print(f"Using Cross Entropy Loss (label_smoothing={args.label_smoothing})")
    
    # Create optimizer and scheduler
    print("\nCreating optimizer and scheduler...")
    optimizer, scheduler = create_optimizer_and_scheduler(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler if args.scheduler != "none" else None,
        num_epochs=args.num_epochs
    )
    
    # Create trainer
    print("\nCreating trainer...")
    trainer = X3DTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        mixed_precision=args.mixed_precision,
        checkpoint_dir=args.checkpoint_dir,
        patience=args.patience
    )
    
    # Start training
    print("\nStarting training...")
    try:
        history = trainer.train(
            num_epochs=args.num_epochs,
            resume_from=args.resume_from
        )
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Best validation accuracy: {max(history['val_acc']):.2f}%")
        print(f"Best validation F1 score: {max(history['val_f1']):.2f}%")
        print(f"Checkpoints saved to: {args.checkpoint_dir}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
        print("Checkpoints have been saved.")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        print("Check the logs and try again.")
        raise
    
    finally:
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Cleared GPU memory cache")


if __name__ == "__main__":
    main()