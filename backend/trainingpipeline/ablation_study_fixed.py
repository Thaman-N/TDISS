"""
Fixed Ablation Study for X3D Violence Detection.
Connects the new Configs, Dataset, and Model modifications.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Fix OpenMP duplicate library issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import your modules
from x3d_model import create_model, StableCrossEntropyLoss
from x3d_trainer import OptimizedX3DTrainer, create_optimized_optimizer_and_scheduler
from x3d_dataset import create_cuenet_dataloaders

# Import the configuration dictionary (This replaces the 100+ lines of inline config)
from ablation_config import get_core_ablations

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_ablation_dataloaders(
    dataset_path: str,
    config: Dict,
    batch_size: int = 8,
    num_workers: int = 4,
    max_videos_per_class: int = None
):
    """
    Create dataloaders based on ablation configuration.
    Passes flags directly to the robust x3d_dataset.py.
    """
    
    # Extract flags from config with defaults
    use_yolo = config.get('use_yolo_cropping', False)
    use_randaug = config.get('use_randaugment', False)
    use_content_aware = config.get('use_content_aware_sampling', False)
    spatial_size = config.get('spatial_size', 224)
    
    print(f"  - Data Config: YOLO={use_yolo}, RandAug={use_randaug}, ContentAware={use_content_aware}, Size={spatial_size}")
    
    train_loader, val_loader, train_dataset, val_dataset = create_cuenet_dataloaders(
        dataset_path=dataset_path,
        batch_size=batch_size,
        num_workers=num_workers,
        clip_len=16,
        spatial_size=spatial_size,
        max_videos_per_class=max_videos_per_class,
        # Pass flags directly to the new robust dataset class
        use_cuenet_cropping=use_yolo,
        use_randaugment=use_randaug,
        cache_yolo_detections=use_yolo, # Cache if using YOLO
        use_content_aware_sampling=use_content_aware # NEW: Pass the sampling flag
    )
    
    return train_loader, val_loader

def create_ablation_model(config: Dict, device: str = "cuda"):
    """Create model with ablation-specific settings"""
    
    use_motion = config.get('use_motion_enhancement', False)
    use_kernel_opt = config.get('use_temporal_kernel_optimization', False)
    use_tsa = config.get('use_tsa_block', False)
    
    print(f"  - Model Config: Motion={use_motion}, KernelOpt={use_kernel_opt}, TSA_Block={use_tsa}")
    
    model = create_model(
        model_name="x3d_m",
        num_classes=2,
        use_motion_enhancement=use_motion,
        use_temporal_kernel_optimization=use_kernel_opt, # Pass flag
        use_tsa_block=use_tsa, # NEW: Pass novelty flag
        device=device
    )
    
    return model

def run_single_ablation(
    config_name: str,
    config: Dict,
    args: argparse.Namespace,
    results_dir: Path
) -> Dict:
    """Run a single ablation experiment"""
    
    print(f"\n{'='*80}")
    print(f"RUNNING ABLATION: {config_name.upper()}")
    print(f"{'='*80}")
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*80)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create experiment directory
    exp_dir = results_dir / config_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = exp_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    try:
        # Create dataloaders
        print(f"\n📊 Creating dataloaders for {config_name}...")
        train_loader, val_loader = create_ablation_dataloaders(
            dataset_path=args.dataset_path,
            config=config,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_videos_per_class=args.max_videos_per_class
        )
        
        # Create model
        print(f"\n🏗️ Creating model for {config_name}...")
        model = create_ablation_model(config, device)
        
        # Create loss function
        criterion = StableCrossEntropyLoss(label_smoothing=0.05)
        
        # Create optimizer and scheduler
        optimizer, scheduler = create_optimized_optimizer_and_scheduler(
            model=model,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            scheduler_type=args.scheduler,
            num_epochs=args.num_epochs,
            warmup_epochs=args.warmup_epochs
        )
        
        # Check for Dense Eval flag
        use_dense_eval = config.get('use_dense_eval', False)
        print(f"  - Validation Config: Dense Evaluation = {use_dense_eval}")
        
        # Create trainer
        trainer = OptimizedX3DTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            mixed_precision=args.mixed_precision,
            checkpoint_dir=str(exp_dir / "checkpoints"),
            patience=args.patience,
            gradient_clip_val=args.gradient_clip_val,
            warmup_epochs=args.warmup_epochs,
            use_dense_eval=use_dense_eval # NEW: Pass the eval flag
        )
        
        # Train model
        print(f"\n🚀 Starting training for {config_name}...")
        start_time = time.time()
        
        history = trainer.train(num_epochs=args.num_epochs)
        
        training_time = time.time() - start_time
        
        # Get best metrics
        best_val_acc = max(history['val_acc']) if history['val_acc'] else 0.0
        best_val_f1 = max(history['val_f1']) if history['val_f1'] else 0.0
        best_val_loss = min(history['val_loss']) if history['val_loss'] else float('inf')
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Prepare results
        results = {
            'config_name': config_name,
            'config': config,
            'best_val_accuracy': float(best_val_acc),
            'best_val_f1': float(best_val_f1),
            'best_val_loss': float(best_val_loss),
            'training_time_minutes': float(training_time / 60),
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'final_gradient_norm': float(history['gradient_norms'][-1] if history['gradient_norms'] else 0.0),
            'training_stable': bool(history['gradient_norms'][-1] < 2.0 if history['gradient_norms'] else True)
        }
        
        # Save results
        results_path = exp_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ {config_name} completed successfully!")
        print(f"   Best Val Accuracy: {best_val_acc:.2f}%")
        print(f"   Best Val F1: {best_val_f1:.2f}%")
        
        return results
        
    except Exception as e:
        print(f"\n❌ {config_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'config_name': config_name,
            'config': config,
            'best_val_accuracy': 0.0,
            'error': str(e)
        }
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def plot_ablation_comparison(results: List[Dict], output_dir: Path):
    """Generate comparative visualization of results"""
    if not results: return
    
    df = pd.DataFrame(results)
    df = df[df['best_val_accuracy'] > 0] # Filter failures
    if df.empty: return

    # Sort
    df = df.sort_values('best_val_accuracy', ascending=True)
    
    # 1. Accuracy Plot
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Color bar based on baseline/novelty
    colors = ['gray'] * len(df)
    for i, name in enumerate(df['config_name']):
        if 'proposed' in name: colors[i] = 'green'
        elif 'baseline' in name: colors[i] = 'blue'
        
    ax = sns.barplot(x='best_val_accuracy', y='config_name', data=df, palette=colors)
    
    plt.title('Ablation Study: Validation Accuracy Comparison', fontsize=16)
    plt.xlabel('Accuracy (%)', fontsize=12)
    plt.ylabel('Configuration', fontsize=12)
    
    # Add values
    for i, v in enumerate(df['best_val_accuracy']):
        ax.text(v + 0.1, i, f'{v:.2f}%', va='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(output_dir / 'ablation_accuracy_comparison.png', dpi=300)
    plt.close()
    
    print(f"📊 comparative plot saved to {output_dir / 'ablation_accuracy_comparison.png'}")

def generate_ablation_report(results: List[Dict], output_dir: Path):
    """Generate comprehensive ablation study report"""
    
    # Create results table
    df = pd.DataFrame(results)
    
    if df.empty or 'best_val_accuracy' not in df.columns:
        print("❌ No successful experiments to report!")
        return
    
    # Sort by validation accuracy
    df = df.sort_values('best_val_accuracy', ascending=False)
    
    # Create summary table
    summary_cols = [
        'config_name', 'best_val_accuracy', 'best_val_f1', 
        'training_time_minutes', 'total_parameters'
    ]
    # Handle missing cols gracefully
    existing_cols = [c for c in summary_cols if c in df.columns]
    summary_df = df[existing_cols].copy()
    
    # Calculate improvement
    baseline_rows = df[df['config_name'].astype(str).str.contains('baseline')]
    if not baseline_rows.empty:
        baseline_acc = baseline_rows['best_val_accuracy'].iloc[0]
        summary_df['diff_baseline'] = summary_df['best_val_accuracy'] - baseline_acc
    
    # Save CSVs
    summary_df.to_csv(output_dir / "ablation_summary.csv", index=False)
    
    # Print Report
    print("\n" + "="*80)
    print("FINAL ABLATION RESULTS SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80 + "\n")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run RWF-2000 Ablation Study")
    
    # Dataset arguments
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to RWF-2000 dataset")
    parser.add_argument("--output_dir", type=str, default="ablation_results", help="Output directory")
    parser.add_argument("--max_videos_per_class", type=int, default=None, help="Limit videos (debug)")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=50) # Set to 50 as standard
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--scheduler", type=str, default="cosine")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--mixed_precision", action="store_true", default=True)
    
    # System arguments
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    
    # Ablation control
    parser.add_argument("--skip_configs", type=str, nargs="*", default=[], help="Skip specific configs")
    parser.add_argument("--only_configs", type=str, nargs="*", default=[], help="Run ONLY these configs")
    
    return parser.parse_args()

def main():
    """Main ablation study function"""
    args = parse_args()
    set_seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("X3D VIOLENCE DETECTION - FIXED ABLATION STUDY")
    print("="*80)
    
    # Get ablation configurations
    all_configs = get_core_ablations()
    
    # Filter configurations
    if args.only_configs:
        configs_to_run = {k: v for k, v in all_configs.items() if k in args.only_configs}
        # Partial match support
        if not configs_to_run:
            print(f"⚠️ Exact match not found. Searching...")
            for req in args.only_configs:
                for name, cfg in all_configs.items():
                    if req in name: configs_to_run[name] = cfg
    else:
        configs_to_run = {k: v for k, v in all_configs.items() if k not in args.skip_configs}
    
    if not configs_to_run:
        print("❌ No configurations selected!")
        return

    print(f"Selected {len(configs_to_run)} configurations.")
    
    # Run experiments
    all_results = []
    
    for i, (config_name, config) in enumerate(configs_to_run.items(), 1):
        print(f"\n🔬 EXPERIMENT {i}/{len(configs_to_run)}: {config_name}")
        
        results = run_single_ablation(
            config_name=config_name,
            config=config,
            args=args,
            results_dir=output_dir
        )
        
        all_results.append(results)
        
        # Save intermediate
        with open(output_dir / "intermediate_results.json", 'w') as f:
            # Simple serializer helper
            json.dump(all_results, f, default=str, indent=2)

    # Generate Reports
    generate_ablation_report(all_results, output_dir)
    plot_ablation_comparison(all_results, output_dir)
    
    print(f"\n🎉 ALL EXPERIMENTS COMPLETED!")

if __name__ == "__main__":
    main()