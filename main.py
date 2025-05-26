import os
import sys
import argparse
import torch
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Install required packages if not already installed
try:
    import open_clip
except ImportError:
    print("Installing open_clip-torch...")
    os.system("pip install open-clip-torch")
    import open_clip

try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    os.system("pip install ultralytics")
    from ultralytics import YOLO

# Import local modules
from preprocessing import RWF2000Preprocessor, RWF2000Dataset
from model import CUENet
from training import CUENetTrainer
from visualization import ModelVisualizer
from augmentation import get_training_transforms, get_validation_transforms


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='CUE-Net Violence Detection Pipeline')
    
    # General arguments
    parser.add_argument('--dataset_path', type=str, default='D:\\Thaman\\archive\\RWF-2000',
                        help='Path to the RWF-2000 dataset')
    parser.add_argument('--output_path', type=str, default='D:\\Thaman\\archive\\RWF-2000-processed',
                        help='Path to save processed data')
    parser.add_argument('--checkpoint_dir', type=str, default='D:\\Thaman\\models\\cuenet',
                        help='Directory to save model checkpoints')
    parser.add_argument('--visualization_dir', type=str, default='D:\\Thaman\\visualizations',
                        help='Directory to save visualizations')
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['preprocess', 'train', 'evaluate', 'visualize', 'all'],
                        default='all', help='Pipeline mode')
    
    # Preprocessing arguments
    parser.add_argument('--num_frames', type=int, default=64,
                        help='Number of frames to extract from each video')
    parser.add_argument('--spatial_size', type=int, default=336,  # Paper uses 336x336
                        help='Spatial size of frames')
    parser.add_argument('--use_spatial_cropping', action='store_true', default=True,
                        help='Use spatial cropping with YOLO V8')
    parser.add_argument('--visualize_samples', type=int, default=5,
                        help='Number of samples to visualize during preprocessing')
    parser.add_argument('--splits', type=str, default='train,val',
                        help='Comma-separated list of splits to process')
    parser.add_argument('--test_mode', type=int, default=None,
                    help='If specified, process only this many videos per class (for testing)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,  # Paper uses 50
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-5,  # Paper uses 1e-5
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # Model arguments
    parser.add_argument('--embed_dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--depth', type=int, default=4,
                        help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Evaluation/visualization arguments
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to model checkpoint for evaluation/visualization')
    parser.add_argument('--num_viz_samples', type=int, default=5,
                        help='Number of samples to visualize')
    
    return parser.parse_args()


def preprocess(args):
    """Preprocess the RWF-2000 dataset with YOLO V8 for person detection"""
    print("\n" + "="*50)
    print("Starting preprocessing...")
    print("="*50)
    
    start_time = time.time()
    
    # Initialize preprocessor
    preprocessor = RWF2000Preprocessor(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        num_frames=args.num_frames,
        spatial_size=args.spatial_size,
        use_spatial_cropping=args.use_spatial_cropping,
        batch_size=50  # Use a batch size of 50 for memory management
    )
    
    # Convert splits string to tuple
    splits = tuple(args.splits.split(','))
    print(f"Processing splits: {splits}")
    
    # Process and save the dataset
    preprocessor.process_and_save(
        splits=splits,
        visualize_samples=args.visualize_samples,
        max_videos_per_class=args.test_mode  # Pass the test_mode parameter
    )
    
    end_time = time.time()
    print(f"Preprocessing completed in {(end_time - start_time) / 60:.2f} minutes.")
    print("="*50)


def create_dataloaders(args):
    """Create DataLoaders for training and validation with proper transforms"""
    # Get transforms
    train_transform = get_training_transforms(args.spatial_size)
    val_transform = get_validation_transforms(args.spatial_size)
    
    # Create datasets
    train_dataset = RWF2000Dataset(args.output_path, split="train", transform=train_transform)
    val_dataset = RWF2000Dataset(args.output_path, split="val", transform=val_transform)
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset, val_dataset


def train(args):
    """Train the CUE-Net model with settings from the paper"""
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    start_time = time.time()
    
    # Check if preprocessed data exists
    if not os.path.exists(args.output_path):
        print(f"Preprocessed data not found at {args.output_path}.")
        print("Please run preprocessing first.")
        return None
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Create data loaders
    train_loader, val_loader, _, _ = create_dataloaders(args)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create CUE-Net model as described in the paper
    model = CUENet(
        num_frames=args.num_frames,
        input_size=args.spatial_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        dropout=args.dropout
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer with paper's hyperparameters
    trainer = CUENetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device
    )
    
    # Train model
    results = trainer.train(
        num_epochs=args.num_epochs,
        checkpoint_dir=checkpoint_dir
    )
    
    end_time = time.time()
    print(f"Training completed in {(end_time - start_time) / 60:.2f} minutes.")
    
    # Plot and save training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot loss
    epochs = range(1, len(results['train_losses']) + 1)
    ax1.plot(epochs, results['train_losses'], 'b-', label='Training Loss')
    ax1.plot(epochs, results['val_losses'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(epochs, results['train_accs'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, results['val_accs'], 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(checkpoint_dir / 'training_curves.png', dpi=150)
    print(f"Saved training curves to {checkpoint_dir / 'training_curves.png'}")
    
    # Save final results
    results_file = checkpoint_dir / 'training_results.txt'
    with open(results_file, 'w') as f:
        f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of epochs: {args.num_epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"Weight decay: {args.weight_decay}\n")
        f.write(f"Final training loss: {results['train_losses'][-1]:.4f}\n")
        f.write(f"Final training accuracy: {results['train_accs'][-1]:.2f}%\n")
        f.write(f"Final validation loss: {results['val_losses'][-1]:.4f}\n")
        f.write(f"Final validation accuracy: {results['val_accs'][-1]:.2f}%\n")
        f.write(f"Best validation accuracy: {max(results['val_accs']):.2f}%\n")
    
    print(f"Saved training results to {results_file}")
    print("="*50)
    
    return model, results


def evaluate(args, model=None):
    """Evaluate the CUE-Net model"""
    print("\n" + "="*50)
    print("Starting evaluation...")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model if not provided
    if model is None:
        # Determine checkpoint path
        if args.checkpoint_path is None:
            # Find the best checkpoint
            checkpoint_dir = Path(args.checkpoint_dir)
            checkpoints = list(checkpoint_dir.glob("*.pt"))
            if not checkpoints:
                print("No checkpoints found. Please train the model first.")
                return
            
            # Find checkpoint with highest accuracy
            best_checkpoint = None
            best_acc = 0
            
            for checkpoint_path in checkpoints:
                # Extract accuracy from filename (assuming format like "cuenet_epoch_X_acc_Y.pt")
                filename = checkpoint_path.name
                if "acc" in filename:
                    try:
                        acc = float(filename.split("acc_")[1].split(".pt")[0])
                        if acc > best_acc:
                            best_acc = acc
                            best_checkpoint = checkpoint_path
                    except:
                        continue
            
            if best_checkpoint is None:
                print("Could not determine best checkpoint. Please specify a checkpoint path.")
                return
            
            checkpoint_path = best_checkpoint
            print(f"Using best checkpoint: {checkpoint_path} (Accuracy: {best_acc:.2f}%)")
        else:
            checkpoint_path = args.checkpoint_path
        
        # Load model
        model = CUENet(
            num_frames=args.num_frames,
            input_size=args.spatial_size,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            dropout=args.dropout
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    # Create data loaders
    _, val_loader, _, val_dataset = create_dataloaders(args)
    
    # Create output directory
    viz_dir = Path(args.visualization_dir) / "evaluation"
    viz_dir.mkdir(exist_ok=True, parents=True)
    
    # Create visualizer
    visualizer = ModelVisualizer(
        model=model,
        dataset=val_dataset,
        device=device,
        output_dir=viz_dir
    )
    
    # Evaluate model
    print("Evaluating model on validation set...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for frames, labels in val_loader:
            frames, labels = frames.to(device), labels.to(device)
            outputs = model(frames)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    report = classification_report(all_labels, all_preds, target_names=['NonFight', 'Fight'], output_dict=True)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['NonFight', 'Fight'], 
                yticklabels=['NonFight', 'Fight'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(viz_dir / 'confusion_matrix.png', dpi=150)
    plt.close()
    
    # Print classification report
    print("\nClassification Report:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Fight Class - Precision: {report['Fight']['precision']:.4f}, Recall: {report['Fight']['recall']:.4f}, F1: {report['Fight']['f1-score']:.4f}")
    print(f"NonFight Class - Precision: {report['NonFight']['precision']:.4f}, Recall: {report['NonFight']['recall']:.4f}, F1: {report['NonFight']['f1-score']:.4f}")
    
    # Save report to file
    report_file = viz_dir / "classification_report.txt"
    with open(report_file, 'w') as f:
        f.write(f"Evaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Classification Report:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Fight Class - Precision: {report['Fight']['precision']:.4f}, Recall: {report['Fight']['recall']:.4f}, F1: {report['Fight']['f1-score']:.4f}\n")
        f.write(f"NonFight Class - Precision: {report['NonFight']['precision']:.4f}, Recall: {report['NonFight']['recall']:.4f}, F1: {report['NonFight']['f1-score']:.4f}\n")
    
    print(f"Saved classification report to {report_file}")
    print("="*50)


def visualize(args):
    """Visualize model predictions and attention maps"""
    print("\n" + "="*50)
    print("Starting visualization...")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Determine checkpoint path
    if args.checkpoint_path is None:
        # Find the best checkpoint
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        if not checkpoints:
            print("No checkpoints found. Please train the model first.")
            return
        
        # Find checkpoint with highest accuracy
        best_checkpoint = None
        best_acc = 0
        
        for checkpoint_path in checkpoints:
            # Extract accuracy from filename
            filename = checkpoint_path.name
            if "acc" in filename:
                try:
                    acc = float(filename.split("acc_")[1].split(".pt")[0])
                    if acc > best_acc:
                        best_acc = acc
                        best_checkpoint = checkpoint_path
                except:
                    continue
        
        if best_checkpoint is None:
            print("Could not determine best checkpoint. Please specify a checkpoint path.")
            return
        
        checkpoint_path = best_checkpoint
        print(f"Using best checkpoint: {checkpoint_path} (Accuracy: {best_acc:.2f}%)")
    else:
        checkpoint_path = args.checkpoint_path
    
    # Load model
    model = CUENet(
        num_frames=args.num_frames,
        input_size=args.spatial_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        dropout=args.dropout
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create data loaders
    _, _, _, val_dataset = create_dataloaders(args)
    
    # Create output directory
    viz_dir = Path(args.visualization_dir) / "predictions"
    viz_dir.mkdir(exist_ok=True, parents=True)
    
    # Create visualizer
    visualizer = ModelVisualizer(
        model=model,
        dataset=val_dataset,
        device=device,
        output_dir=viz_dir
    )
    
    # Select samples to visualize
    num_samples = min(args.num_viz_samples, len(val_dataset))
    
    # Find some fight and non-fight samples
    fight_indices = []
    nonfight_indices = []
    
    for i in range(len(val_dataset)):
        _, label = val_dataset[i]
        if label == 1 and len(fight_indices) < num_samples // 2:
            fight_indices.append(i)
        elif label == 0 and len(nonfight_indices) < num_samples // 2:
            nonfight_indices.append(i)
        
        if len(fight_indices) >= num_samples // 2 and len(nonfight_indices) >= num_samples // 2:
            break
    
    indices = fight_indices + nonfight_indices
    
    # Visualize predictions
    print(f"Visualizing predictions for {len(indices)} samples...")
    
    for i, idx in enumerate(indices):
        print(f"Visualizing sample {i+1}/{len(indices)} (index {idx})...")
        
        # Get sample
        frames, label = val_dataset[idx]
        
        # Make prediction
        with torch.no_grad():
            frames_batch = frames.unsqueeze(0).to(device)
            outputs = model(frames_batch)
            _, predicted = outputs.max(1)
            confidence = torch.softmax(outputs, dim=1)[0]
        
        # Get attention visualizations
        visualizations = model.get_attention_visualizations()
        
        # Visualize prediction and attention maps
        visualizer.visualize_prediction_with_attention(
            idx, 
            frames, 
            label, 
            predicted.item(), 
            confidence, 
            visualizations
        )
    
    print(f"Saved prediction visualizations to {viz_dir}")
    print("="*50)


def main():
    """Main function to run the CUE-Net pipeline"""
    # Parse command-line arguments
    args = parse_args()
    
    # Print settings
    print("\nCUE-Net Violence Detection Pipeline")
    print("\nSettings:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    
    # Create output directories
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.visualization_dir, exist_ok=True)
    
    # Run selected pipeline mode
    model = None
    
    if args.mode == 'preprocess' or args.mode == 'all':
        preprocess(args)
    
    if args.mode == 'train' or args.mode == 'all':
        model, _ = train(args)
    
    if args.mode == 'evaluate' or args.mode == 'all':
        evaluate(args, model)
    
    if args.mode == 'visualize' or args.mode == 'all':
        visualize(args)
    
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()