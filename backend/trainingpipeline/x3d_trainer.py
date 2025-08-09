import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class X3DTrainer:
    """
    Advanced trainer for X3D violence detection with mixed precision training,
    learning rate scheduling, and comprehensive monitoring.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        mixed_precision: bool = True,
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 10,
        patience: int = 10,
        min_delta: float = 1e-4
    ):
        """
        Args:
            model: X3D violence detection model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            mixed_precision: Whether to use mixed precision training
            checkpoint_dir: Directory to save checkpoints
            log_interval: Logging interval
            patience: Early stopping patience
            min_delta: Minimum change for early stopping
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.mixed_precision = mixed_precision
        self.log_interval = log_interval
        self.patience = patience
        self.min_delta = min_delta
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Mixed precision scaler
        if self.mixed_precision:
            self.scaler = GradScaler()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'learning_rates': []
        }
        
        # Best validation metrics
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.epochs_without_improvement = 0
        
        print(f"Trainer initialized with:")
        print(f"  Device: {device}")
        print(f"  Mixed precision: {mixed_precision}")
        print(f"  Checkpoint dir: {checkpoint_dir}")
        print(f"  Early stopping patience: {patience}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {epoch+1} [Train]",
            leave=False
        )
        
        for batch_idx, (data, labels) in enumerate(progress_bar):
            # Move data to device
            if isinstance(data, dict):
                data = {k: v.to(self.device, non_blocking=True) for k, v in data.items()}
            else:
                data = data.to(self.device, non_blocking=True)
            
            labels = labels.to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.mixed_precision:
                with autocast():
                    outputs = self.model(data)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            if batch_idx % self.log_interval == 0:
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
        
        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_predictions) * 100
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        progress_bar = tqdm(
            self.val_loader, 
            desc=f"Epoch {epoch+1} [Val]",
            leave=False
        )
        
        with torch.no_grad():
            for data, labels in progress_bar:
                # Move data to device
                if isinstance(data, dict):
                    data = {k: v.to(self.device, non_blocking=True) for k, v in data.items()}
                else:
                    data = data.to(self.device, non_blocking=True)
                
                labels = labels.to(self.device, non_blocking=True)
                
                # Forward pass
                if self.mixed_precision:
                    with autocast():
                        outputs = self.model(data)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(data)
                    loss = self.criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                predictions = outputs.argmax(dim=1)
                probabilities = torch.softmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate detailed metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_predictions) * 100
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
        
        # Calculate per-class metrics
        cm = confusion_matrix(all_labels, all_predictions)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.mixed_precision:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"New best model saved! Val Acc: {metrics['accuracy']:.2f}%, Val F1: {metrics['f1']:.2f}%")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and self.mixed_precision:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
            
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    
    def plot_confusion_matrix(self, cm: np.ndarray, epoch: int):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(8, 6))
        
        class_names = ['Non-Violence', 'Violence']
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True
        )
        
        plt.title(f'Confusion Matrix - Epoch {epoch+1}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Save the plot
        cm_path = self.checkpoint_dir / f"confusion_matrix_epoch_{epoch+1}.png"
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score
        axes[1, 0].plot(epochs, self.history['val_f1'], 'g-', label='Val F1 Score', linewidth=2)
        axes[1, 0].set_title('Validation F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 1].plot(epochs, self.history['learning_rates'], 'orange', linewidth=2)
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        curves_path = self.checkpoint_dir / "training_curves.png"
        plt.savefig(curves_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {curves_path}")
    
    def train(self, num_epochs: int, resume_from: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Train the model
        
        Args:
            num_epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume from
            
        Returns:
            Training history
        """
        start_epoch = 0
        
        # Resume training if checkpoint provided
        if resume_from and Path(resume_from).exists():
            start_epoch = self.load_checkpoint(resume_from)
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("="*50)
        
        training_start_time = time.time()
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate_epoch(epoch)
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Check for improvement
            is_best = False
            if val_metrics['accuracy'] > self.best_val_acc + self.min_delta:
                self.best_val_acc = val_metrics['accuracy']
                self.best_val_f1 = val_metrics['f1']
                self.epochs_without_improvement = 0
                is_best = True
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Plot confusion matrix
            self.plot_confusion_matrix(val_metrics['confusion_matrix'], epoch)
            
            # Print epoch results
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1}/{num_epochs} - {epoch_time:.1f}s")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
            print(f"  Val Precision: {val_metrics['precision']:.2f}%, Val Recall: {val_metrics['recall']:.2f}%")
            print(f"  Val F1: {val_metrics['f1']:.2f}%, LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            if is_best:
                print("  *** NEW BEST MODEL ***")
            
            print(f"  Epochs without improvement: {self.epochs_without_improvement}")
            print("-" * 50)
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
                break
        
        # Training completed
        training_time = time.time() - training_start_time
        print(f"\nTraining completed in {training_time/60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"Best validation F1: {self.best_val_f1:.2f}%")
        
        # Plot final training curves
        self.plot_training_curves()
        
        # Save final history
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            history_json = {}
            for key, value in self.history.items():
                if isinstance(value, list):
                    history_json[key] = [float(x) if isinstance(x, (np.integer, np.floating)) else x for x in value]
                else:
                    history_json[key] = value
            json.dump(history_json, f, indent=2)
        
        print(f"Training history saved to {history_path}")
        
        return self.history


def create_optimizer_and_scheduler(
    model: nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    scheduler_type: str = "cosine",
    num_epochs: int = 50,
    warmup_epochs: int = 5
):
    """
    Create optimizer and learning rate scheduler
    
    Args:
        model: Model to optimize
        learning_rate: Initial learning rate
        weight_decay: Weight decay factor
        scheduler_type: Type of scheduler ('cosine', 'step', 'plateau')
        num_epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs
    """
    # AdamW optimizer (works well with transformers)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler
    if scheduler_type == "cosine":
        # Fix: Ensure T_max is at least 1 to avoid division by zero
        T_max = max(1, num_epochs - warmup_epochs)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=learning_rate * 0.01
        )
        print(f"Cosine scheduler: T_max={T_max}, eta_min={learning_rate * 0.01}")
    elif scheduler_type == "step":
        # For short training, use smaller milestones
        if num_epochs <= 10:
            milestones = [num_epochs // 2, 3 * num_epochs // 4]
        else:
            milestones = [num_epochs//3, 2*num_epochs//3]
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=0.1
        )
        print(f"Step scheduler: milestones={milestones}")
    elif scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=max(2, num_epochs // 10),  # Adaptive patience
            threshold=1e-4,
            min_lr=learning_rate * 0.01
        )
        print(f"Plateau scheduler: patience={max(2, num_epochs // 10)}")
    else:
        scheduler = None
    
    print(f"Optimizer: AdamW (lr={learning_rate}, wd={weight_decay})")
    print(f"Scheduler: {scheduler_type}")
    
    return optimizer, scheduler


if __name__ == "__main__":
    # Test the trainer
    print("Testing trainer components...")
    
    # This would normally be called from the main training script
    print("Trainer test completed!")