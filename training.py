import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from datetime import datetime
import os

class CUENetTrainer:
    """Trainer class for CUE-Net model following the paper's methodology"""
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        learning_rate=1e-5,  # Paper specifies 1e-5
        weight_decay=1e-4,
        device='cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        
        # AdamW optimizer with exact parameters from the paper
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Cosine learning rate schedule as specified in paper
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=50,
            eta_min=learning_rate / 10
        )
    
    def train_one_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for frames, labels in self.train_loader:
            frames, labels = frames.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(frames)
            loss = self.criterion(outputs, labels)
            
            # Backward and optimize
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(self.train_loader)
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for frames, labels in self.val_loader:
                frames, labels = frames.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(frames)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(self.val_loader)
        
        return avg_loss, accuracy, all_preds, all_labels
    
    def train(self, num_epochs=50, checkpoint_dir=None):
        """
        Train the model for a specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train (paper uses 50)
            checkpoint_dir: Directory to save checkpoints
        """
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        best_val_acc = 0
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_one_epoch()
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc, _, _ = self.validate()
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # Update scheduler
            self.scheduler.step()
            
            # Print statistics
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.2e}")
            print("-" * 50)
            
            # Save checkpoint
            if checkpoint_dir and val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint_path = checkpoint_dir / f"cuenet_epoch_{epoch+1}_acc_{val_acc:.2f}.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
        
        return {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs
        }