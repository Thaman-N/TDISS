import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from pathlib import Path
import os
import cv2
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm


class ModelVisualizer:
    """
    Enhanced visualizer for CUE-Net model predictions and attention mechanisms.
    """
    def __init__(self, model, dataset, device='cuda', output_dir='visualizations'):
        """
        Initialize the visualizer.
        
        Args:
            model: Trained model
            dataset: Dataset to visualize
            device: Device to use for inference
            output_dir: Directory to save visualizations
        """
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # ImageNet mean and std for denormalization
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    
    def denormalize_frame(self, frame_tensor):
        """
        Denormalize a frame tensor for visualization.
        
        Args:
            frame_tensor: Normalized frame tensor [C, H, W]
            
        Returns:
            Denormalized frame as numpy array [H, W, C]
        """
        # Convert tensor to numpy and transpose
        frame = frame_tensor.permute(1, 2, 0).cpu().numpy()
        
        # Denormalize
        frame = frame * self.std + self.mean
        frame = np.clip(frame, 0, 1)
        
        return frame
    
    def visualize_prediction_with_attention(self, idx, frames, label, pred, confidence, visualizations):
        """
        Visualize the prediction and attention maps for a single sample.
        
        Args:
            idx: Sample index
            frames: Frame tensor [T, C, H, W]
            label: True label
            pred: Predicted label
            confidence: Prediction confidences
            visualizations: Dictionary of attention visualizations
        """
        # Get class names
        class_names = ['NonFight', 'Fight']
        predicted_class = class_names[pred]
        true_class = class_names[label]
        
        # Create figure with subplots for frames and attention maps
        fig = plt.figure(figsize=(20, 10))
        
        # Define grid layout
        frame_grid = plt.GridSpec(2, 4, left=0.05, right=0.48, wspace=0.05, hspace=0.15)
        attn_grid = plt.GridSpec(2, 2, left=0.55, right=0.98, wspace=0.2, hspace=0.3)
        
        # Set title
        fig.suptitle(
            f"CUE-Net Prediction: {predicted_class} ({confidence[pred]:.2f})\n"
            f"True Class: {true_class}",
            fontsize=16
        )
        
        # Select frames to display
        num_frames = frames.shape[0]
        indices = np.linspace(0, num_frames - 1, 8, dtype=int)
        
        # Display frames
        for i, frame_idx in enumerate(indices):
            ax = fig.add_subplot(frame_grid[i // 4, i % 4])
            
            # Denormalize frame
            frame = self.denormalize_frame(frames[frame_idx])
            
            ax.imshow(frame)
            ax.set_title(f"Frame {frame_idx}")
            ax.axis('off')
        
        # Display attention maps if available
        plot_idx = 0
        
        # Spatial attention
        if 'spatial_attention' in visualizations and visualizations['spatial_attention'] is not None:
            ax = fig.add_subplot(attn_grid[0, 0])
            spatial_attn = visualizations['spatial_attention']
            
            if len(spatial_attn.shape) > 2:
                # If 3D, take mean across channels
                spatial_attn = np.mean(spatial_attn, axis=0)
            
            im = ax.imshow(spatial_attn, cmap='hot')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title("Spatial Attention")
            ax.axis('off')
            plot_idx += 1
        
        # Self-attention
        if 'self_attention' in visualizations and visualizations['self_attention'] is not None:
            ax = fig.add_subplot(attn_grid[0, 1])
            self_attn = visualizations['self_attention']
            
            im = ax.imshow(self_attn, cmap='viridis')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title("Self-Attention")
            ax.axis('off')
            plot_idx += 1
        
        # MEAA attention
        if 'meaa_attention' in visualizations and visualizations['meaa_attention'] is not None:
            ax = fig.add_subplot(attn_grid[1, 0])
            meaa_attn = visualizations['meaa_attention']
            
            # If scalar, create a simple bar plot
            if np.isscalar(meaa_attn) or (isinstance(meaa_attn, np.ndarray) and meaa_attn.size == 1):
                ax.bar(['Attention Weight'], [float(meaa_attn)], color='teal')
                ax.set_ylim(0, 1)
            else:
                # If vector, plot as heatmap
                im = ax.imshow(meaa_attn.reshape(1, -1), cmap='plasma', aspect='auto')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            ax.set_title("MEAA Attention")
        
        # Additional attention visualization if needed
        if plot_idx < 3 and len(visualizations) > plot_idx:
            ax = fig.add_subplot(attn_grid[1, 1])
            ax.text(0.5, 0.5, "Additional attention maps\ncan be visualized here", 
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        
        # Save the visualization
        fig_path = self.output_dir / f"prediction_attention_{idx}.png"
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)
        
        print(f"Saved visualization to {fig_path}")
        
        # Create separate attention map visualization
        self.create_attention_map_video(idx, frames, visualizations)
    
    def create_attention_map_video(self, idx, frames, visualizations):
        """
        Create a video showing attention maps overlaid on frames.
        
        Args:
            idx: Sample index
            frames: Frame tensor [T, C, H, W]
            visualizations: Dictionary of attention visualizations
        """
        # Only proceed if we have spatial attention
        if 'spatial_attention' not in visualizations or visualizations['spatial_attention'] is None:
            return
        
        # Create output directory
        video_dir = self.output_dir / "videos"
        video_dir.mkdir(exist_ok=True, parents=True)
        
        # Get spatial attention
        spatial_attn = visualizations['spatial_attention']
        
        # Process a subset of frames (every 4th frame)
        num_frames = min(16, frames.shape[0])
        step = max(1, frames.shape[0] // num_frames)
        frame_indices = np.arange(0, frames.shape[0], step)[:num_frames]
        
        # Prepare frames for video
        video_frames = []
        
        for i, frame_idx in enumerate(frame_indices):
            # Create a figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Denormalize frame
            frame = self.denormalize_frame(frames[frame_idx])
            
            # Display frame
            ax.imshow(frame)
            
            # Overlay spatial attention if available
            if spatial_attn is not None:
                if len(spatial_attn.shape) > 2:
                    # If 3D, take the corresponding time step if possible
                    if spatial_attn.shape[1] > frame_idx:
                        attn = spatial_attn[:, frame_idx, :]
                    else:
                        attn = np.mean(spatial_attn, axis=(1, 2))
                else:
                    attn = spatial_attn
                
                # Resize attention to match frame
                attn_resized = cv2.resize(attn.astype(np.float32), 
                                         (frame.shape[1], frame.shape[0]))
                
                # Create a mask with alpha channel based on attention
                attn_norm = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
                attn_colored = plt.cm.hot(attn_norm)
                attn_colored[..., 3] = attn_norm * 0.7  # Alpha channel
                
                # Overlay attention on frame
                ax.imshow(attn_colored, alpha=0.5)
            
            ax.set_title(f"Frame {frame_idx} with Attention Overlay")
            ax.axis('off')
            
            # Convert figure to image
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            video_frames.append(img)
            plt.close(fig)
        
        # Create video
        if video_frames:
            height, width, _ = video_frames[0].shape
            video_path = video_dir / f"attention_overlay_{idx}.mp4"
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, 2, (width, height))
            
            for frame in video_frames:
                # Convert RGB to BGR for OpenCV
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            out.release()
            print(f"Saved attention overlay video to {video_path}")
    
    def visualize_class_activation_maps(self, idx):
        """
        Visualize Grad-CAM class activation maps for a sample.
        
        Args:
            idx: Sample index
        """
        # Get sample
        frames, label = self.dataset[idx]
        
        # Currently not implemented as it requires model architecture changes
        # This would be implemented when the model supports extracting
        # the activation maps from the convolutional layers
        
        pass
    
    def evaluate_model(self, dataloader):
        """
        Evaluate the model on the given dataloader.
        
        Args:
            dataloader: DataLoader to evaluate on
            
        Returns:
            Dictionary with evaluation metrics
        """
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for frames, labels in tqdm(dataloader, desc="Evaluating"):
                frames, labels = frames.to(self.device), labels.to(self.device)
                
                outputs = self.model(frames)
                probs = torch.softmax(outputs, dim=1)
                _, preds = outputs.max(1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, target_names=['NonFight', 'Fight'], output_dict=True)
        
        return {
            'confusion_matrix': cm,
            'classification_report': report,
            'labels': all_labels,
            'predictions': all_preds,
            'probabilities': all_probs
        }
    
    def plot_confusion_matrix(self, cm, save=True):
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            save: Whether to save the plot
            
        Returns:
            Figure object
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['NonFight', 'Fight'], yticklabels=['NonFight', 'Fight'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save:
            plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=150)
            print(f"Saved confusion matrix to {self.output_dir / 'confusion_matrix.png'}")
        
        return plt.gcf()
    
    def plot_misclassifications(self, dataloader, num_samples=5):
        """
        Plot misclassified examples.
        
        Args:
            dataloader: DataLoader to evaluate
            num_samples: Number of misclassified samples to plot
        """
        # Collect misclassified samples
        misclassified = []
        
        with torch.no_grad():
            for i, (frames, labels) in enumerate(dataloader):
                frames, labels = frames.to(self.device), labels.to(self.device)
                
                outputs = self.model(frames)
                _, preds = outputs.max(1)
                
                # Find misclassified samples
                for j in range(len(labels)):
                    if preds[j] != labels[j]:
                        misclassified.append({
                            'idx': i * dataloader.batch_size + j,
                            'frames': frames[j],
                            'label': labels[j].item(),
                            'pred': preds[j].item(),
                            'confidence': torch.softmax(outputs[j], dim=0)[preds[j]].item()
                        })
                        
                        if len(misclassified) >= num_samples:
                            break
                
                if len(misclassified) >= num_samples:
                    break
        
        # Plot misclassified samples
        if not misclassified:
            print("No misclassified samples found.")
            return
        
        print(f"Found {len(misclassified)} misclassified samples.")
        
        # Create output directory
        misclass_dir = self.output_dir / "misclassified"
        misclass_dir.mkdir(exist_ok=True, parents=True)
        
        # Class names
        class_names = ['NonFight', 'Fight']
        
        for i, sample in enumerate(misclassified):
            # Create figure
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            fig.suptitle(
                f"Misclassified Sample {i+1}\n"
                f"True: {class_names[sample['label']]}, Pred: {class_names[sample['pred']]} ({sample['confidence']:.2f})",
                fontsize=16
            )
            
            # Select frames to display
            num_frames = sample['frames'].shape[0]
            indices = np.linspace(0, num_frames - 1, 8, dtype=int)
            
            # Display frames
            for j, idx in enumerate(indices):
                ax = axes[j // 4, j % 4]
                
                # Denormalize frame
                frame = self.denormalize_frame(sample['frames'][idx])
                
                ax.imshow(frame)
                ax.set_title(f"Frame {idx}")
                ax.axis('off')
            
            plt.tight_layout()
            
            # Save figure
            fig_path = misclass_dir / f"misclassified_{i+1}.png"
            plt.savefig(fig_path, dpi=150)
            plt.close(fig)
            
            print(f"Saved misclassified sample {i+1} to {fig_path}")