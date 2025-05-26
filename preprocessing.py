import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
import shutil
from sklearn.model_selection import train_test_split
import gc  # For garbage collection

class RWF2000Preprocessor:
    """
    Preprocessing class for the RWF-2000 dataset.
    Implements the preprocessing techniques from the CUE-Net paper.
    """
    def __init__(
        self,
        dataset_path,
        output_path=None,
        num_frames=64,
        spatial_size=224,
        use_spatial_cropping=True,
        batch_size=50,  # Process videos in batches to manage memory
        seed=42
    ):
        """
        Initialize the preprocessor.
        
        Args:
            dataset_path: Path to the RWF-2000 dataset
            output_path: Path to save processed data (if None, will use dataset_path/processed)
            num_frames: Number of frames to extract from each video
            spatial_size: Target frame resolution (spatial_size x spatial_size)
            use_spatial_cropping: Whether to use spatial cropping as in CUE-Net
            batch_size: Number of videos to process at once before garbage collection
            seed: Random seed for reproducibility
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path) if output_path else self.dataset_path / "processed"
        self.num_frames = num_frames
        self.spatial_size = spatial_size
        self.use_spatial_cropping = use_spatial_cropping
        self.batch_size = batch_size
        self.seed = seed
        
        # Set seeds for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        # ImageNet normalization parameters
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]
        
        # Create output directory
        self.output_path.mkdir(exist_ok=True, parents=True)
        
        # Setup augmentation transforms
        self._setup_transforms()
        
        print(f"RWF2000Preprocessor initialized with:")
        print(f"- Dataset path: {self.dataset_path}")
        print(f"- Output path: {self.output_path}")
        print(f"- Frames per video: {self.num_frames}")
        print(f"- Spatial size: {self.spatial_size}x{self.spatial_size}")
        print(f"- Spatial cropping: {self.use_spatial_cropping}")
        print(f"- Batch size: {self.batch_size}")
    
    def _setup_transforms(self):
        """Setup various transforms for preprocessing."""
        # Basic transform (resize only)
        self.basic_transform = A.Compose([
            A.Resize(height=self.spatial_size, width=self.spatial_size),
            A.Normalize(mean=self.norm_mean, std=self.norm_std),
            ToTensorV2(),
        ])
        
        # Training transforms with augmentation
        self.train_transform = A.Compose([
            A.Resize(height=self.spatial_size + 32, width=self.spatial_size + 32),
            A.RandomCrop(height=self.spatial_size, width=self.spatial_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.Normalize(mean=self.norm_mean, std=self.norm_std),
            ToTensorV2(),
        ])
        
        # Validation transform (no augmentation)
        self.val_transform = A.Compose([
            A.Resize(height=self.spatial_size, width=self.spatial_size),
            A.Normalize(mean=self.norm_mean, std=self.norm_std),
            ToTensorV2(),
        ])
    
    def find_videos(self, split="train"):
        """
        Find all video files in the specified split.
        
        Args:
            split: Dataset split, either "train" or "val"
            
        Returns:
            List of (video_path, label) tuples, where label is 1 for Fight, 0 for NonFight
        """
        split_dir = self.dataset_path / split
        fight_dir = split_dir / "Fight"
        nonfight_dir = split_dir / "NonFight"
        
        # Check if directories exist
        if not split_dir.exists():
            raise ValueError(f"Split directory {split_dir} does not exist.")
        if not fight_dir.exists() or not nonfight_dir.exists():
            raise ValueError(f"Fight or NonFight directory missing in {split_dir}.")
        
        # Find all video files
        fight_videos = [(path, 1) for path in fight_dir.glob("*.avi")]
        nonfight_videos = [(path, 0) for path in nonfight_dir.glob("*.avi")]
        
        # Combine and shuffle
        all_videos = fight_videos + nonfight_videos
        random.shuffle(all_videos)
        
        print(f"Found {len(fight_videos)} fight videos and {len(nonfight_videos)} non-fight videos in {split} split.")
        return all_videos
    
    def detect_people(self, frame):
        """
        Detects people in a frame using YOLO V8.
        
        Args:
            frame: Input frame (RGB format)
            
        Returns:
            List of bounding boxes [(x_min, y_min, x_max, y_max), ...]
        """
        if not hasattr(self, 'yolo_model'):
            # Load YOLO model on first use
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO('yolov8n.pt')  # Use smaller model for faster processing
            except ImportError:
                print("YOLO not installed. Please install with: pip install ultralytics")
                # Fall back to Haar cascades if YOLO isn't available
                return self._detect_people_fallback(frame)
        
        # Run detection (YOLO expects BGR format)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = self.yolo_model(frame_bgr, conf=0.25)  # Lower confidence for better recall
        
        # Extract person detections
        bboxes = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Check if class is person (class 0 in COCO)
                if int(box.cls) == 0:  
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bboxes.append((int(x1), int(y1), int(x2), int(y2)))
        
        return bboxes
    
    def _detect_people_fallback(self, frame):
        """Fallback to Haar cascades if YOLO isn't available"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_fullbody.xml"
        body_cascade = cv2.CascadeClassifier(cascade_path)
        
        people = body_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Convert from (x,y,w,h) to (x1,y1,x2,y2)
        return [(x, y, x+w, y+h) for (x, y, w, h) in people]

    def spatial_crop(self, frames):
        """
        Apply spatial cropping based on detected people following Algorithm 1 from the CUE-Net paper.
        
        Args:
            frames: List of video frames
            
        Returns:
            Spatially cropped frames
        """
        if not self.use_spatial_cropping or not frames:
            return frames
        
        # Initialize with extreme values
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = 0, 0
        max_people = 0
        
        # Detect people in every frame and find max bounding box
        for frame in frames:
            people = self.detect_people(frame)
            
            if len(people) > 0:
                for (x1, y1, x2, y2) in people:
                    x_min = min(x_min, x1)
                    y_min = min(y_min, y1)
                    x_max = max(x_max, x2)
                    y_max = max(y_max, y2)
                
                max_people = max(max_people, len(people))
        
        # Only crop if more than one person is detected (as per Algorithm 1)
        if max_people > 1 and x_min < float('inf'):
            # Add margin
            height, width = frames[0].shape[:2]
            margin = min(width, height) // 10
            
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(width, x_max + margin)
            y_max = min(height, y_max + margin)
            
            # Crop all frames
            cropped_frames = [frame[y_min:y_max, x_min:x_max] for frame in frames]
            return cropped_frames
        
        # Return original frames if only 1 or 0 people detected (per Algorithm 1)
        return frames
    
    def extract_frames(self, video_path, apply_crop=True):
        """
        Extract frames from a video file with uniform sampling.
        
        Args:
            video_path: Path to the video file
            apply_crop: Whether to apply spatial cropping
            
        Returns:
            List of extracted frames
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            return None
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames uniformly
        if total_frames >= self.num_frames:
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            # For shorter videos, loop frames to reach required count
            base_indices = np.linspace(0, total_frames - 1, total_frames, dtype=int)
            # Repeat indices as needed
            repeats = np.ceil(self.num_frames / total_frames).astype(int)
            extended_indices = np.tile(base_indices, repeats)
            frame_indices = extended_indices[:self.num_frames]
        
        # Extract frames
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                print(f"Failed to read frame {idx} from {video_path}")
                # If frame reading fails, add a blank frame
                if frames:
                    blank_frame = np.zeros_like(frames[0])
                    frames.append(blank_frame)
                else:
                    # Default size if no frames are available
                    blank_frame = np.zeros((self.spatial_size, self.spatial_size, 3), dtype=np.uint8)
                    frames.append(blank_frame)
        
        cap.release()
        
        # Apply spatial cropping if enabled and if there are frames
        if apply_crop and frames:
            frames = self.spatial_crop(frames)
            
        return frames
    
    def process_video(self, video_path, label, split="train", visualize=False):
        """
        Process a single video.
        
        Args:
            video_path: Path to the video file
            label: Video label (1 for Fight, 0 for NonFight)
            split: Dataset split ("train" or "val")
            visualize: Whether to visualize the processing steps
            
        Returns:
            Processed frames tensor
        """
        # Extract frames
        frames = self.extract_frames(video_path, apply_crop=self.use_spatial_cropping)
        
        if frames is None or len(frames) == 0:
            print(f"Failed to extract frames from {video_path}")
            return None
        
        # Select transform based on split
        transform = self.train_transform if split == "train" else self.val_transform
        
        # Apply transforms to each frame
        processed_frames = []
        for frame in frames:
            # Resize frame if needed (some frames may be oddly sized after cropping)
            if frame.shape[0] != frame.shape[1]:
                frame = cv2.resize(frame, (self.spatial_size, self.spatial_size))
            
            # Apply transformation
            processed = transform(image=frame)["image"]
            processed_frames.append(processed)
        
        # Stack frames
        # [T, C, H, W] - Time, Channels, Height, Width
        processed_tensor = torch.stack(processed_frames)
        
        # Visualize if requested
        if visualize:
            self.visualize_processing(frames, processed_tensor, label, video_path)
        
        return processed_tensor
    
    def visualize_processing(self, original_frames, processed_tensor, label, video_path):
        """
        Visualize the preprocessing steps.
        
        Args:
            original_frames: List of original frames
            processed_tensor: Tensor of processed frames
            label: Video label
            video_path: Path to the video file
        """
        # Create figure
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f"Preprocessing Visualization\nVideo: {Path(video_path).name}, Label: {'Fight' if label == 1 else 'NonFight'}", fontsize=16)
        
        # Select frames to display (uniformly)
        indices = np.linspace(0, len(original_frames) - 1, 4, dtype=int)
        
        # Display original frames
        for i, idx in enumerate(indices):
            axes[0, i].imshow(original_frames[idx])
            axes[0, i].set_title(f"Original Frame {idx}")
            axes[0, i].axis('off')
        
        # Display processed frames
        for i, idx in enumerate(indices):
            # Convert tensor to numpy for display
            processed_frame = processed_tensor[idx].permute(1, 2, 0).numpy()
            
            # Denormalize for display
            mean = np.array(self.norm_mean).reshape(1, 1, 3)
            std = np.array(self.norm_std).reshape(1, 1, 3)
            processed_frame = processed_frame * std + mean
            processed_frame = np.clip(processed_frame, 0, 1)
            
            axes[1, i].imshow(processed_frame)
            axes[1, i].set_title(f"Processed Frame {idx}")
            axes[1, i].axis('off')
        
        plt.tight_layout()
        # Save the visualization
        viz_dir = self.output_path / "visualizations"
        viz_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(viz_dir / f"{Path(video_path).stem}_preprocessing.png", dpi=150)
        plt.close(fig)
        
        # Clean up to release memory
        plt.close('all')
        
    def process_dataset_in_batches(self, split="train", visualize_samples=0, max_videos_per_class=None):
        """
        Process the dataset in batches to manage memory.
        
        Args:
            split: Dataset split to process
            visualize_samples: Number of samples to visualize
            max_videos_per_class: Maximum number of videos to process per class (for testing)
        """
        print(f"\nProcessing {split} split in batches...")
        all_videos = self.find_videos(split)
        
        # If max_videos_per_class is specified, limit the number of videos per class
        if max_videos_per_class is not None:
            print(f"⚠️ USING TEST MODE: Processing only {max_videos_per_class} videos per class")
            fight_videos = [v for v in all_videos if v[1] == 1][:max_videos_per_class]
            nonfight_videos = [v for v in all_videos if v[1] == 0][:max_videos_per_class]
            videos = fight_videos + nonfight_videos
            print(f"Selected {len(videos)} videos ({len(fight_videos)} fight, {len(nonfight_videos)} non-fight)")
        else:
            videos = all_videos
        
        # Determine which videos to visualize
        if visualize_samples > 0:
            visualize_indices = np.linspace(0, len(videos) - 1, visualize_samples, dtype=int)
        else:
            visualize_indices = []
        
        # Process videos in batches
        total_processed = 0
        batch_start = 0
        
        # Create directory
        split_dir = self.output_path / split
        split_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize labels file
        labels_file = split_dir / "labels.txt"
        with open(labels_file, 'w') as f:
            pass  # Create empty file
        
        while batch_start < len(videos):
            batch_end = min(batch_start + self.batch_size, len(videos))
            batch_videos = videos[batch_start:batch_end]
            
            print(f"Processing batch {batch_start//self.batch_size + 1} ({batch_start} to {batch_end-1})...")
            
            # Process each video in the batch
            for i, (video_path, label) in enumerate(tqdm(batch_videos, desc=f"Processing {split} videos (batch {batch_start//self.batch_size + 1})")):
                idx = batch_start + i
                visualize = idx in visualize_indices
                
                tensor = self.process_video(video_path, label, split, visualize)
                
                if tensor is not None:
                    # Create filename
                    file_path = split_dir / f"{idx:04d}_{label}.pt"
                    
                    # Save tensor
                    torch.save(tensor, file_path)
                    
                    # Append to labels file
                    with open(labels_file, 'a') as f:
                        f.write(f"{idx:04d}_{label}.pt {label}\n")
                    
                    total_processed += 1
                
                # Clean up memory
                tensor = None
                
                if (i + 1) % 10 == 0:  # Clean memory every 10 videos
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Clean up memory after batch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            batch_start = batch_end
            print(f"Completed batch. Total processed so far: {total_processed}")
    
        print(f"Processed {total_processed} videos for {split} split.")
        
    def process_and_save(self, splits=("train", "val"), visualize_samples=0, max_videos_per_class=None):
        """
        Process and save the dataset.
        
        Args:
            splits: Tuple of splits to process
            visualize_samples: Number of samples to visualize per split
            max_videos_per_class: Maximum number of videos to process per class (for testing)
        """
        start_time = time.time()
        
        for split in splits:
            self.process_dataset_in_batches(split, visualize_samples, max_videos_per_class)
        
        end_time = time.time()
        print(f"\nComplete preprocessing completed in {end_time - start_time:.2f} seconds.")


class RWF2000Dataset(Dataset):
    """
    PyTorch Dataset for the preprocessed RWF-2000 dataset.
    """
    def __init__(self, data_path, split="train", transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the preprocessed data
            split: Dataset split ("train" or "val")
            transform: Additional transforms to apply
        """
        self.data_path = Path(data_path) / split
        self.transform = transform
        
        # Load labels
        self.samples = []
        labels_file = self.data_path / "labels.txt"
        
        if not labels_file.exists():
            raise FileNotFoundError(f"Labels file not found at {labels_file}. Please run preprocessing first.")
        
        with open(labels_file, 'r') as f:
            for line in f:
                filename, label = line.strip().split()
                self.samples.append((filename, int(label)))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        
        # Load tensor
        tensor_path = self.data_path / filename
        frames = torch.load(tensor_path)  # Shape: [T, C, H, W]
        
        # Apply transform if provided
        if self.transform:
            # Convert to appropriate format and apply transforms
            from PIL import Image
            import numpy as np
            
            transformed_frames = []
            
            for t in range(frames.shape[0]):
                # Convert tensor to numpy array (in HWC format for PIL)
                frame_np = frames[t].permute(1, 2, 0).numpy()
                # Ensure values are in [0, 255] uint8 range for PIL
                frame_np = (frame_np * 255).astype(np.uint8)
                # Convert to PIL image directly
                frame_pil = Image.fromarray(frame_np)
                # Apply transforms
                frame_tensor = self.transform(frame_pil)
                transformed_frames.append(frame_tensor)
            
            # Stack back into tensor
            frames = torch.stack(transformed_frames)
        
        # Change order from [T, C, H, W] to [C, T, H, W]
        frames = frames.permute(1, 0, 2, 3)
        
        return frames, label


def main():
    """
    Main function to preprocess the RWF-2000 dataset.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess RWF-2000 dataset')
    parser.add_argument('--dataset_path', type=str, default="D:\\Thaman\\archive\\RWF-2000", 
                        help='Path to the RWF-2000 dataset')
    parser.add_argument('--output_path', type=str, default="D:\\Thaman\\archive\\RWF-2000-processed", 
                        help='Path to save processed data')
    parser.add_argument('--num_frames', type=int, default=64, 
                        help='Number of frames to extract from each video')
    parser.add_argument('--spatial_size', type=int, default=224, 
                        help='Spatial size of frames')
    parser.add_argument('--use_spatial_cropping', action='store_true', 
                        help='Use spatial cropping')
    parser.add_argument('--batch_size', type=int, default=50, 
                        help='Number of videos to process at once')
    parser.add_argument('--visualize_samples', type=int, default=0, 
                        help='Number of samples to visualize')
    parser.add_argument('--splits', type=str, default='train,val', 
                        help='Comma-separated list of splits to process')
    args = parser.parse_args()
    
    # Convert splits string to tuple
    splits = tuple(args.splits.split(','))
    
    # Initialize preprocessor
    preprocessor = RWF2000Preprocessor(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        num_frames=args.num_frames,
        spatial_size=args.spatial_size,
        use_spatial_cropping=args.use_spatial_cropping,
        batch_size=args.batch_size
    )
    
    # Process and save the dataset
    preprocessor.process_and_save(
        splits=splits,
        visualize_samples=args.visualize_samples
    )


if __name__ == "__main__":
    main()