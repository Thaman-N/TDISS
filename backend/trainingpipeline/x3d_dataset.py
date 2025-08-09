import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from pathlib import Path
import random
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class RWF2000X3DDataset(Dataset):
    """
    Optimized RWF-2000 dataset for X3D training.
    Focuses on motion-enhanced feature extraction.
    """
    
    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        clip_len: int = 16,
        spatial_size: int = 224,
        sampling_rate: int = 4,
        num_retries: int = 10,
        transform: Optional[A.Compose] = None,
        compute_optical_flow: bool = True,
        max_videos_per_class: Optional[int] = None  # For testing
    ):
        """
        Args:
            dataset_path: Path to RWF-2000 dataset
            split: 'train' or 'val'  
            clip_len: Number of frames to extract (X3D works well with 16)
            spatial_size: Spatial resolution (224 for X3D)
            sampling_rate: Temporal sampling rate
            num_retries: Number of retries for corrupted videos
            transform: Albumentations transforms
            compute_optical_flow: Whether to compute optical flow for motion
            max_videos_per_class: Limit videos per class (for testing)
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.clip_len = clip_len
        self.spatial_size = spatial_size
        self.sampling_rate = sampling_rate
        self.num_retries = num_retries
        self.transform = transform
        self.compute_optical_flow = compute_optical_flow
        
        # Load video paths and labels
        self.video_paths, self.labels = self._load_dataset(max_videos_per_class)
        
        # ImageNet normalization (required for X3D)
        self.mean = np.array([0.45, 0.45, 0.45])
        self.std = np.array([0.225, 0.225, 0.225])
        
        print(f"Loaded {len(self.video_paths)} videos for {split} split")
        print(f"Fight videos: {sum(self.labels)}, Non-fight videos: {len(self.labels) - sum(self.labels)}")
    
    def _load_dataset(self, max_videos_per_class: Optional[int]) -> Tuple[list, list]:
        """Load video paths and labels"""
        video_paths = []
        labels = []
        
        split_dir = self.dataset_path / self.split
        
        # Load Fight videos (label = 1)
        fight_dir = split_dir / "Fight"
        if fight_dir.exists():
            fight_videos = list(fight_dir.glob("*.avi"))
            if max_videos_per_class:
                fight_videos = fight_videos[:max_videos_per_class]
            video_paths.extend(fight_videos)
            labels.extend([1] * len(fight_videos))
        
        # Load NonFight videos (label = 0)
        nonfight_dir = split_dir / "NonFight"
        if nonfight_dir.exists():
            nonfight_videos = list(nonfight_dir.glob("*.avi"))
            if max_videos_per_class:
                nonfight_videos = nonfight_videos[:max_videos_per_class]
            video_paths.extend(nonfight_videos)
            labels.extend([0] * len(nonfight_videos))
        
        # Shuffle the data
        combined = list(zip(video_paths, labels))
        random.shuffle(combined)
        video_paths, labels = zip(*combined)
        
        print(f"Found {len(video_paths)} videos ({sum(labels)} fight, {len(labels) - sum(labels)} non-fight)")
        
        return list(video_paths), list(labels)
    
    def _extract_frames(self, video_path: Path) -> np.ndarray:
        """Extract frames from video with temporal sampling"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame indices for temporal sampling
        required_frames = self.clip_len * self.sampling_rate
        
        if total_frames >= required_frames:
            # Uniform sampling
            start_idx = random.randint(0, total_frames - required_frames)
            frame_indices = np.arange(start_idx, start_idx + required_frames, self.sampling_rate)
        else:
            # Handle short videos by repeating frames
            frame_indices = np.linspace(0, total_frames - 1, self.clip_len).astype(int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize to target resolution
                frame = cv2.resize(frame, (self.spatial_size, self.spatial_size))
                frames.append(frame)
            else:
                # If frame reading fails, duplicate the last frame
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    # Create a black frame if no frames were read
                    black_frame = np.zeros((self.spatial_size, self.spatial_size, 3), dtype=np.uint8)
                    frames.append(black_frame)
        
        cap.release()
        
        # Ensure we have the right number of frames
        while len(frames) < self.clip_len:
            frames.append(frames[-1].copy())
        
        return np.array(frames[:self.clip_len])  # [T, H, W, C]
    
    def _compute_optical_flow(self, frames: np.ndarray) -> np.ndarray:
        """Compute optical flow between consecutive frames using Farneback method"""
        flow_frames = []
        
        for i in range(len(frames) - 1):
            try:
                # Convert to grayscale
                gray1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
                gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
                
                # Compute dense optical flow using Farneback method
                flow = cv2.calcOpticalFlowFarneback(
                    gray1, gray2, 
                    None,
                    pyr_scale=0.5, 
                    levels=3, 
                    winsize=15, 
                    iterations=3, 
                    poly_n=5, 
                    poly_sigma=1.2, 
                    flags=0
                )
                
                # Extract magnitude and angle
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                
                # Create HSV representation
                hsv = np.zeros((self.spatial_size, self.spatial_size, 3), dtype=np.uint8)
                hsv[..., 0] = angle * 180 / np.pi / 2  # Hue represents direction
                hsv[..., 1] = 255  # Full saturation
                hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value represents magnitude
                
                # Convert HSV to RGB
                flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                flow_frames.append(flow_rgb)
                
            except Exception as e:
                print(f"Optical flow computation failed: {e}")
                # Create zero flow as fallback
                flow_frames.append(np.zeros_like(frames[i]))
        
        # For the last frame, duplicate the previous flow
        if flow_frames:
            flow_frames.append(flow_frames[-1].copy())
        else:
            # If no flows computed, create zero flows for all frames
            flow_frames = [np.zeros_like(frame) for frame in frames]
        
        return np.array(flow_frames)  # [T, H, W, C]
    
    def _apply_transforms(self, frames: np.ndarray) -> np.ndarray:
        """Apply albumentations transforms to frames"""
        if self.transform is None:
            return frames
        
        transformed_frames = []
        for frame in frames:
            transformed = self.transform(image=frame)
            transformed_frames.append(transformed['image'])
        
        return np.array(transformed_frames)
    
    def _normalize_frames(self, frames: np.ndarray) -> torch.Tensor:
        """Normalize frames and convert to tensor"""
        # Convert to float32 and normalize to [0, 1]
        frames = frames.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization (ensure float32)
        mean = np.array(self.mean, dtype=np.float32)
        std = np.array(self.std, dtype=np.float32)
        frames = (frames - mean) / std
        
        # Convert to tensor [T, H, W, C] -> [C, T, H, W]
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2)
        
        # Ensure tensor is float32 (compatible with mixed precision)
        frames = frames.float()
        
        return frames
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        """Get a video clip and its label"""
        for _ in range(self.num_retries):
            try:
                video_path = self.video_paths[idx]
                label = self.labels[idx]
                
                # Extract frames
                frames = self._extract_frames(video_path)  # [T, H, W, C]
                
                # Apply spatial transforms
                frames = self._apply_transforms(frames)
                
                # Prepare output
                output = {}
                
                # Main RGB frames
                rgb_frames = self._normalize_frames(frames)  # [C, T, H, W]
                output['rgb'] = rgb_frames
                
                # Optical flow frames (motion enhancement)
                if self.compute_optical_flow:
                    flow_frames = self._compute_optical_flow(frames)  # [T, H, W, C]
                    flow_tensor = self._normalize_frames(flow_frames)  # [C, T, H, W]
                    output['flow'] = flow_tensor
                
                return output, label
                
            except Exception as e:
                print(f"Error loading video {self.video_paths[idx]}: {e}")
                # Try a different video
                idx = random.randint(0, len(self.video_paths) - 1)
        
        # If all retries failed, return a dummy sample
        print(f"Failed to load any video after {self.num_retries} retries. Returning dummy sample.")
        
        # Create dummy data
        dummy_rgb = torch.zeros((3, self.clip_len, self.spatial_size, self.spatial_size))
        dummy_output = {'rgb': dummy_rgb}
        
        if self.compute_optical_flow:
            dummy_flow = torch.zeros((3, self.clip_len, self.spatial_size, self.spatial_size))
            dummy_output['flow'] = dummy_flow
            
        return dummy_output, 0


def get_transforms(split: str = "train", spatial_size: int = 224):
    """Get augmentation transforms for training or validation"""
    
    if split == "train":
        # Training transforms with augmentation (compatible with albumentations 0.4.6)
        available_transforms = []
        
        # Basic geometric transforms
        available_transforms.append(A.HorizontalFlip(p=0.5))
        
        # Try to add transforms that exist in 0.4.6
        try:
            available_transforms.append(A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1, 
                rotate_limit=10,
                p=0.5
            ))
        except:
            pass
        
        # Color transforms (using older albumentations API)
        try:
            available_transforms.extend([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.5)
            ])
        except:
            # Fallback for very old versions
            pass
        
        # Blur transforms
        blur_transforms = []
        try:
            blur_transforms.append(A.GaussianBlur(blur_limit=(3, 7), p=1.0))
        except:
            pass
        try:
            blur_transforms.append(A.MedianBlur(blur_limit=5, p=1.0))
        except:
            pass
        try:
            blur_transforms.append(A.MotionBlur(blur_limit=5, p=1.0))
        except:
            pass
        
        if blur_transforms:
            available_transforms.append(A.OneOf(blur_transforms, p=0.3))
        
        # Noise transforms
        noise_transforms = []
        try:
            noise_transforms.append(A.GaussNoise(var_limit=(10, 50), p=1.0))
        except:
            pass
        try:
            noise_transforms.append(A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0))
        except:
            pass
        
        if noise_transforms:
            available_transforms.append(A.OneOf(noise_transforms, p=0.2))
        
        transform = A.Compose(available_transforms)
    else:
        # Validation transforms (no augmentation)
        transform = A.Compose([])
    
    return transform


def create_dataloaders(
    dataset_path: str,
    batch_size: int = 8,
    num_workers: int = 4,
    clip_len: int = 16,
    spatial_size: int = 224,
    max_videos_per_class: Optional[int] = None
):
    """Create train and validation dataloaders"""
    
    # Create datasets
    train_dataset = RWF2000X3DDataset(
        dataset_path=dataset_path,
        split="train",
        clip_len=clip_len,
        spatial_size=spatial_size,
        transform=get_transforms("train", spatial_size),
        compute_optical_flow=True,
        max_videos_per_class=max_videos_per_class
    )
    
    val_dataset = RWF2000X3DDataset(
        dataset_path=dataset_path,
        split="val",
        clip_len=clip_len,
        spatial_size=spatial_size,
        transform=get_transforms("val", spatial_size),
        compute_optical_flow=True,
        max_videos_per_class=max_videos_per_class
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset, val_dataset


if __name__ == "__main__":
    # Test the dataset
    dataset_path = r"D:\Thaman\archive\RWF-2000"
    
    train_loader, val_loader, train_dataset, val_dataset = create_dataloaders(
        dataset_path=dataset_path,
        batch_size=2,
        num_workers=0,  # Set to 0 for testing
        max_videos_per_class=5  # Test with small subset
    )
    
    print("Testing dataset loading...")
    for i, (data, labels) in enumerate(train_loader):
        print(f"Batch {i+1}:")
        print(f"  RGB shape: {data['rgb'].shape}")
        if 'flow' in data:
            print(f"  Flow shape: {data['flow'].shape}")
        print(f"  Labels: {labels}")
        
        if i >= 2:  # Test first 3 batches
            break
    
    print("Dataset test completed successfully!")