"""
Extended dataset class that supports different sampling methods for ablation study
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import random
from typing import Tuple, Optional


class AblationFrameSampler:
    """
    Frame sampler that can switch between different sampling strategies
    """
    
    def __init__(self, sampling_method: str = "uniform", clip_len: int = 16):
        self.sampling_method = sampling_method
        self.clip_len = clip_len
        
    def sample_frames(self, total_frames: int, sampling_rate: int = 4) -> list:
        """Sample frames based on the specified method"""
        
        if self.sampling_method == "uniform":
            return self._uniform_sampling(total_frames, sampling_rate)
        elif self.sampling_method == "adaptive":
            return self._adaptive_sampling(total_frames)
        elif self.sampling_method == "intelligent":
            return self._intelligent_sampling(total_frames)
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")
    
    def _uniform_sampling(self, total_frames: int, sampling_rate: int) -> list:
        """
        Uniform sampling - your simple uniform method
        """
        required_frames = self.clip_len * sampling_rate
        
        if total_frames >= required_frames:
            max_start = total_frames - required_frames
            start_idx = random.randint(0, max_start) if max_start > 0 else 0
            frame_indices = list(range(start_idx, start_idx + required_frames, sampling_rate))
        else:
            # Not enough frames, use linear interpolation
            frame_indices = np.linspace(0, total_frames - 1, self.clip_len).astype(int).tolist()
        
        return frame_indices[:self.clip_len]
    
    def _adaptive_sampling(self, total_frames: int) -> list:
        """
        Adaptive sampling - your old adaptive method
        """
        if total_frames >= self.clip_len:
            adaptive_sampling_rate = max(1, total_frames // self.clip_len)
            required_frames = self.clip_len * adaptive_sampling_rate
            
            if total_frames >= required_frames:
                max_start = total_frames - required_frames
                start_idx = random.randint(0, max_start) if max_start > 0 else 0
                frame_indices = list(range(start_idx, start_idx + required_frames, adaptive_sampling_rate))
            else:
                frame_indices = np.linspace(0, total_frames - 1, self.clip_len).astype(int).tolist()
        else:
            frame_indices = list(range(total_frames)) + [total_frames - 1] * (self.clip_len - total_frames)
        
        return frame_indices[:self.clip_len]
    
    def _intelligent_sampling(self, total_frames: int) -> list:
        """
        Intelligent sampling - your breakthrough method
        """
        frames_per_second = total_frames / self.clip_len  # Rough estimate
        
        if total_frames < 80:  # Very short videos - preserve temporal detail
            sampling_rate = 1
        elif frames_per_second > 8:  # Long videos - use adaptive coverage
            sampling_rate = max(1, total_frames // self.clip_len)
        else:  # Medium videos - moderate sampling
            sampling_rate = min(3, max(1, total_frames // 20))
        
        # Now apply the calculated sampling rate
        required_frames = self.clip_len * sampling_rate
        
        if total_frames >= required_frames:
            max_start = total_frames - required_frames
            start_idx = random.randint(0, max_start) if max_start > 0 else 0
            frame_indices = list(range(start_idx, start_idx + required_frames, sampling_rate))
        else:
            frame_indices = np.linspace(0, total_frames - 1, self.clip_len).astype(int).tolist()
        
        return frame_indices[:self.clip_len]


# Add this method to your existing CUENetStyleDataset class
def add_sampling_method_to_dataset():
    """
    Monkey patch to add sampling method switching to existing dataset
    """
    from x3d_dataset import CUENetStyleDataset
    
    def _replace_sampling_method(self, sampling_method: str):
        """Replace the sampling method in the dataset"""
        self.sampling_method = sampling_method
        self.frame_sampler = AblationFrameSampler(sampling_method, self.clip_len)
        print(f"📊 Switched to {sampling_method} frame sampling")
    
    def _extract_frames_with_sampling_method(self, video_path: Path) -> np.ndarray:
        """
        Extract frames using the specified sampling method
        This is a modified version of your original _extract_frames
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Read all frames first
        all_frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_frames.append(frame_rgb)
            frame_count += 1
        
        cap.release()
        
        if not all_frames:
            raise ValueError(f"No frames extracted from {video_path}")
        
        # Use the frame sampler to get indices
        if hasattr(self, 'frame_sampler'):
            frame_indices = self.frame_sampler.sample_frames(len(all_frames))
        else:
            # Fallback to uniform sampling
            frame_indices = list(range(0, min(len(all_frames), self.clip_len)))
        
        # Extract selected frames
        selected_frames = []
        for idx in frame_indices:
            idx = min(idx, len(all_frames) - 1)  # Ensure valid index
            frame = all_frames[idx]
            
            # Apply YOLO cropping if enabled
            if hasattr(self, 'use_cuenet_cropping') and self.use_cuenet_cropping:
                frame = self._apply_cuenet_spatial_cropping_to_frame(frame, video_path)
            
            # Resize frame
            frame_resized = cv2.resize(frame, (self.spatial_size, self.spatial_size))
            selected_frames.append(frame_resized)
        
        # Ensure we have exactly clip_len frames
        while len(selected_frames) < self.clip_len:
            selected_frames.append(selected_frames[-1].copy())
        
        return np.array(selected_frames[:self.clip_len])
    
    # Add methods to the class
    CUENetStyleDataset._replace_sampling_method = _replace_sampling_method
    CUENetStyleDataset._extract_frames_with_sampling = _extract_frames_with_sampling_method


# Function to create a basic dataset without advanced features
def create_basic_dataset(
    dataset_path: str,
    split: str = "train",
    clip_len: int = 16,
    spatial_size: int = 224,
    sampling_method: str = "uniform",
    compute_optical_flow: bool = True,
    max_videos_per_class: Optional[int] = None
):
    """
    Create a basic dataset without YOLO cropping or RandAugment
    but with different sampling methods
    """
    from x3d_dataset import CUENetStyleDataset
    
    # Monkey patch the dataset class
    add_sampling_method_to_dataset()
    
    # Create dataset with basic settings
    dataset = CUENetStyleDataset(
        dataset_path=dataset_path,
        split=split,
        clip_len=clip_len,
        spatial_size=spatial_size,
        compute_optical_flow=compute_optical_flow,
        max_videos_per_class=max_videos_per_class,
        use_cuenet_cropping=False,
        use_randaugment=False,
        cache_yolo_detections=False
    )
    
    # Set the sampling method
    dataset._replace_sampling_method(sampling_method)
    
    # Replace the _extract_frames method to use our sampling
    original_extract_frames = dataset._extract_frames
    def new_extract_frames(video_path):
        try:
            return dataset._extract_frames_with_sampling(video_path)
        except:
            # Fallback to original method
            return original_extract_frames(video_path)
    
    dataset._extract_frames = new_extract_frames
    
    return dataset


if __name__ == "__main__":
    # Test different sampling methods
    sampler = AblationFrameSampler("intelligent", clip_len=16)
    
    test_cases = [50, 80, 120, 200, 300]
    for total_frames in test_cases:
        indices = sampler.sample_frames(total_frames)
        print(f"Frames {total_frames} -> Sampled: {len(indices)} frames, indices: {indices[:5]}...{indices[-5:]}")
