import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from pathlib import Path
from typing import Tuple, Optional, Dict
import warnings
import pickle
import hashlib
warnings.filterwarnings('ignore')

# Import YOLO for CUE-Net spatial cropping
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    # print("✅ YOLO available for CUE-Net spatial cropping")
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLO not available - install with: pip install ultralytics")

class CUENetStyleDataset(Dataset):
    """
    EXACT CUE-Net methodology dataset with YOLO detection caching:
    - RandAugment ONLY (no complex augmentation)
    - YOLO V8 video-level spatial cropping with maximum bounding box
    - 336×336 resolution (CUE-Net setting)
    - Consistent cropping across entire video
    - CACHED YOLO detection results for speed
    """
    
    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        clip_len: int = 16,  # Keep 16 frames for now, CUE-Net uses 64
        spatial_size: int = 336,  # CUE-Net uses 336×336
        sampling_rate: int = 4,
        num_retries: int = 10,
        compute_optical_flow: bool = True,
        max_videos_per_class: Optional[int] = None,
        # CUE-Net settings
        use_cuenet_cropping: bool = True,
        yolo_model_size: str = "yolov8n",
        # RandAugment only - no complex augmentation
        use_randaugment: bool = True,
        randaugment_n: int = 2,  # Number of augmentation operations
        randaugment_m: int = 10,  # Magnitude of augmentations
        # YOLO caching settings
        cache_yolo_detections: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """
        CUE-Net style dataset with their EXACT methodology + YOLO caching
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.clip_len = clip_len
        self.spatial_size = spatial_size
        self.sampling_rate = sampling_rate
        self.num_retries = num_retries
        self.compute_optical_flow = compute_optical_flow
        
        # CUE-Net spatial cropping settings
        self.use_cuenet_cropping = use_cuenet_cropping and YOLO_AVAILABLE and split == "train"
        self.yolo_model_size = yolo_model_size
        
        # YOLO detection caching
        self.cache_yolo_detections = cache_yolo_detections
        self.yolo_cache: Dict[str, list] = {}  # In-memory cache
        
        # Setup cache directory
        if cache_dir is None:
            cache_dir = self.dataset_path / "yolo_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / f"yolo_detections_{yolo_model_size}_{split}.pkl"
        
        # Load YOLO model for person detection (CUE-Net methodology)
        if self.use_cuenet_cropping:
            try:
                self.yolo_model = YOLO(f'{yolo_model_size}.pt')
                print(f"✅ Loaded YOLO {yolo_model_size} for CUE-Net spatial cropping")
                
                # Load existing cache
                if self.cache_yolo_detections:
                    self._load_yolo_cache()
                    
            except Exception as e:
                print(f"⚠️  Failed to load YOLO: {e}")
                self.use_cuenet_cropping = False
        
        # RandAugment settings (CUE-Net approach)
        self.use_randaugment = use_randaugment and split == "train"
        self.randaugment_n = randaugment_n
        self.randaugment_m = randaugment_m
        
        # Load video paths and labels
        self.video_paths, self.labels = self._load_dataset(max_videos_per_class)
        
        # ImageNet normalization
        self.mean = np.array([0.45, 0.45, 0.45])
        self.std = np.array([0.225, 0.225, 0.225])
        
        print(f"Loaded {len(self.video_paths)} videos for {split} split")
        print(f"Fight videos: {sum(self.labels)}, Non-fight videos: {len(self.labels) - sum(self.labels)}")
        
        if split == "train":
            print("CUE-Net methodology enabled:")
            print(f"  🎯 CUE-Net spatial cropping: {self.use_cuenet_cropping}")
            print(f"  🎨 RandAugment only: {self.use_randaugment} (N={randaugment_n}, M={randaugment_m})")
            print(f"  📏 Resolution: {spatial_size}×{spatial_size} (CUE-Net: 336×336)")
            print(f"  🚫 NO complex augmentations (following CUE-Net paper)")
            
            if self.cache_yolo_detections:
                print(f"  💾 YOLO detection caching: enabled")
                print(f"  📁 Cache directory: {self.cache_dir}")
                print(f"  🗃️  Cached detections: {len(self.yolo_cache)}")
    
    def _get_video_hash(self, video_path: Path) -> str:
        """Generate a hash for video file to ensure cache validity"""
        # Use file path + size + modification time for hash
        stat = video_path.stat()
        hash_input = f"{video_path}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _load_yolo_cache(self):
        """Load YOLO detection cache from disk"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    self.yolo_cache = pickle.load(f)
                print(f"📁 Loaded YOLO cache with {len(self.yolo_cache)} entries")
            else:
                self.yolo_cache = {}
                print("📁 No existing YOLO cache found, starting fresh")
        except Exception as e:
            print(f"⚠️  Failed to load YOLO cache: {e}")
            self.yolo_cache = {}
    
    def _save_yolo_cache(self):
        """Save YOLO detection cache to disk"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.yolo_cache, f)
            print(f"💾 Saved YOLO cache with {len(self.yolo_cache)} entries")
        except Exception as e:
            print(f"⚠️  Failed to save YOLO cache: {e}")
    
    def _load_dataset(self, max_videos_per_class: Optional[int]) -> Tuple[list, list]:
        """Load video paths and labels"""
        video_paths = []
        labels = []
        
        split_dir = self.dataset_path / self.split
        
        # Load Fight videos (label = 1)
        fight_dir = split_dir / "Fight"
        if fight_dir.exists():
            fight_videos = list(fight_dir.glob("*.avi")) + list(fight_dir.glob("*.mp4")) + list(fight_dir.glob("*.mov"))
            if max_videos_per_class:
                fight_videos = fight_videos[:max_videos_per_class]
            video_paths.extend(fight_videos)
            labels.extend([1] * len(fight_videos))
        
        # Load NonFight videos (label = 0)
        nonfight_dir = split_dir / "NonFight"
        if nonfight_dir.exists():
            nonfight_videos = list(nonfight_dir.glob("*.avi")) + list(nonfight_dir.glob("*.mp4")) + list(nonfight_dir.glob("*.mov"))
            if max_videos_per_class:
                nonfight_videos = nonfight_videos[:max_videos_per_class]
            video_paths.extend(nonfight_videos)
            labels.extend([0] * len(nonfight_videos))
        
        # Shuffle the data
        combined = list(zip(video_paths, labels))
        random.shuffle(combined)
        video_paths, labels = zip(*combined)
        
        return list(video_paths), list(labels)
    
    def _get_cached_detections(self, video_path: Path) -> Optional[list]:
        """Get cached YOLO detections for a video"""
        if not self.cache_yolo_detections:
            return None
        
        video_hash = self._get_video_hash(video_path)
        cache_key = f"{video_path.name}_{video_hash}"
        
        return self.yolo_cache.get(cache_key, None)
    
    def _cache_detections(self, video_path: Path, detections: list):
        """Cache YOLO detections for a video"""
        if not self.cache_yolo_detections:
            return
        
        video_hash = self._get_video_hash(video_path)
        cache_key = f"{video_path.name}_{video_hash}"
        
        self.yolo_cache[cache_key] = detections
    
    def _cuenet_spatial_cropping(self, frames: np.ndarray, video_path: Path) -> np.ndarray:
        """
        EXACT CUE-Net spatial cropping methodology with YOLO caching:
        - Check cache first for existing detections
        - Detect people in ALL frames using YOLO V8 (if not cached)
        - Calculate MAXIMUM bounding box across ENTIRE video
        - Apply SAME crop to all frames (consistent spatial attention)
        - Cache results for future epochs
        """
        if not self.use_cuenet_cropping:
            return frames
        
        T, H, W, C = frames.shape
        
        # Check cache first
        cached_detections = self._get_cached_detections(video_path)
        
        if cached_detections is not None:
            # Use cached detections
            all_detections = cached_detections
            # print(f"🗃️  Using cached YOLO detections for {video_path.name}")
        else:
            # Run YOLO detection and cache results
            all_detections = []
            
            try:
                # CUE-Net: Process every frame to find people
                for frame in frames:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # YOLO V8 detection
                    results = self.yolo_model(
                        frame_bgr, 
                        verbose=False, 
                        conf=0.5,  # Person detection confidence
                        classes=[0]  # Only detect persons (class 0)
                    )
                    
                    # Extract person bounding boxes
                    for result in results:
                        if result.boxes is not None:
                            boxes = result.boxes.xyxy.cpu().numpy()
                            classes = result.boxes.cls.cpu().numpy()
                            confidences = result.boxes.conf.cpu().numpy()
                            
                            # Filter for persons only
                            person_mask = (classes == 0) & (confidences > 0.5)
                            person_boxes = boxes[person_mask]
                            
                            if len(person_boxes) > 0:
                                all_detections.extend(person_boxes.tolist())
                
                # Cache the detection results
                self._cache_detections(video_path, all_detections)
                # print(f"💾 Cached YOLO detections for {video_path.name}")
                if len(self.yolo_cache) % 50 == 0:
                    self._save_yolo_cache()
                
            except Exception as e:
                print(f"⚠️ YOLO detection failed for {video_path.name}: {e}")
                return frames
        
        # CUE-Net: Calculate maximum bounding box across entire video
        if len(all_detections) > 0:
            all_detections = np.array(all_detections)
            
            # Find maximum bounding box that encompasses all people
            x_min = int(max(0, np.min(all_detections[:, 0])))
            y_min = int(max(0, np.min(all_detections[:, 1])))
            x_max = int(min(W, np.max(all_detections[:, 2])))
            y_max = int(min(H, np.max(all_detections[:, 3])))
            
            crop_w = x_max - x_min
            crop_h = y_max - y_min
            
            # Apply consistent crop to ALL frames
            if crop_w > 0 and crop_h > 0:
                cropped_frames = []
                for frame in frames:
                    cropped_frame = frame[y_min:y_max, x_min:x_max]
                    resized_frame = cv2.resize(cropped_frame, (W, H))
                    cropped_frames.append(resized_frame)
                
                # print(f"✅ CUE-Net crop applied: ({x_min},{y_min})-({x_max},{y_max})")
                return np.array(cropped_frames)
        
        # If no people detected, return original frames
        return frames
    
    def _extract_frames(self, video_path: Path) -> np.ndarray:
        """Extract frames with simple uniform sampling"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                raise ValueError(f"Video has 0 frames: {video_path}")
            
            # Read all frames
            all_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Convert and resize to CUE-Net resolution
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.spatial_size, self.spatial_size))
                all_frames.append(frame)
            
            if len(all_frames) == 0:
                raise ValueError(f"No frames extracted from video: {video_path}")
            
            # Adaptive sampling to maximize video coverage
            total_extracted = len(all_frames)

            # Calculate adaptive sampling rate based on video length
            if total_extracted >= self.clip_len:
                adaptive_sampling_rate = max(1, total_extracted // self.clip_len)
                required_frames = self.clip_len * adaptive_sampling_rate
                
                if total_extracted >= required_frames:
                    max_start = total_extracted - required_frames
                    start_idx = random.randint(0, max_start) if max_start > 0 else 0
                    frame_indices = list(range(start_idx, start_idx + required_frames, adaptive_sampling_rate))
                else:
                    frame_indices = np.linspace(0, total_extracted - 1, self.clip_len).astype(int).tolist()
            else:
                frame_indices = np.linspace(0, total_extracted - 1, self.clip_len).astype(int).tolist()
            
            # Extract the sampled frames
            sampled_frames = []
            for idx in frame_indices:
                if idx < len(all_frames):
                    sampled_frames.append(all_frames[idx])
                else:
                    sampled_frames.append(all_frames[-1])
            
            # Ensure we have exactly clip_len frames
            while len(sampled_frames) < self.clip_len:
                sampled_frames.append(sampled_frames[-1])
            
            frames_array = np.array(sampled_frames[:self.clip_len])  # [T, H, W, C]
            
            # Apply CUE-Net spatial cropping with caching
            frames_array = self._cuenet_spatial_cropping(frames_array, video_path)
            
            return frames_array
            
        except Exception as e:
            raise ValueError(f"Error processing video {video_path}: {str(e)}")
        finally:
            cap.release()
    
    def _apply_randaugment(self, frames: np.ndarray) -> np.ndarray:
        """
        Apply RandAugment as used in CUE-Net paper
        Simple implementation of key RandAugment operations
        """
        if not self.use_randaugment or random.random() > 0.8:
            return frames
        
        augmented_frames = []
        
        # RandAugment: Select N random operations
        operations = [
            'AutoContrast', 'Brightness', 'Color', 'Contrast', 
            'Equalize', 'Rotate', 'Sharpness', 'TranslateX', 'TranslateY'
        ]
        
        # Select N operations randomly
        selected_ops = random.sample(operations, self.randaugment_n)
        
        for frame in frames:
            aug_frame = frame.copy()
            
            for op in selected_ops:
                magnitude = random.uniform(0, self.randaugment_m)
                
                if op == 'Brightness':
                    factor = 1.0 + magnitude * 0.1 * random.choice([-1, 1])
                    aug_frame = np.clip(aug_frame * factor, 0, 255).astype(np.uint8)
                    
                elif op == 'Contrast':
                    factor = 1.0 + magnitude * 0.1 * random.choice([-1, 1])
                    aug_frame = np.clip((aug_frame - 128) * factor + 128, 0, 255).astype(np.uint8)
                    
                elif op == 'Rotate':
                    angle = magnitude * 3 * random.choice([-1, 1])  # Max 30 degrees
                    h, w = aug_frame.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    aug_frame = cv2.warpAffine(aug_frame, M, (w, h), 
                                             borderMode=cv2.BORDER_REFLECT)
                    
                elif op == 'TranslateX':
                    shift = int(magnitude * 0.05 * aug_frame.shape[1] * random.choice([-1, 1]))
                    M = np.float32([[1, 0, shift], [0, 1, 0]])
                    aug_frame = cv2.warpAffine(aug_frame, M, 
                                             (aug_frame.shape[1], aug_frame.shape[0]),
                                             borderMode=cv2.BORDER_REFLECT)
                    
                elif op == 'TranslateY':
                    shift = int(magnitude * 0.05 * aug_frame.shape[0] * random.choice([-1, 1]))
                    M = np.float32([[1, 0, 0], [0, 1, shift]])
                    aug_frame = cv2.warpAffine(aug_frame, M, 
                                             (aug_frame.shape[1], aug_frame.shape[0]),
                                             borderMode=cv2.BORDER_REFLECT)
            
            augmented_frames.append(aug_frame)
        
        return np.array(augmented_frames)
    
    def _compute_optical_flow(self, frames: np.ndarray) -> np.ndarray:
        """Simple optical flow computation"""
        flow_frames = []
        
        for i in range(len(frames) - 1):
            try:
                gray1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
                gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
                
                flow = cv2.calcOpticalFlowFarneback(
                    gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                
                hsv = np.zeros((self.spatial_size, self.spatial_size, 3), dtype=np.uint8)
                hsv[..., 0] = angle * 180 / np.pi / 2
                hsv[..., 1] = 255
                hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
                
                flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                flow_frames.append(flow_rgb)
                
            except Exception:
                flow_frames.append(np.zeros_like(frames[i]))
        
        if flow_frames:
            flow_frames.append(flow_frames[-1].copy())
        else:
            flow_frames = [np.zeros_like(frame) for frame in frames]
        
        return np.array(flow_frames)
    
    def _normalize_frames(self, frames: np.ndarray) -> torch.Tensor:
        """Normalize frames and convert to tensor"""
        frames = frames.astype(np.float32) / 255.0
        
        mean = np.array(self.mean, dtype=np.float32)
        std = np.array(self.std, dtype=np.float32)
        frames = (frames - mean) / std
        
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2)
        frames = frames.float()
        
        return frames
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        """Get video clip with CUE-Net methodology and YOLO caching"""
        for attempt in range(self.num_retries):
            try:
                video_path = self.video_paths[idx]
                label = self.labels[idx]
                
                # Extract frames with CUE-Net spatial cropping (now cached)
                frames = self._extract_frames(video_path)
                
                # Apply RandAugment (CUE-Net approach)
                frames = self._apply_randaugment(frames)
                
                # Compute optical flow
                flow_frames = None
                if self.compute_optical_flow:
                    flow_frames = self._compute_optical_flow(frames)
                
                # Prepare output
                output = {}
                
                # Main RGB frames
                rgb_frames = self._normalize_frames(frames)
                output['rgb'] = rgb_frames
                
                # Optical flow frames
                if self.compute_optical_flow and flow_frames is not None:
                    flow_tensor = self._normalize_frames(flow_frames)
                    output['flow'] = flow_tensor
                else:
                    dummy_flow = torch.zeros_like(rgb_frames)
                    output['flow'] = dummy_flow
                
                return output, label
                
            except Exception as e:
                print(f"Error loading video {self.video_paths[idx]} (attempt {attempt + 1}): {e}")
                if attempt < self.num_retries - 1:
                    idx = random.randint(0, len(self.video_paths) - 1)
        
        # Fallback dummy sample
        print(f"Failed to load any video after {self.num_retries} retries. Returning dummy sample.")
        dummy_rgb = torch.zeros((3, self.clip_len, self.spatial_size, self.spatial_size))
        dummy_output = {'rgb': dummy_rgb}
        
        if self.compute_optical_flow:
            dummy_flow = torch.zeros((3, self.clip_len, self.spatial_size, self.spatial_size))
            dummy_output['flow'] = dummy_flow
            
        return dummy_output, 0
    
    def __del__(self):
        """Save cache when dataset is destroyed"""
        if hasattr(self, 'cache_yolo_detections') and self.cache_yolo_detections:
            if hasattr(self, 'yolo_cache') and len(self.yolo_cache) > 0:
                self._save_yolo_cache()


def create_cuenet_dataloaders(
    dataset_path: str,
    batch_size: int = 8,
    num_workers: int = 4,
    clip_len: int = 16,
    spatial_size: int = 336,  # CUE-Net uses 336×336
    max_videos_per_class: Optional[int] = None,
    cache_yolo_detections: bool = True  # Enable caching by default
):
    """Create dataloaders with CUE-Net methodology and YOLO caching"""
    
    # Create datasets with CUE-Net methodology and caching
    train_dataset = CUENetStyleDataset(
        dataset_path=dataset_path,
        split="train",
        clip_len=clip_len,
        spatial_size=spatial_size,
        compute_optical_flow=True,
        max_videos_per_class=max_videos_per_class,
        # CUE-Net settings
        use_cuenet_cropping=True,
        yolo_model_size="yolov8n",
        use_randaugment=True,
        randaugment_n=2,
        randaugment_m=10,
        # YOLO caching
        cache_yolo_detections=cache_yolo_detections
    )
    
    val_dataset = CUENetStyleDataset(
        dataset_path=dataset_path,
        split="val",
        clip_len=clip_len,
        spatial_size=spatial_size,
        compute_optical_flow=True,
        max_videos_per_class=max_videos_per_class,
        # No augmentation for validation
        use_cuenet_cropping=False,
        use_randaugment=False,
        # No caching needed for validation (no cropping)
        cache_yolo_detections=False
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
    
    print("\n" + "="*60)
    print("🎯 CUE-NET EXACT METHODOLOGY + YOLO CACHING")
    print("="*60)
    print("✅ YOLO V8 video-level spatial cropping")
    print("✅ RandAugment ONLY (N=2, M=10)")
    print("✅ 336×336 resolution (CUE-Net paper setting)")
    print("✅ Consistent spatial attention across video")
    print("✅ YOLO detection caching for speed")
    print("🚫 NO complex augmentations")
    print("🚫 NO ROI crops, motion flips, keyframe focus")
    print("="*60)
    
    return train_loader, val_loader, train_dataset, val_dataset


if __name__ == "__main__":
    # Test the CUE-Net dataset with caching
    dataset_path = r"D:\Thaman\archive\RWF-2000"
    
    print("Testing CUE-Net style dataset with YOLO caching...")
    
    train_loader, val_loader, train_dataset, val_dataset = create_cuenet_dataloaders(
        dataset_path=dataset_path,
        batch_size=2,
        num_workers=0,
        max_videos_per_class=5,
        spatial_size=336,  # CUE-Net resolution
        cache_yolo_detections=True
    )
    
    print("\nTesting CUE-Net dataset loading with caching...")
    for i, (data, labels) in enumerate(train_loader):
        print(f"Batch {i+1}:")
        print(f"  RGB shape: {data['rgb'].shape}")
        print(f"  Flow shape: {data['flow'].shape}")
        print(f"  Labels: {labels}")
        
        if i >= 1:
            break
    
    print("\nCUE-Net dataset test with caching completed!")