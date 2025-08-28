import os
import cv2
import numpy as np
import time
import torch
from pathlib import Path

# Import the model definition
from model import X3DViolenceDetector, create_model

# Constants for X3D model - MATCH TRAINING SETTINGS
NUM_FRAMES = 16       # X3D works with 16 frames
INPUT_SIZE = 336      # X3D uses 336x336 input (matching training)
SAMPLING_RATE = 4     # Temporal sampling rate

def load_violence_detection_model(model_path, device=None):
    """
    Load the trained PyTorch X3D violence detection model.
    
    Args:
        model_path: Path to the .pth model file
        device: Device to run the model on ('cuda' or 'cpu')
        
    Returns:
        model: The loaded PyTorch model
        use_gpu: Whether GPU is being used
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Determine device with compatibility check
    if device is None:
        if torch.cuda.is_available():
            try:
                # Test CUDA compatibility
                test_tensor = torch.zeros(1, device='cuda')
                test_tensor = test_tensor + 1
                device = torch.device('cuda')
                print("Using CUDA")
            except Exception as e:
                print(f"CUDA incompatible ({e}), using CPU")
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')
            print("CUDA not available, using CPU")
    else:
        device = torch.device(device)
    
    use_gpu = device.type == 'cuda'
    print(f"Using device: {device}")
    
    try:
        print(f"Loading trained X3D model from {model_path}...")
        
        # Load the saved checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Determine model architecture from checkpoint or use x3d_m (training default)
        model_name = "x3d_m"  # Match training script default
        
        # Try to extract model info from checkpoint metadata
        if isinstance(checkpoint, dict):
            if 'model_info' in checkpoint:
                model_name = checkpoint['model_info'].get('architecture', 'x3d_m')
            elif 'history' in checkpoint and checkpoint['history']:
                # Check if we can infer from history
                print("Checkpoint contains training history - using x3d_m")
        
        print(f"Creating model with architecture: {model_name}")
        
        # Create model instance with matching architecture
        model = X3DViolenceDetector(
            x3d_model_name=model_name,
            num_classes=2,
            use_motion_enhancement=True,  # Match training
            dropout_rate=0.2,  # Match training
            motion_weight=0.3,   # Match training
            device=device.type   # Pass the detected device type
        )
        
        # Rest of the function remains the same...
        # Load state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("Loading from checkpoint with model_state_dict")
            
            # Try strict loading first
            try:
                model.load_state_dict(state_dict, strict=True)
                print("✅ Loaded model with strict=True")
            except Exception as e:
                print(f"Strict loading failed: {e}")
                print("Trying with strict=False...")
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                print(f"Missing keys: {missing_keys}")
                print(f"Unexpected keys: {unexpected_keys}")
                print("⚠️ Loaded model with strict=False")
                
        elif isinstance(checkpoint, dict):
            # Direct state dict
            print("Loading from raw state dict")
            try:
                model.load_state_dict(checkpoint, strict=True)
                print("✅ Loaded model with strict=True")
            except Exception as e:
                print(f"Strict loading failed: {e}")
                model.load_state_dict(checkpoint, strict=False)
                print("⚠️ Loaded model with strict=False")
        else:
            # Complete model object
            print("Loading complete model object")
            model = checkpoint
        
        # Move model to device and set to eval mode
        model = model.to(device)
        model.eval()
        
        print("Trained X3D model loaded successfully and ready for inference.")
        
        # Rest remains the same (warm-up prediction)...
        print("Running model warm-up prediction...")
        dummy_rgb = torch.zeros((1, 3, NUM_FRAMES, INPUT_SIZE, INPUT_SIZE), dtype=torch.float32, device=device)
        dummy_data = {'rgb': dummy_rgb}
        
        # Add optical flow if model uses motion enhancement
        if hasattr(model, 'use_motion_enhancement') and model.use_motion_enhancement:
            dummy_flow = torch.zeros((1, 3, NUM_FRAMES, INPUT_SIZE, INPUT_SIZE), dtype=torch.float32, device=device)
            dummy_data['flow'] = dummy_flow
        
        with torch.no_grad():
            output = model(dummy_data)
            print(f"Warm-up output shape: {output.shape}")
            print(f"Warm-up output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
            
            # Check for extreme logits (sign of training instability)
            logit_range = output.max().item() - output.min().item()
            if logit_range > 50:
                print("⚠️ WARNING: Large logit range detected - model may have training issues")
            elif abs(output.max().item()) > 20:
                print("⚠️ WARNING: Extreme logit values detected")
            else:
                print("✅ Model producing reasonable logit values")
        
        return model, use_gpu
        
    except Exception as e:
        print(f"Error loading trained X3D model: {e}")
        import traceback
        traceback.print_exc()
        raise

def extract_frames(video_path, num_frames=NUM_FRAMES, sampling_rate=SAMPLING_RATE):
    """Extract frames from a video file optimized for X3D - MATCH TRAINING PIPELINE"""
    print(f"Extracting {num_frames} frames from: {video_path}")
    
    # Try with FFMPEG backend explicitly
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        # If FFMPEG fails, try the default backend
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video has {total_frames} frames at {fps} FPS")
    
    # Calculate frame indices for temporal sampling (MATCH TRAINING DATASET LOGIC)
    required_frames = num_frames * sampling_rate
    
    if total_frames >= required_frames:
        # Uniform sampling from video (random start during training, center for inference)
        start_idx = max(0, (total_frames - required_frames) // 2)  # Center sampling for consistent inference
        frame_indices = np.arange(start_idx, start_idx + required_frames, sampling_rate)
    else:
        # Handle short videos by repeating frames (MATCH TRAINING)
        if total_frames > 0:
            frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
        else:
            frame_indices = [0] * num_frames
    
    # Extract frames (MATCH TRAINING FORMAT)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, total_frames - 1))
        ret, frame = cap.read()
        if ret:
            # Resize frame to required input size (MATCH TRAINING)
            frame = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
            # Convert BGR to RGB (MATCH TRAINING - model expects RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            print(f"Failed to read frame {idx}")
            # Add a blank frame if read fails (MATCH TRAINING FALLBACK)
            frames.append(np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8))
    
    cap.release()
    
    # Ensure we have the right number of frames (MATCH TRAINING)
    while len(frames) < num_frames:
        frames.append(frames[-1].copy() if frames else np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8))
    
    # Convert to numpy array [T, H, W, C] (MATCH TRAINING OUTPUT)
    frames = np.array(frames[:num_frames])
    print(f"Extracted {len(frames)} frames of shape {frames.shape[1:]}")
    
    return frames

def compute_optical_flow(frames):
    """Compute optical flow between consecutive frames - MATCH TRAINING EXACTLY"""
    
    # Handle empty frames case early
    if len(frames) == 0:
        return np.array([], dtype=np.uint8).reshape(0, INPUT_SIZE, INPUT_SIZE, 3)
    
    flow_frames = []
    
    for i in range(len(frames) - 1):
        try:
            # Convert to grayscale (MATCH TRAINING)
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
            
            # Compute dense optical flow using Farneback method (MATCH TRAINING PARAMETERS)
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
            
            # Extract magnitude and angle (MATCH TRAINING)
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Create HSV representation (MATCH TRAINING)
            hsv = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
            hsv[..., 0] = angle * 180 / np.pi / 2  # Hue represents direction
            hsv[..., 1] = 255  # Full saturation
            hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value represents magnitude
            
            # Convert HSV to RGB (MATCH TRAINING)
            flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            flow_frames.append(flow_rgb)
            
        except Exception as e:
            print(f"Optical flow computation failed: {e}")
            # Create zero flow as fallback (MATCH TRAINING)
            flow_frames.append(np.zeros_like(frames[i]))
    
    # For the last frame, duplicate the previous flow (MATCH TRAINING)
    if flow_frames:
        flow_frames.append(flow_frames[-1].copy())
    else:
        # If no flows computed, create zero flows for all frames (MATCH TRAINING)
        flow_frames = [np.zeros_like(frame) for frame in frames]
    
    return np.array(flow_frames)

def preprocess_frames(frames, compute_flow=True):
    """
    Preprocess frames for X3D model input - MATCH TRAINING PREPROCESSING EXACTLY
    
    Args:
        frames: numpy array of frames [T, H, W, C] in RGB format, uint8
        compute_flow: whether to compute optical flow
        
    Returns:
        data: dictionary with 'rgb' and optionally 'flow' tensors
    """
    # ImageNet normalization values (MATCH TRAINING - X3D uses ImageNet pretrained weights)
    mean = np.array([0.45, 0.45, 0.45]).reshape(1, 1, 1, 3)
    std = np.array([0.225, 0.225, 0.225]).reshape(1, 1, 1, 3)
    
    # Normalize RGB frames (MATCH TRAINING PIPELINE)
    rgb_frames = frames.astype(np.float32) / 255.0
    rgb_frames = (rgb_frames - mean) / std
    
    # Convert to tensor and reorder dimensions (MATCH TRAINING)
    # From [T, H, W, C] to [C, T, H, W] which is what PyTorch expects
    rgb_tensor = torch.from_numpy(rgb_frames).float()
    rgb_tensor = rgb_tensor.permute(3, 0, 1, 2)
    
    # Prepare output data
    data = {'rgb': rgb_tensor}
    
    # Compute optical flow if requested (MATCH TRAINING)
    if compute_flow:
        try:
            flow_frames = compute_optical_flow(frames)
            
            # Normalize flow frames (MATCH TRAINING)
            flow_frames = flow_frames.astype(np.float32) / 255.0
            flow_frames = (flow_frames - mean) / std
            
            # Convert to tensor (MATCH TRAINING)
            flow_tensor = torch.from_numpy(flow_frames).float()
            flow_tensor = flow_tensor.permute(3, 0, 1, 2)
            
            data['flow'] = flow_tensor
            
        except Exception as e:
            print(f"Failed to compute optical flow: {e}")
            # Create dummy flow tensor if computation fails (MATCH TRAINING FALLBACK)
            data['flow'] = torch.zeros_like(rgb_tensor)
    
    return data

def predict_violence(model, data, threshold=0.5, debug=False, device=None):
    """
    Make a violence prediction - ENHANCED WITH TRAINING INSIGHTS
    
    Args:
        model: PyTorch X3D model
        data: Dictionary with 'rgb' and optionally 'flow' tensors [C, T, H, W]
        threshold: Detection threshold
        debug: Whether to print debug information
        device: Device to run on ('cuda' or 'cpu')
        
    Returns:
        is_fight: Whether violence is detected
        fight_prob: Confidence score
        inference_time: Time taken for inference
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Add batch dimension if not already present
    for key in data:
        if len(data[key].shape) == 4:  # [C, T, H, W]
            data[key] = data[key].unsqueeze(0)  # [1, C, T, H, W]
    
    # Move to device and ensure float32 precision
    for key in data:
        data[key] = data[key].to(device).float()
    
    if debug:
        for key, tensor in data.items():
            print(f"{key} tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
            print(f"{key} tensor range: [{tensor.min().item():.3f}, {tensor.max().item():.3f}]")
        try:
            print(f"Model weight type: {next(model.parameters()).dtype}")
        except (StopIteration, AttributeError):
            print("Could not access model parameters")
        print(f"Detection threshold: {threshold}")
    
    # Run prediction with timing
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model(data)
    
    end_time = time.time()
    inference_time = end_time - start_time
    
    # Ensure minimum inference time for testing purposes
    if inference_time <= 0:
        inference_time = 0.001  # 1ms minimum
    
    # Convert output to numpy for processing
    outputs_np = outputs.cpu().numpy()
    
    # STABILITY CHECK: Check for extreme logits
    logit_range = outputs_np.max() - outputs_np.min()
    if debug:
        print(f"Raw logits: {outputs_np[0]}")
        print(f"Logit range: {logit_range:.2f}")
        print(f"Inference time: {inference_time:.3f} seconds")
    
    if logit_range > 50:
        print(f"⚠️ WARNING: Extreme logit range detected: {logit_range:.2f}")
    
    # Process predictions - MATCH TRAINING LABEL MAPPING
    if outputs_np.shape[1] >= 2:
        # Two-class output: [NonFight_logit, Fight_logit]
        logit_0 = float(outputs_np[0][0])  # NonFight logit
        logit_1 = float(outputs_np[0][1])  # Fight logit
        
        if debug:
            print(f"NonFight logit: {logit_0:.4f}")
            print(f"Fight logit: {logit_1:.4f}")
        
        # Apply softmax to get proper probabilities
        exp_logits = np.exp(outputs_np[0] - np.max(outputs_np[0]))  # Numerical stability
        probs = exp_logits / np.sum(exp_logits)
        prob_0, prob_1 = float(probs[0]), float(probs[1])
        
        if debug:
            print(f"After softmax - NonFight prob: {prob_0:.4f}, Fight prob: {prob_1:.4f}")
        
        # TRAINING LABEL MAPPING:
        # Label 0 = NonFight → model outputs higher prob_0 for NonFight
        # Label 1 = Fight → model outputs higher prob_1 for Fight
        non_fight_prob = prob_0
        fight_prob = prob_1
        
        if debug:
            print(f"FINAL: NonFight confidence: {non_fight_prob:.4f}, Fight confidence: {fight_prob:.4f}")
            print(f"Prediction: {'FIGHT' if fight_prob > threshold else 'NON-FIGHT'}")
        
        # STABILITY CHECK: Warn about overconfident predictions
        if fight_prob > 0.999 or fight_prob < 0.001:
            if debug:
                print(f"⚠️ WARNING: Overconfident prediction ({fight_prob:.4f})")
        
        # Final decision
        is_fight = fight_prob > threshold
        
        return is_fight, fight_prob, inference_time
        
    else:
        # Single output model (shouldn't happen with standard training)
        if debug:
            print("Single output detected - unexpected for two-class model")
        
        fight_prob = float(outputs_np[0][0])
        
        # Apply sigmoid if output looks like a logit
        if abs(fight_prob) > 5:
            fight_prob = 1.0 / (1.0 + np.exp(-fight_prob))
        
        is_fight = fight_prob > threshold
        
        return is_fight, fight_prob, inference_time

def test_detection(model_path, video_path, threshold=0.5):
    """Test the trained violence detection on a single video"""
    print("="*60)
    print("TRAINED X3D VIOLENCE DETECTION TEST")
    print("="*60)
    
    # Load trained model
    model, use_gpu = load_violence_detection_model(model_path)
    
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Using GPU: {use_gpu}")
    print(f"Detection threshold: {threshold}")
    print("-"*60)
    
    # Extract and preprocess frames
    frames = extract_frames(video_path)
    
    # Determine if model uses motion enhancement
    use_motion = hasattr(model, 'use_motion_enhancement') and model.use_motion_enhancement
    print(f"Motion enhancement: {use_motion}")
    
    # Preprocess with or without optical flow
    processed_data = preprocess_frames(frames, compute_flow=use_motion)
    
    print(f"Preprocessed data shapes:")
    for key, tensor in processed_data.items():
        print(f"  {key}: {tensor.shape}")
    
    print("-"*60)
    
    # Make prediction with full debugging
    is_fight, confidence, inference_time = predict_violence(
        model, processed_data, threshold, debug=True
    )
    
    # Print results
    print("="*60)
    result = "🚨 VIOLENCE DETECTED" if is_fight else "✅ NO VIOLENCE DETECTED"
    print(f"RESULT: {result}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Inference time: {inference_time:.3f}s")
    print("="*60)
    
    # Additional analysis
    if confidence > 0.95:
        print("📊 ANALYSIS: Very high confidence - strong signal")
    elif confidence > threshold + 0.1:
        print("📊 ANALYSIS: Good confidence - reliable detection")
    elif is_fight and confidence < threshold + 0.05:
        print("📊 ANALYSIS: Borderline detection - review recommended")
    else:
        print("📊 ANALYSIS: Clear non-violence or confident non-detection")
    
    if inference_time > 1.0:
        print("⏱️ PERFORMANCE: Slow inference - check GPU utilization")
    else:
        print("⏱️ PERFORMANCE: Good inference speed")
    
    return is_fight, confidence, inference_time

def extract_consecutive_frame_sequences(video_path, sequence_length=16, hop_seconds=2.0):
    """
    Extract consecutive frame sequences from video for proper temporal analysis.
    Mimics RTSPStreamProcessor approach for uploaded videos.
    """
    print(f"Extracting consecutive frame sequences from: {video_path}")
    
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s")
    
    # Calculate hop in frames
    hop_frames = int(hop_seconds * fps)
    sequences = []
    timestamps = []
    
    start_frame = 0
    while start_frame + sequence_length <= total_frames:
        sequence_frames = []
        
        for frame_idx in range(start_frame, start_frame + sequence_length):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frame = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sequence_frames.append(frame)
            else:
                if sequence_frames:
                    sequence_frames.append(sequence_frames[-1].copy())
                else:
                    sequence_frames.append(np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8))
        
        # Ensure exact length
        while len(sequence_frames) < sequence_length:
            sequence_frames.append(sequence_frames[-1].copy() if sequence_frames else 
                                   np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8))
        
        sequence_array = np.array(sequence_frames[:sequence_length])
        sequences.append(sequence_array)
        
        start_time = start_frame / fps
        end_time = (start_frame + sequence_length) / fps
        timestamps.append((start_time, end_time))
        
        start_frame += hop_frames
    
    cap.release()
    print(f"Extracted {len(sequences)} consecutive sequences")
    return sequences, timestamps

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained X3D violence detection on a video')
    parser.add_argument('--model', type=str, default='stable_checkpoints/stable_best_model.pth', 
                       help='Path to trained model file')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')
    
    args = parser.parse_args()
    
    test_detection(args.model, args.video, args.threshold)