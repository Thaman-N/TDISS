import os
import cv2
import numpy as np
import time
import torch
from pathlib import Path

# Import the FIXED X3D model definition
from model import X3DViolenceDetector, create_model

# Constants for X3D model
NUM_FRAMES = 16       # X3D works with 16 frames
INPUT_SIZE = 224      # X3D uses 224x224 input
SAMPLING_RATE = 4     # Temporal sampling rate

def load_violence_detection_model(model_path, device=None):
    """
    Load the FIXED PyTorch X3D violence detection model.
    
    Args:
        model_path: Path to the .pth model file
        device: Device to run the model on ('cuda' or 'cpu')
        
    Returns:
        model: The loaded PyTorch model
        use_gpu: Whether GPU is being used
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Determine device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    use_gpu = device.type == 'cuda'
    print(f"Using device: {device}")
    
    try:
        print(f"Loading STABLE X3D model from {model_path}...")
        
        # Load the saved object
        saved_object = torch.load(model_path, map_location=device, weights_only=False)
        
        # Create STABLE model instance
        model = X3DViolenceDetector(
            x3d_model_name="x3d_s",
            num_classes=2,
            use_motion_enhancement=True,
            dropout_rate=0.2,  # Match training settings
            motion_weight=0.3
        )
        
        # Load state dict
        if isinstance(saved_object, dict) and 'model_state_dict' in saved_object:
            state_dict = saved_object['model_state_dict']
            print("Loading from checkpoint with model_state_dict")
        elif isinstance(saved_object, dict):
            state_dict = saved_object
            print("Loading from raw state dict")
        else:
            # It's a complete model object
            model = saved_object
            print("Loading complete model object")
            model = model.to(device)
            model.eval()
            
            # Test model
            print("Running model warm-up prediction...")
            dummy_rgb = torch.zeros((1, 3, NUM_FRAMES, INPUT_SIZE, INPUT_SIZE), dtype=torch.float32, device=device)
            dummy_data = {'rgb': dummy_rgb}
            
            if hasattr(model, 'use_motion_enhancement') and model.use_motion_enhancement:
                dummy_flow = torch.zeros((1, 3, NUM_FRAMES, INPUT_SIZE, INPUT_SIZE), dtype=torch.float32, device=device)
                dummy_data['flow'] = dummy_flow
            
            with torch.no_grad():
                output = model(dummy_data)
                print(f"Warm-up output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
                
                if abs(output.max().item()) > 20:
                    print("⚠️  WARNING: Model still producing extreme logits!")
                else:
                    print("✅ Model producing reasonable logits")
            
            return model, use_gpu
        
        # Load state dict into model
        try:
            model.load_state_dict(state_dict, strict=True)
            print("✅ Loaded model with strict=True")
        except Exception as e:
            print(f"Strict loading failed: {e}")
            print("Trying with strict=False...")
            model.load_state_dict(state_dict, strict=False)
            print("⚠️  Loaded model with strict=False")
        
        # Move model to device and set to eval mode
        model = model.to(device)
        model.eval()
        
        print("STABLE X3D model loaded successfully and ready for inference.")
        
        # Run a test to check for extreme logits
        print("Running model warm-up prediction...")
        dummy_rgb = torch.zeros((1, 3, NUM_FRAMES, INPUT_SIZE, INPUT_SIZE), dtype=torch.float32, device=device)
        dummy_data = {'rgb': dummy_rgb}
        
        # Add optical flow if model uses motion enhancement
        if hasattr(model, 'use_motion_enhancement') and model.use_motion_enhancement:
            dummy_flow = torch.zeros((1, 3, NUM_FRAMES, INPUT_SIZE, INPUT_SIZE), dtype=torch.float32, device=device)
            dummy_data['flow'] = dummy_flow
        
        with torch.no_grad():
            output = model(dummy_data)
            print(f"Warm-up output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
            
            # Check for extreme logits
            if abs(output.max().item()) > 20:
                print("⚠️  WARNING: Model still producing extreme logits!")
                print("This suggests the model needs to be retrained with the stable pipeline.")
            else:
                print("✅ Model producing reasonable logits")
        
        return model, use_gpu
        
    except Exception as e:
        print(f"Error loading STABLE X3D model: {e}")
        import traceback
        traceback.print_exc()
        raise

def extract_frames(video_path, num_frames=NUM_FRAMES, sampling_rate=SAMPLING_RATE):
    """Extract frames from a video file optimized for X3D"""
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
    
    # Calculate frame indices for temporal sampling (similar to training)
    required_frames = num_frames * sampling_rate
    
    if total_frames >= required_frames:
        # Uniform sampling from video
        start_idx = max(0, (total_frames - required_frames) // 2)  # Center sampling
        frame_indices = np.arange(start_idx, start_idx + required_frames, sampling_rate)
    else:
        # Handle short videos by repeating frames
        if total_frames > 0:
            frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
        else:
            frame_indices = [0] * num_frames
    
    # Extract frames
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, total_frames - 1))
        ret, frame = cap.read()
        if ret:
            # Resize frame to required input size
            frame = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
            # Convert BGR to RGB (model expects RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            print(f"Failed to read frame {idx}")
            # Add a blank frame if read fails
            frames.append(np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8))
    
    cap.release()
    
    # Ensure we have the right number of frames
    while len(frames) < num_frames:
        frames.append(frames[-1].copy() if frames else np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8))
    
    # Convert to numpy array
    frames = np.array(frames[:num_frames])
    print(f"Extracted {len(frames)} frames of shape {frames.shape[1:]}")
    
    return frames

def compute_optical_flow(frames):
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
            hsv = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
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
    
    return np.array(flow_frames)

def preprocess_frames(frames, compute_flow=True):
    """
    Preprocess frames for X3D model input.
    
    Args:
        frames: numpy array of frames [T, H, W, C]
        compute_flow: whether to compute optical flow
        
    Returns:
        data: dictionary with 'rgb' and optionally 'flow' tensors
    """
    # ImageNet normalization values (X3D uses ImageNet pretrained weights)
    mean = np.array([0.45, 0.45, 0.45]).reshape(1, 1, 1, 3)
    std = np.array([0.225, 0.225, 0.225]).reshape(1, 1, 1, 3)
    
    # Normalize RGB frames
    rgb_frames = frames.astype(np.float32) / 255.0
    rgb_frames = (rgb_frames - mean) / std
    
    # Convert to tensor and reorder dimensions
    # From [T, H, W, C] to [C, T, H, W] which is what PyTorch expects
    rgb_tensor = torch.from_numpy(rgb_frames).float()
    rgb_tensor = rgb_tensor.permute(3, 0, 1, 2)
    
    # Prepare output data
    data = {'rgb': rgb_tensor}
    
    # Compute optical flow if requested
    if compute_flow:
        try:
            flow_frames = compute_optical_flow(frames)
            
            # Normalize flow frames
            flow_frames = flow_frames.astype(np.float32) / 255.0
            flow_frames = (flow_frames - mean) / std
            
            # Convert to tensor
            flow_tensor = torch.from_numpy(flow_frames).float()
            flow_tensor = flow_tensor.permute(3, 0, 1, 2)
            
            data['flow'] = flow_tensor
            
        except Exception as e:
            print(f"Failed to compute optical flow: {e}")
            # Create dummy flow tensor if computation fails
            data['flow'] = torch.zeros_like(rgb_tensor)
    
    return data

def predict_violence(model, data, threshold=0.5, debug=False, device=None):
    """
    Make a violence prediction on the given data with STABILITY CHECKS.
    
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
        print(f"Model weight type: {next(model.parameters()).dtype}")
    
    # Run prediction with timing
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model(data)
    
    end_time = time.time()
    inference_time = end_time - start_time
    
    # Convert output to numpy for processing
    outputs_np = outputs.cpu().numpy()
    
    # STABILITY CHECK: Check for extreme logits
    logit_range = outputs_np.max() - outputs_np.min()
    if logit_range > 50:
        print(f"⚠️  WARNING: Extreme logit range detected: {logit_range:.2f}")
        print("This suggests the model was not trained with the stable pipeline.")
    
    # Print raw predictions if in debug mode
    if debug:
        print(f"Raw logits: {outputs_np}")
        print(f"Output shape: {outputs_np.shape}")
        print(f"Logit range: {logit_range:.2f}")
        print(f"Inference time: {inference_time:.2f} seconds")
    
    # Process predictions
    if outputs_np.shape[1] >= 2:
        logit_0 = float(outputs_np[0][0])  # First class logit
        logit_1 = float(outputs_np[0][1])  # Second class logit
        
        if debug:
            print(f"Logit[0]: {logit_0:.4f}")
            print(f"Logit[1]: {logit_1:.4f}")
        
        # Apply softmax to get proper probabilities
        import scipy.special
        probs = scipy.special.softmax(outputs_np[0])
        prob_0, prob_1 = float(probs[0]), float(probs[1])
        
        if debug:
            print(f"After softmax - Prob[0]: {prob_0:.4f}, Prob[1]: {prob_1:.4f}")
        
        # Based on training code: 
        # NonFight videos get label 0 -> model outputs higher values at index 0 for NonFight
        # Fight videos get label 1 -> model outputs higher values at index 1 for Fight
        
        # So: outputs[0] = NonFight probability, outputs[1] = Fight probability
        non_fight_prob = prob_0
        fight_prob = prob_1
        
        if debug:
            print(f"INTERPRETATION: NonFight: {non_fight_prob:.4f}, Fight: {fight_prob:.4f}")
            print(f"Current prediction: {'FIGHT' if fight_prob > threshold else 'NON-FIGHT'}")
        
        # STABILITY CHECK: Warn about overconfident predictions
        if fight_prob > 0.99 or fight_prob < 0.01:
            if debug:
                print(f"⚠️  WARNING: Overconfident prediction ({fight_prob:.4f})")
                print("This suggests training instability or extreme model weights.")
        
        # Check if fight probability exceeds threshold
        is_fight = fight_prob > threshold
        
        return is_fight, fight_prob, inference_time
    else:
        # Single output model (shouldn't happen with X3D, but handle it)
        fight_prob = float(outputs_np[0][0])
        
        if debug:
            print(f"Single output value: {fight_prob:.4f}")
        
        # Apply sigmoid if output looks like a logit
        if abs(fight_prob) > 5:  # Likely a logit
            fight_prob = 1.0 / (1.0 + np.exp(-fight_prob))
        
        is_fight = fight_prob > threshold
        
        return is_fight, fight_prob, inference_time
    
# Function to test the STABLE model on a single video
def test_detection(model_path, video_path, threshold=0.5):
    """Test the STABLE violence detection on a single video"""
    # Load STABLE model
    model, _ = load_violence_detection_model(model_path)
    
    # Extract and preprocess frames
    frames = extract_frames(video_path)
    
    # Determine if model uses motion enhancement
    use_motion = hasattr(model, 'use_motion_enhancement') and model.use_motion_enhancement
    
    # Preprocess with or without optical flow
    processed_data = preprocess_frames(frames, compute_flow=use_motion)
    
    # Make prediction
    is_fight, confidence, inference_time = predict_violence(model, processed_data, threshold, True)
    
    # Print results
    result = "VIOLENCE DETECTED" if is_fight else "NO VIOLENCE DETECTED"
    print(f"Result: {result} (Confidence: {confidence:.4f}, Inference time: {inference_time:.2f}s)")
    
    # Additional stability checks
    if confidence > 0.99:
        print("⚠️  WARNING: Overconfident prediction - model may need retraining")
    elif confidence < 0.01:
        print("⚠️  WARNING: Underconfident prediction - model may need retraining")
    else:
        print("✅ Confidence level appears reasonable")
    
    return is_fight, confidence, inference_time

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test STABLE X3D violence detection on a video')
    parser.add_argument('--model', type=str, default='stable_checkpoints/stable_best_model.pth', help='Path to STABLE model file')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--threshold', type=float, default=0.6, help='Detection threshold')
    
    args = parser.parse_args()
    
    print("Testing STABLE X3D Violence Detection")
    print("=====================================")
    test_detection(args.model, args.video, args.threshold)