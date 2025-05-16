import os
import cv2
import numpy as np
import time
import torch
from pathlib import Path

# Import the CUENet model definition - assuming you have model.py in the same directory
# Only needed when creating a new model - if loading a full model we won't use this
from model import CUENet

# Constants
NUM_FRAMES = 64       # Keep at 64 as required by the model
INPUT_SIZE = 336      # This should match what you trained with (CUE-Net used 336x336)

def load_violence_detection_model(model_path, device=None):
    """
    Load the PyTorch violence detection model.
    
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
        print(f"Loading model from {model_path}...")
        
        # Load the saved object
        saved_object = torch.load(model_path, map_location=device)
        
        # Determine if it's a complete model or a state dict
        if isinstance(saved_object, dict) and 'model_state_dict' in saved_object:
            # Create a new model instance
            model = CUENet(
                num_frames=NUM_FRAMES,
                input_size=INPUT_SIZE,
                embed_dim=256,
                depth=4,
                num_heads=8,
                dropout=0.1
            )
            # Load state dict
            model.load_state_dict(saved_object['model_state_dict'])
            print("Loaded model from state dictionary")
            
        elif isinstance(saved_object, dict) and any(k.endswith('.weight') for k in saved_object.keys()):
            # It's a raw state dict
            model = CUENet(
                num_frames=NUM_FRAMES,
                input_size=INPUT_SIZE,
                embed_dim=256,
                depth=4,
                num_heads=8,
                dropout=0.1
            )
            model.load_state_dict(saved_object)
            print("Loaded model from raw state dictionary")
            
        else:
            # It's likely a full model
            model = saved_object
            print("Loaded complete model object")
        
        # Move model to the appropriate device and set to eval mode
        model = model.to(device)
        model.eval()
        
        print("Model loaded successfully and ready for inference.")
        
        # Run a small test to make sure everything's working
        print("Running model warm-up prediction...")
        dummy_input = torch.zeros((1, 3, NUM_FRAMES, INPUT_SIZE, INPUT_SIZE), dtype=torch.float32, device=device)
        with torch.no_grad():
            _ = model(dummy_input)
        
        return model, use_gpu
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        raise

def extract_frames(video_path, num_frames=NUM_FRAMES, frame_skip=2):
    """Extract frames from a video file"""
    print(f"Extracting frames from: {video_path}")
    
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
    
    # For very short videos, we might need to duplicate frames
    if total_frames < num_frames * frame_skip:
        # Sample all frames and duplicate as needed
        frame_indices = list(range(0, total_frames))
        # Repeat last frame if needed
        if total_frames > 0:
            frame_indices.extend([total_frames-1] * (num_frames - len(frame_indices)))
    else:
        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames-1, num_frames * frame_skip, dtype=int)
        # Take only every frame_skip frame
        frame_indices = frame_indices[::frame_skip][:num_frames]
    
    # Extract frames
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
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
    
    # Convert to numpy array
    frames = np.array(frames)
    print(f"Extracted {len(frames)} frames of shape {frames.shape[1:]}")
    
    return frames

def preprocess_frames(frames):
    """
    Preprocess frames for model input.
    
    For PyTorch models, we need to:
    1. Normalize with ImageNet mean/std
    2. Convert from numpy to tensor
    3. Change dimension order
    """
    # ImageNet normalization values
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    
    # Normalize pixel values
    normalized_frames = frames.astype(np.float32) / 255.0
    normalized_frames = (normalized_frames - mean) / std
    
    # Convert to tensor and reorder dimensions
    # From [T, H, W, C] to [C, T, H, W] which is what PyTorch expects
    tensor_frames = torch.from_numpy(normalized_frames).float()  # Explicitly cast to float32
    tensor_frames = tensor_frames.permute(3, 0, 1, 2)
    
    return tensor_frames

def predict_violence(model, frames, threshold=0.5, debug=False, device=None):
    """
    Make a violence prediction on the given frames.
    
    Args:
        model: PyTorch model
        frames: Tensor of frames [C, T, H, W]
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
    if len(frames.shape) == 4:  # [C, T, H, W]
        frames = frames.unsqueeze(0)  # [1, C, T, H, W]
    
    # Move to device and ensure float32 precision
    frames = frames.to(device).float()
    
    if debug:
        print(f"Input tensor type: {frames.dtype}")
        print(f"Model weight type: {next(model.parameters()).dtype}")
    
    # Run prediction with timing
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model(frames)
    
    end_time = time.time()
    inference_time = end_time - start_time
    
    # Convert output to numpy for processing
    outputs_np = outputs.cpu().numpy()
    
    # Print raw predictions if in debug mode
    if debug:
        print(f"Raw predictions: {outputs_np}")
        print(f"Output shape: {outputs_np.shape}")
        print(f"Inference time: {inference_time:.2f} seconds")
    
    # Process predictions (assuming binary classification: [non-fight, fight])
    if outputs_np.shape[1] >= 2:
        non_fight_prob = float(outputs_np[0][0])
        fight_prob = float(outputs_np[0][1])
        
        if debug:
            print(f"Non-fight logit: {non_fight_prob:.4f}")
            print(f"Fight logit: {fight_prob:.4f}")
        
        # Apply softmax to get proper probabilities
        # Need to apply when sum is not ~1.0 or any value is negative
        if abs(non_fight_prob + fight_prob - 1.0) > 0.01 or non_fight_prob < 0 or fight_prob < 0:
            import scipy.special
            probs = scipy.special.softmax(outputs_np[0])
            non_fight_prob, fight_prob = float(probs[0]), float(probs[1])
            
            if debug:
                print(f"After softmax - Non-fight: {non_fight_prob:.4f}, Fight: {fight_prob:.4f}")
        
        # Check if fight probability exceeds threshold
        is_fight = fight_prob > threshold
        
        return is_fight, fight_prob, inference_time
    else:
        # Single output model (assumes output is fight probability)
        fight_prob = float(outputs_np[0][0])
        
        # For single output where 0 = non-fight, 1 = fight, we might need to adjust
        # If the model outputs values <0.5 for fights, we may need to invert
        if debug:
            print(f"Single output value: {fight_prob:.4f}")
        
        # Using the threshold directly, adjust if needed
        is_fight = fight_prob > threshold
        
        return is_fight, fight_prob, inference_time

# Function to test the model on a single video
def test_detection(model_path, video_path, threshold=0.5):
    """Test the violence detection on a single video"""
    # Load model
    model, _ = load_violence_detection_model(model_path)
    
    # Extract and preprocess frames
    frames = extract_frames(video_path)
    processed_frames = preprocess_frames(frames)
    
    # Make prediction
    is_fight, confidence, inference_time = predict_violence(model, processed_frames, threshold, True)
    
    # Print results
    result = "VIOLENCE DETECTED" if is_fight else "NO VIOLENCE DETECTED"
    print(f"Result: {result} (Confidence: {confidence:.4f}, Inference time: {inference_time:.2f}s)")
    
    return is_fight, confidence, inference_time

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test violence detection on a video')
    parser.add_argument('--model', type=str, default='model_final.pth', help='Path to model file')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--threshold', type=float, default=0.6, help='Detection threshold')
    
    args = parser.parse_args()
    
    test_detection(args.model, args.video, args.threshold)