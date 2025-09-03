import os
import json
import torch
from pathlib import Path
from torch_detection import load_violence_detection_model, extract_frames, preprocess_frames, predict_violence

def evaluate_rlvs_dataset(dataset_path):
    """
    Evaluate the trained X3D violence detection model on the RLVS dataset.
    Uses hardcoded paths for the model and dataset.
    """
    # Hardcoded paths
    model_path = r"rlvs9875.pth"
    threshold = 0.5
    output_json = "rlvs_evaluation_results.json"
    
    # Load the trained model
    print("Loading trained model...")
    model, use_gpu = load_violence_detection_model(model_path)
    device = next(model.parameters()).device
    
    # Define dataset paths
    violence_dir = Path(dataset_path) / "Fight"
    non_violence_dir = Path(dataset_path) / "NonFight"
    
    # Check if directories exist
    if not violence_dir.exists():
        raise ValueError(f"Violence directory not found: {violence_dir}")
    if not non_violence_dir.exists():
        raise ValueError(f"NonViolence directory not found: {non_violence_dir}")
    
    # Get all video files (common video extensions)
    violence_videos = list(violence_dir.glob("*.mp4")) + list(violence_dir.glob("*.avi"))
    non_violence_videos = list(non_violence_dir.glob("*.mp4")) + list(non_violence_dir.glob("*.avi"))
    
    print(f"Found {len(violence_videos)} violence videos and {len(non_violence_videos)} non-violence videos")
    
    # Initialize results storage
    results = {
        "violence": {"correct": 0, "incorrect": 0, "details": []},
        "non_violence": {"correct": 0, "incorrect": 0, "details": []},
        "failed_predictions": [],
        "summary": {}
    }
    
    # Process violence videos (should be classified as violence)
    print("\nProcessing violence videos...")
    for i, video_path in enumerate(violence_videos):
        print(f"  [{i+1}/{len(violence_videos)}] Processing: {video_path.name}")
        
        try:
            # Extract and preprocess frames
            frames = extract_frames(str(video_path))
            
            # Determine if model uses motion enhancement
            use_motion = hasattr(model, 'use_motion_enhancement') and model.use_motion_enhancement
            
            # Preprocess with or without optical flow
            processed_data = preprocess_frames(frames, compute_flow=use_motion)
            
            # Make prediction
            is_fight, confidence, inference_time = predict_violence(
                model, processed_data, threshold, debug=False, device=device
            )
            
            # Check if prediction is correct
            is_correct = is_fight  # Should be True for violence videos
            
            # Store result
            result_detail = {
                "video_path": str(video_path),
                "true_label": "violence",
                "predicted_label": "violence" if is_fight else "non_violence",
                "confidence": float(confidence),
                "inference_time": float(inference_time),
                "correct": bool(is_correct)
            }
            
            if is_correct:
                results["violence"]["correct"] += 1
            else:
                results["violence"]["incorrect"] += 1
                results["failed_predictions"].append(result_detail)
                
            results["violence"]["details"].append(result_detail)
            
        except Exception as e:
            error_detail = {
                "video_path": str(video_path),
                "true_label": "violence",
                "error": str(e),
                "correct": False
            }
            results["violence"]["incorrect"] += 1
            results["failed_predictions"].append(error_detail)
            results["violence"]["details"].append(error_detail)
            print(f"    Error processing video: {e}")
    
    # Process non-violence videos (should be classified as non-violence)
    print("\nProcessing non-violence videos...")
    for i, video_path in enumerate(non_violence_videos):
        print(f"  [{i+1}/{len(non_violence_videos)}] Processing: {video_path.name}")
        
        try:
            # Extract and preprocess frames
            frames = extract_frames(str(video_path))
            
            # Determine if model uses motion enhancement
            use_motion = hasattr(model, 'use_motion_enhancement') and model.use_motion_enhancement
            
            # Preprocess with or without optical flow
            processed_data = preprocess_frames(frames, compute_flow=use_motion)
            
            # Make prediction
            is_fight, confidence, inference_time = predict_violence(
                model, processed_data, threshold, debug=False, device=device
            )
            
            # Check if prediction is correct
            is_correct = not is_fight  # Should be False for non-violence videos
            
            # Store result
            result_detail = {
                "video_path": str(video_path),
                "true_label": "non_violence",
                "predicted_label": "violence" if is_fight else "non_violence",
                "confidence": float(confidence),
                "inference_time": float(inference_time),
                "correct": bool(is_correct)
            }
            
            if is_correct:
                results["non_violence"]["correct"] += 1
            else:
                results["non_violence"]["incorrect"] += 1
                results["failed_predictions"].append(result_detail)
                
            results["non_violence"]["details"].append(result_detail)
            
        except Exception as e:
            error_detail = {
                "video_path": str(video_path),
                "true_label": "non_violence",
                "error": str(e),
                "correct": False
            }
            results["non_violence"]["incorrect"] += 1
            results["failed_predictions"].append(error_detail)
            results["non_violence"]["details"].append(error_detail)
            print(f"    Error processing video: {e}")
    
    # Calculate summary statistics
    total_videos = len(violence_videos) + len(non_violence_videos)
    correct_predictions = results["violence"]["correct"] + results["non_violence"]["correct"]
    incorrect_predictions = results["violence"]["incorrect"] + results["non_violence"]["incorrect"]
    
    results["summary"] = {
        "total_videos": total_videos,
        "correct_predictions": correct_predictions,
        "incorrect_predictions": incorrect_predictions,
        "accuracy": correct_predictions / total_videos if total_videos > 0 else 0,
        "violence_accuracy": results["violence"]["correct"] / len(violence_videos) if len(violence_videos) > 0 else 0,
        "non_violence_accuracy": results["non_violence"]["correct"] / len(non_violence_videos) if len(non_violence_videos) > 0 else 0,
        "false_positives": results["non_violence"]["incorrect"],  # Non-violence classified as violence
        "false_negatives": results["violence"]["incorrect"],     # Violence classified as non-violence
        "threshold_used": threshold,
        "model_path": model_path,
        "dataset_path": dataset_path
    }
    
    # Save results to JSON file
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Total videos processed: {total_videos}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Incorrect predictions: {incorrect_predictions}")
    print(f"Overall accuracy: {results['summary']['accuracy']:.4f}")
    print(f"Violence accuracy: {results['summary']['violence_accuracy']:.4f}")
    print(f"Non-violence accuracy: {results['summary']['non_violence_accuracy']:.4f}")
    print(f"False positives: {results['summary']['false_positives']}")
    print(f"False negatives: {results['summary']['false_negatives']}")
    print(f"Failed predictions saved to: {output_json}")
    
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python evaluaterlvs.py <dataset_path>")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    results = evaluate_rlvs_dataset(dataset_path)