import torch
import sys
import os
sys.path.append(os.getcwd())

from x3d_model import create_model
from x3d_dataset import CUENetStyleDataset
from torch.utils.data import DataLoader
from pathlib import Path
import json

MODEL_PATH = r"D:\Thaman\College\Capstone\capstoneproj\TDISS\backend\tp2\cuenet_methodology_checkpoints\optimized_best_model.pth"
VAL_PATH = r"D:\Thaman\archive\RWF-2000"

def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = create_model(
        model_name="x3d_m",
        num_classes=2,
        use_motion_enhancement=True,
        device=device
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def analyze_model(model_path, output_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = load_model(model_path, device)
    
    val_dataset = CUENetStyleDataset(
        dataset_path=VAL_PATH,
        split="val",
        clip_len=16,
        spatial_size=336,  # Use CUE-Net resolution
        compute_optical_flow=True,
        # No augmentation for validation
        use_cuenet_cropping=False,
        use_randaugment=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    misclassified = []
    total_correct = 0
    total_samples = 0
    
    print(f"Analyzing {len(val_dataset)} validation videos...")
    
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(val_loader):
            data = {k: v.to(device, non_blocking=True) for k, v in data.items()}
            label = label.to(device, non_blocking=True)
            
            output = model(data)
            prediction = output.argmax(dim=1)
            
            is_correct = (prediction == label).item()
            total_correct += is_correct
            total_samples += 1
            
            if not is_correct:
                video_path = val_dataset.video_paths[batch_idx]
                video_name = Path(video_path).name
                
                actual_class = "Fight" if label.item() == 1 else "NonFight"
                predicted_class = "Fight" if prediction.item() == 1 else "NonFight"
                confidence = torch.softmax(output, dim=1).max().item()
                
                fight_folder = "Fight" in str(video_path)
                
                misclassified.append({
                    'video_name': video_name,
                    'video_path': str(video_path),
                    'actual_class': actual_class,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'in_fight_folder': fight_folder
                })
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Processed {batch_idx + 1}/{len(val_dataset)} videos")
    
    accuracy = (total_correct / total_samples) * 100
    
    result = {
        'model_path': model_path,
        'total_videos': total_samples,
        'accuracy': accuracy,
        'correct_predictions': total_correct,
        'misclassified_count': len(misclassified),
        'misclassified_videos': misclassified
    }
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nModel: {Path(model_path).name}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Misclassified: {len(misclassified)}/{total_samples}")
    print(f"Results saved to: {output_file}")
    
    false_positives = [v for v in misclassified if v['actual_class'] == 'NonFight' and v['predicted_class'] == 'Fight']
    false_negatives = [v for v in misclassified if v['actual_class'] == 'Fight' and v['predicted_class'] == 'NonFight']
    
    print(f"False Positives: {len(false_positives)}")
    print(f"False Negatives: {len(false_negatives)}")
    
    return result

if __name__ == "__main__":
    output_file = f"misclassified_analysis_cuenet_{Path(MODEL_PATH).stem}.json"
    analyze_model(MODEL_PATH, output_file)