import torch
import sys
import os
import json
import argparse
from datetime import datetime
from pathlib import Path
import time

sys.path.append(os.getcwd())

from x3d_model import create_model
from x3d_dataset import CUENetStyleDataset
from torch.utils.data import DataLoader

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

def run_single_evaluation(model_path, device, val_loader, val_dataset, run_id):
    model = load_model(model_path, device)
    
    misclassified = []
    total_correct = 0
    total_samples = 0
    
    print(f"Run {run_id}: Analyzing {len(val_dataset)} validation videos...")
    
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
                print(f"Run {run_id}: Processed {batch_idx + 1}/{len(val_dataset)} videos")
    
    accuracy = (total_correct / total_samples) * 100
    
    false_positives = [v for v in misclassified if v['actual_class'] == 'NonFight' and v['predicted_class'] == 'Fight']
    false_negatives = [v for v in misclassified if v['actual_class'] == 'Fight' and v['predicted_class'] == 'NonFight']
    
    return {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'total_videos': total_samples,
        'accuracy': accuracy,
        'correct_predictions': total_correct,
        'misclassified_count': len(misclassified),
        'false_positives': len(false_positives),
        'false_negatives': len(false_negatives),
        'misclassified_videos': misclassified
    }

def analyze_model_multiple_runs(model_path, dataset_path, num_runs=5, output_dir=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create output directory with timestamp
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"{timestamp}_val_output")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Create validation dataset and loader
    val_dataset = CUENetStyleDataset(
        dataset_path=dataset_path,
        split="val",
        clip_len=16,
        spatial_size=336,
        compute_optical_flow=True,
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
    
    # Run multiple evaluations
    all_results = []
    
    for run_id in range(1, num_runs + 1):
        print(f"\n{'='*50}")
        print(f"Starting evaluation run {run_id}/{num_runs}")
        print(f"{'='*50}")
        
        result = run_single_evaluation(model_path, device, val_loader, val_dataset, run_id)
        all_results.append(result)
        
        # Save individual run result (maintaining original functionality)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        individual_output_file = output_dir / f"misclassified_analysis_cuenet_{Path(model_path).stem}_run{run_id}_{timestamp}.json"
        
        individual_result = {
            'model_path': model_path,
            'dataset_path': dataset_path,
            **result  # Include all the run-specific data
        }
        
        with open(individual_output_file, 'w') as f:
            json.dump(individual_result, f, indent=2)
        
        print(f"Individual run {run_id} results saved to: {individual_output_file}")
        print(f"Run {run_id} Accuracy: {result['accuracy']:.2f}%")
        print(f"Run {run_id} Misclassified: {result['misclassified_count']}/{result['total_videos']}")
        print(f"Run {run_id} False Positives: {result['false_positives']}")
        print(f"Run {run_id} False Negatives: {result['false_negatives']}")
    
    # Save consolidated results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(model_path).stem
    consolidated_output_file = output_dir / f"consolidated_analysis_{model_name}_{timestamp}.json"
    
    consolidated_result = {
        'model_path': model_path,
        'dataset_path': dataset_path,
        'device': device,
        'total_runs': num_runs,
        'timestamp': timestamp,
        'average_accuracy': sum(r['accuracy'] for r in all_results) / num_runs,
        'runs': all_results
    }
    
    with open(consolidated_output_file, 'w') as f:
        json.dump(consolidated_result, f, indent=2)
    
    # Print summary
    print(f"\n{'='*50}")
    print("CONSOLIDATED RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_path}")
    print(f"Total runs: {num_runs}")
    print(f"Average accuracy: {consolidated_result['average_accuracy']:.2f}%")
    
    for run in all_results:
        print(f"Run {run['run_id']}: {run['accuracy']:.2f}%")
    
    print(f"\nConsolidated results saved to: {consolidated_output_file}")
    print(f"All outputs saved in: {output_dir}")
    
    return consolidated_result

def main():
    parser = argparse.ArgumentParser(description='Run multiple validation evaluations on X3D model')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--model', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--runs', type=int, default=5, help='Number of evaluation runs (default: 5)')
    parser.add_argument('--output', type=str, help='Custom output directory (optional)')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset path '{args.dataset}' does not exist!")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"Error: Model path '{args.model}' does not exist!")
        sys.exit(1)
    
    print(f"Dataset path: {args.dataset}")
    print(f"Model path: {args.model}")
    print(f"Number of runs: {args.runs}")
    
    analyze_model_multiple_runs(
        model_path=args.model,
        dataset_path=args.dataset,
        num_runs=args.runs,
        output_dir=args.output
    )

if __name__ == "__main__":
    main()