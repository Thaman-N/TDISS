"""
Configuration and utility functions for ablation study
"""

import json
from pathlib import Path
from typing import Dict, List

# Ablation configurations for your specific study
ABLATION_CONFIGS = {
    # -------------------------------------------------------------------------
    # 1. BASELINE CONFIGURATIONS
    # -------------------------------------------------------------------------
    'baseline_x3d': {
        'use_motion_enhancement': False,
        'use_temporal_kernel_optimization': False,
        'use_yolo_cropping': False,
        'use_randaugment': False,
        # New Flags (Disabled for Baseline)
        'use_content_aware_sampling': False,
        'use_tsa_block': False,
        'use_dense_eval': False,
        'spatial_size': 224,
        'description': 'Baseline X3D (Uniform Sampling, No Augmentation)',
        'expected_accuracy': 85.0
    },

    # -------------------------------------------------------------------------
    # 2. COMPONENT ABLATIONS (Testing individual features)
    # -------------------------------------------------------------------------
    'x3d_motion': {
        'use_motion_enhancement': True,
        'use_temporal_kernel_optimization': False,
        'use_yolo_cropping': False,
        'use_randaugment': False,
        # New Flags (Disabled)
        'use_content_aware_sampling': False,
        'use_tsa_block': False,
        'use_dense_eval': False,
        'spatial_size': 224,
        'description': 'X3D + Motion Enhancement Module Only',
        'expected_accuracy': 87.5
    },

    'x3d_kernel_opt': {
        'use_motion_enhancement': False,
        'use_temporal_kernel_optimization': True,
        'use_yolo_cropping': False,
        'use_randaugment': False,
        # New Flags (Disabled)
        'use_content_aware_sampling': False,
        'use_tsa_block': False,
        'use_dense_eval': False,
        'spatial_size': 224,
        'description': 'X3D + Temporal Kernel Optimization Only',
        'expected_accuracy': 86.5
    },

    'x3d_yolo': {
        'use_motion_enhancement': False,
        'use_temporal_kernel_optimization': False,
        'use_yolo_cropping': True,
        'use_randaugment': False,
        # New Flags (Disabled)
        'use_content_aware_sampling': False,
        'use_tsa_block': False,
        'use_dense_eval': False,
        'spatial_size': 336,
        'description': 'X3D + YOLO Spatial Cropping Only',
        'expected_accuracy': 88.0
    },
    
    'x3d_randaug': {
        'use_motion_enhancement': False,
        'use_temporal_kernel_optimization': False,
        'use_yolo_cropping': False,
        'use_randaugment': True,
        # New Flags (Disabled)
        'use_content_aware_sampling': False,
        'use_tsa_block': False,
        'use_dense_eval': False,
        'spatial_size': 224,
        'description': 'X3D + RandAugment Only',
        'expected_accuracy': 87.0
    },

    # -------------------------------------------------------------------------
    # 3. REQUESTED SPECIFIC CONFIGS
    # -------------------------------------------------------------------------
    'yolo_randaug_only': {
        'use_motion_enhancement': False,     # DISABLED as requested
        'use_temporal_kernel_optimization': False, # DISABLED as requested
        'use_yolo_cropping': True,           # ENABLED
        'use_randaugment': True,             # ENABLED
        # New Flags (Disabled)
        'use_content_aware_sampling': False,
        'use_tsa_block': False,
        'use_dense_eval': False,
        'spatial_size': 336,
        'description': 'YOLO + RandAugment (No Kernel Opt, No Motion Module)',
        'expected_accuracy': 91.0
    },

    # -------------------------------------------------------------------------
    # 4. PROPOSED METHOD (Full System with Novelty)
    # -------------------------------------------------------------------------
    'proposed_method': {
        # Enable Best of Old Features
        'use_motion_enhancement': True,
        'use_temporal_kernel_optimization': True,
        'use_yolo_cropping': True,
        'use_randaugment': True,
        
        # ENABLE NEW FEATURES (The "Fixes")
        'use_content_aware_sampling': True,  # Dedup + Jitter
        'use_tsa_block': True,               # FPS-Aware TSA Block
        'use_dense_eval': True,              # 3-Crop Validation
        
        'spatial_size': 336,
        'description': 'Full System: Content-Aware Sampling + TSA Block + Dense Eval',
        'expected_accuracy': 96.5
    }
}

# Additional test configurations (optional)
ADDITIONAL_CONFIGS = {
    # Test adaptive sampling for comparison
    'adaptive_sampling': {
        'sampling_method': 'adaptive',
        'use_motion_enhancement': False,
        'use_temporal_kernel_optimization': False,
        'use_yolo_cropping': False,
        'use_randaugment': False,
        'spatial_size': 224,
        'description': 'X3D with adaptive frame sampling (your old method)',
        'expected_accuracy': 86.0,
        'priority': 7
    },
    
    # Test just YOLO cropping without RandAugment
    'yolo_only': {
        'sampling_method': 'intelligent',
        'use_motion_enhancement': True,
        'use_temporal_kernel_optimization': True,
        'use_yolo_cropping': True,
        'use_randaugment': False,
        'spatial_size': 336,
        'description': 'YOLO cropping only (no RandAugment)',
        'expected_accuracy': 91.0,
        'priority': 8
    },
    
    # Test just RandAugment without YOLO
    'randaugment_only': {
        'sampling_method': 'intelligent',
        'use_motion_enhancement': True,
        'use_temporal_kernel_optimization': True,
        'use_yolo_cropping': False,
        'use_randaugment': True,
        'spatial_size': 224,
        'description': 'RandAugment only (no YOLO cropping)',
        'expected_accuracy': 91.5,
        'priority': 9
    }
}

def get_core_ablations() -> Dict[str, Dict]:
    """Returns the dictionary of ablation configurations."""
    return ABLATION_CONFIGS

def get_all_ablations() -> Dict[str, Dict]:
    """Get all ablation configs including additional ones"""
    all_configs = {}
    all_configs.update(ABLATION_CONFIGS)
    all_configs.update(ADDITIONAL_CONFIGS)
    return all_configs

def get_quick_ablations() -> Dict[str, Dict]:
    """Get just the most important ablations for quick testing"""
    quick_configs = {
        'baseline_x3d': ABLATION_CONFIGS['baseline_x3d'],
        'intelligent_sampling': ABLATION_CONFIGS['intelligent_sampling'], 
        'motion_enhancement': ABLATION_CONFIGS['motion_enhancement'],
        'full_system': ABLATION_CONFIGS['full_system']
    }
    return quick_configs

def print_ablation_plan():
    """Print the ablation study plan"""
    configs = get_core_ablations()
    
    print("="*80)
    print("X3D VIOLENCE DETECTION - ABLATION STUDY PLAN")
    print("="*80)
    print("Testing pathway from baseline to your SOTA performance:\n")
    
    baseline_acc = 85.0
    print(f"{'Step':<4} {'Config':<20} {'Description':<45} {'Expected':<10} {'Gain':<8}")
    print("-" * 90)
    
    for i, (config_name, config) in enumerate(configs.items(), 1):
        expected = config['expected_accuracy']
        gain = expected - baseline_acc
        print(f"{i:<4} {config_name:<20} {config['description'][:43]:<45} {expected:<10.1f}% {gain:+5.1f}%")
    
    print("\n" + "="*80)
    print(f"Total improvement: {ABLATION_CONFIGS['full_system']['expected_accuracy'] - baseline_acc:.2f}%")
    print(f"Expected time: ~6 hours (6 experiments × 1 hour each)")
    print("="*80)

def save_ablation_config(config_name: str, config: Dict, output_dir: Path):
    """Save individual ablation configuration"""
    config_dir = output_dir / config_name
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = config_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

def create_ablation_commands(dataset_path: str, output_dir: str = "ablation_results") -> List[str]:
    """Create shell commands to run each ablation individually"""
    configs = get_core_ablations()
    commands = []
    
    base_cmd = f"python ablation_study.py --dataset_path {dataset_path} --output_dir {output_dir}"
    
    for config_name in configs.keys():
        cmd = f"{base_cmd} --only_configs {config_name}"
        commands.append(cmd)
    
    return commands

def create_batch_script(dataset_path: str, output_dir: str = "ablation_results") -> str:
    """Create a batch script to run all ablations sequentially"""
    commands = create_ablation_commands(dataset_path, output_dir)
    
    script = "#!/bin/bash\n\n"
    script += "# X3D Violence Detection - Ablation Study Batch Script\n"
    script += f"# Generated for dataset: {dataset_path}\n\n"
    
    script += "echo \"Starting X3D Ablation Study...\"\n"
    script += "echo \"Expected total time: ~6 hours\"\n\n"
    
    for i, cmd in enumerate(commands, 1):
        config_name = cmd.split('--only_configs ')[-1]
        script += f"echo \"Running experiment {i}/6: {config_name}\"\n"
        script += f"{cmd}\n"
        script += f"echo \"Completed experiment {i}/6\"\n\n"
    
    script += "echo \"All ablation experiments completed!\"\n"
    script += f"echo \"Results saved to: {output_dir}\"\n"
    
    return script

# Expected results based on your table and improvements
EXPECTED_RESULTS = {
    'baseline_x3d': {
        'accuracy': 85.0,
        'parameters': 4_000_000,
        'description': 'Standard X3D baseline'
    },
    'intelligent_sampling': {
        'accuracy': 88.0,
        'parameters': 4_000_000,
        'description': 'Your intelligent sampling breakthrough (+3%)'
    },
    'temporal_kernels': {
        'accuracy': 89.0,
        'parameters': 4_000_000,
        'description': 'Add temporal kernel optimization (+1%)'
    },
    'motion_enhancement': {
        'accuracy': 90.5,
        'parameters': 4_200_000,
        'description': 'Add Motion Enhancement Module (+1.5%)'
    },
    'yolo_randaugment': {
        'accuracy': 93.5,
        'parameters': 4_200_000,
        'description': 'Add YOLO + RandAugment (+3%)'
    },
    'full_system': {
        'accuracy': 94.25,
        'parameters': 4_200_000,
        'description': 'Full optimized system (+0.75%)'
    }
}

def analyze_component_contributions():
    """Analyze individual component contributions"""
    results = EXPECTED_RESULTS
    baseline = results['baseline_x3d']['accuracy']
    
    print("\nCOMPONENT CONTRIBUTION ANALYSIS:")
    print("-" * 50)
    
    components = [
        ('intelligent_sampling', 'Intelligent Frame Sampling'),
        ('temporal_kernels', 'Temporal Kernel Optimization'), 
        ('motion_enhancement', 'Motion Enhancement Module'),
        ('yolo_randaugment', 'YOLO Cropping + RandAugment'),
        ('full_system', 'Final Optimizations')
    ]
    
    prev_acc = baseline
    total_gain = 0
    
    for config_name, description in components:
        current_acc = results[config_name]['accuracy']
        component_gain = current_acc - prev_acc
        total_gain += component_gain
        
        print(f"{description:<35}: +{component_gain:4.1f}% (total: {current_acc:5.1f}%)")
        prev_acc = current_acc
    
    print("-" * 50)
    print(f"{'Total improvement':<35}: +{total_gain:4.1f}% ({baseline:.1f}% → {results['full_system']['accuracy']:.2f}%)")
    print(f"{'vs CUE-Net (354M params)':<35}: Similar accuracy, 88x fewer params!")

if __name__ == "__main__":
    print_ablation_plan()
    analyze_component_contributions()
    
    # Example usage
    print(f"\nTo run the ablation study:")
    print(f"python ablation_study.py --dataset_path /path/to/RWF-2000 --output_dir ablation_results")
    
    print(f"\nTo run just key experiments:")
    quick_configs = list(get_quick_ablations().keys())
    print(f"python ablation_study.py --dataset_path /path/to/RWF-2000 --only_configs {' '.join(quick_configs)}")
