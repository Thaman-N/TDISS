import os
import torch
import torch.nn as nn
import onnx
import tensorrt as trt
import argparse
from pathlib import Path

# Correct Import
from x3d_model import CleanX3DViolenceDetector

# Constants
NUM_FRAMES = 16
INPUT_SIZE = 336
NUM_CLASSES = 2

class WrapperModule(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, rgb, flow=None):
        data = {'rgb': rgb}
        if flow is not None:
            data['flow'] = flow
        return self.model(data)

def export_to_onnx(model, onnx_path, max_batch, use_motion=True, device='cuda'):
    print(f"🚀 Starting ONNX export to {onnx_path}...")
    
    # 1. Create Dummy Inputs
    dummy_rgb = torch.randn(1, 3, NUM_FRAMES, INPUT_SIZE, INPUT_SIZE, device=device)
    
    input_names = ['rgb']
    input_tensors = (dummy_rgb,)
    dynamic_axes = {'rgb': {0: 'batch_size'}, 'output': {0: 'batch_size'}}

    if use_motion:
        dummy_flow = torch.randn(1, 3, NUM_FRAMES, INPUT_SIZE, INPUT_SIZE, device=device)
        input_names.append('flow')
        input_tensors = (dummy_rgb, dummy_flow)
        dynamic_axes['flow'] = {0: 'batch_size'}

    # 2. Force Lazy Init
    print("⚡ Running dummy forward pass...")
    with torch.no_grad():
        if use_motion:
            model(dummy_rgb, dummy_flow)
        else:
            model(dummy_rgb)

    # 3. Export with EMBEDDED weights
    print(f"📦 Exporting to ONNX (Opset 19 - Embedded Weights)...")
    try:
        torch.onnx.export(
            model,
            input_tensors,
            onnx_path,
            export_params=True,
            opset_version=19,      
            do_constant_folding=True,
            input_names=input_names,
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
    except Exception as e:
        print(f"⚠️ Export failed with Opset 19, retrying with default...")
        torch.onnx.export(
            model,
            input_tensors,
            onnx_path,
            export_params=True,
            do_constant_folding=True,
            input_names=input_names,
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )
    
    # Verification
    if os.path.exists(onnx_path):
        model_proto = onnx.load(onnx_path)
        onnx.checker.check_model(model_proto)
        onnx.save_model(model_proto, onnx_path, save_as_external_data=False)
        print("✅ ONNX export successful (Single File Verified)!")
    else:
        print("❌ ONNX file was not created!")

def build_tensorrt_engine(onnx_path, engine_path, max_batch_size, fp16=True):
    print(f"🚀 Building TensorRT Engine from {onnx_path}...")
    print(f"⚙️  Max Batch Size set to: {max_batch_size}")
    
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()

    if fp16:
        if builder.platform_has_fast_fp16:
            print("⚡ Enabling FP16 precision")
            config.set_flag(trt.BuilderFlag.FP16)

    # 8GB Workspace
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 * (1 << 30))

    # Parse ONNX
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("❌ Failed to parse ONNX file:")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # Optimization Profile
    profile = builder.create_optimization_profile()
    
    # RGB Input Profile
    profile.set_shape(
        'rgb', 
        (1, 3, NUM_FRAMES, INPUT_SIZE, INPUT_SIZE),        # Min
        (1, 3, NUM_FRAMES, INPUT_SIZE, INPUT_SIZE),        # Opt (Optimize for Batch 1 latency)
        (max_batch_size, 3, NUM_FRAMES, INPUT_SIZE, INPUT_SIZE) # Max (Dynamic)
    )
    
    # Flow Input Profile
    if network.num_inputs > 1:
        profile.set_shape(
            'flow',
            (1, 3, NUM_FRAMES, INPUT_SIZE, INPUT_SIZE),
            (1, 3, NUM_FRAMES, INPUT_SIZE, INPUT_SIZE),
            (max_batch_size, 3, NUM_FRAMES, INPUT_SIZE, INPUT_SIZE)
        )

    config.add_optimization_profile(profile)

    print("🛠️  Building TensorRT engine...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("❌ Engine build failed!")
        return None

    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"🎉 Engine saved to {engine_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="exported_models")
    # NEW FLAG
    parser.add_argument("--max_batch_size", type=int, default=8, help="Max batch size for TensorRT engine")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    onnx_path = output_dir / "x3d_violence.onnx"
    trt_path = output_dir / f"x3d_violence_b{args.max_batch_size}.trt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("📥 Loading PyTorch model...")
    base_model = CleanX3DViolenceDetector(x3d_model_name='x3d_m', num_classes=NUM_CLASSES, use_motion_enhancement=True, device=device)
    
    try:
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.model_path, map_location=device)
        
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    base_model.load_state_dict(state_dict, strict=False)
    base_model.eval().to(device)

    wrapped_model = WrapperModule(base_model).eval().to(device)

    export_to_onnx(wrapped_model, str(onnx_path), args.max_batch_size, use_motion=True, device=device)
    
    if device == "cuda":
        build_tensorrt_engine(str(onnx_path), str(trt_path), args.max_batch_size, fp16=True)

if __name__ == "__main__":
    main()