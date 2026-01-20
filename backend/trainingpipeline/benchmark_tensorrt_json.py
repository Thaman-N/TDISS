import os
import time
import json
import torch
import gc
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import argparse
import psutil
import platform
from datetime import datetime
from pathlib import Path
from x3d_model import CleanX3DViolenceDetector

class MasterBenchmarker:
    def __init__(self, model_path, engine_path, input_size=336, num_frames=16, device='cuda'):
        self.model_path = model_path
        self.engine_path = engine_path
        self.input_size = input_size
        self.num_frames = num_frames
        self.device = device

    def get_torch_vram(self):
        # Returns current VRAM usage in MB
        return torch.cuda.memory_reserved(self.device) / 1024**2

    def get_trt_vram(self):
        # Returns total GPU memory currently free vs used via PyCUDA
        free, total = cuda.mem_get_info()
        return (total - free) / 1024**2

    def get_system_info(self):
        return {
            "timestamp": datetime.now().isoformat(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        }

    def benchmark_pytorch_suite(self, batch_sizes):
        print(f"\n--- [1/2] Starting PyTorch Baseline ---")
        model = CleanX3DViolenceDetector(x3d_model_name='x3d_m', num_classes=2, use_motion_enhancement=True, device=self.device)
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        if any(k.startswith('model.') for k in state_dict.keys()):
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        model.eval().to(self.device)
        
        results = {}
        with torch.no_grad():
            for bs in batch_sizes:
                try:
                    # Offset by 1e-5 to avoid subnormal penalty
                    rgb = (torch.randn(bs, 3, self.num_frames, self.input_size, self.input_size, device=self.device) + 1e-5)
                    flow = (torch.randn(bs, 3, self.num_frames, self.input_size, self.input_size, device=self.device) + 1e-5)
                    
                    for _ in range(5): model({'rgb': rgb, 'flow': flow})
                    torch.cuda.synchronize()

                    vram_usage = self.get_torch_vram() # Measure after warmup
                    latencies = []
                    for _ in range(30):
                        start = time.perf_counter()
                        model({'rgb': rgb, 'flow': flow})
                        torch.cuda.synchronize()
                        latencies.append((time.perf_counter() - start) * 1000)
                    
                    avg_lat = np.mean(latencies)
                    results[bs] = {"latency_ms": avg_lat, "fps": (1000/avg_lat)*bs, "vram_mb": vram_usage}
                    print(f"  Batch {bs} | FPS: {(1000/avg_lat)*bs:.2f} | VRAM: {vram_usage:.1f}MB")
                    del rgb, flow
                    torch.cuda.empty_cache()
                except RuntimeError:
                    results[bs] = {"latency_ms": 0, "fps": 0, "vram_mb": 0}

        del model, checkpoint, state_dict
        gc.collect()
        torch.cuda.empty_cache()
        return results

    def _allocate_trt_buffers(self, engine, context, batch_size):
        inputs, outputs, allocations = [], [], []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            dtype = engine.get_tensor_dtype(name)
            mode = engine.get_tensor_mode(name)
            shape = list(engine.get_tensor_shape(name))
            if shape[0] == -1: shape[0] = batch_size
            if mode == trt.TensorIOMode.INPUT:
                context.set_input_shape(name, tuple(shape))
            
            size = trt.volume(shape) * dtype.itemsize
            allocation = cuda.mem_alloc(size)
            context.set_tensor_address(name, int(allocation))
            
            binding = {'name': name, 'allocation': allocation}
            if mode == trt.TensorIOMode.INPUT: inputs.append(binding)
            else: outputs.append(binding)
            allocations.append(allocation)
        return inputs, outputs, allocations

    def benchmark_tensorrt_suite(self, batch_sizes):
        print(f"\n--- [2/2] Starting TensorRT Optimized ---")
        # Clear cache before TRT
        torch.cuda.empty_cache()
        gc.collect()
        
        base_vram = self.get_trt_vram() # Background VRAM usage
        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        stream = cuda.Stream()
        
        results = {}
        for bs in batch_sizes:
            try:
                ins, outs, allocs = self._allocate_trt_buffers(engine, context, bs)
                # Offset by 1e-5 for subnormals
                rgb = (np.random.randn(bs, 3, self.num_frames, self.input_size, self.input_size).astype(np.float32) + 1e-5)
                flow = (np.random.randn(bs, 3, self.num_frames, self.input_size, self.input_size).astype(np.float32) + 1e-5)
                
                cuda.memcpy_htod_async(ins[0]['allocation'], np.ascontiguousarray(rgb), stream)
                if len(ins) > 1:
                    cuda.memcpy_htod_async(ins[1]['allocation'], np.ascontiguousarray(flow), stream)
                
                for _ in range(10): context.execute_async_v3(stream.handle)
                stream.synchronize()

                vram_usage = self.get_trt_vram() - base_vram # Incremental VRAM for this BS
                latencies = []
                for _ in range(50):
                    start = time.perf_counter()
                    context.execute_async_v3(stream.handle)
                    stream.synchronize()
                    latencies.append((time.perf_counter() - start) * 1000)
                
                avg_lat = np.mean(latencies)
                results[bs] = {"latency_ms": avg_lat, "fps": (1000/avg_lat)*bs, "vram_mb": vram_usage}
                print(f"  Batch {bs} | FPS: {(1000/avg_lat)*bs:.2f} | VRAM: {vram_usage:.1f}MB")

                for a in allocs: a.free()
                del rgb, flow, ins, outs, allocs
            except Exception as e:
                results[bs] = {"latency_ms": 0, "fps": 0, "vram_mb": 0}

        del context, engine, stream
        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--engine_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--batch_sizes", nargs='+', type=int, default=[1, 8, 16, 32])
    args = parser.parse_args()

    bench = MasterBenchmarker(args.model_path, args.engine_path)
    sys_info = bench.get_system_info()
    torch_res = bench.benchmark_pytorch_suite(args.batch_sizes)
    trt_res = bench.benchmark_tensorrt_suite(args.batch_sizes)

    report = {"system_info": sys_info, "results": []}

    print("\n" + "="*85)
    print(f"{'BS':<5} | {'PyTorch FPS':<12} | {'VRAM':<10} | {'TRT FPS':<12} | {'VRAM':<10} | {'Speedup':<8}")
    print("-" * 85)
    
    for bs in args.batch_sizes:
        p = torch_res[bs]
        t = trt_res[bs]
        speedup = t['fps'] / p['fps'] if p['fps'] > 0 else 0
        
        print(f"{bs:<5} | {p['fps']:<12.2f} | {p['vram_mb']:<10.1f} | {t['fps']:<12.2f} | {t['vram_mb']:<10.1f} | {speedup:<8.2f}x")
        
        report["results"].append({"batch_size": bs, "pytorch": p, "tensorrt": t, "speedup": speedup})

    filename = f"benchmark_full_{sys_info['gpu_name'].replace(' ', '_')}_{int(time.time())}.json"
    with open(Path(args.output_dir) / filename, 'w') as f:
        json.dump(report, f, indent=4)
    print("="*85 + f"\n✅ Saved to: {filename}")

if __name__ == "__main__":
    main()