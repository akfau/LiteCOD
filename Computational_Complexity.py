import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import cv2
import time
from thop import profile, clever_format
from torchsummary import summary
from ptflops import get_model_complexity_info
import psutil
import gc

# Uncomment the appropriate import based on your model
# from lib.GLCONet_Swin import Network
from lib.GLCONet_PVT import Network
from utils.data_val import test_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def load_model_state_dict(model, checkpoint_path, device='cuda'):
    """
    Robust state_dict loading function that handles various checkpoint formats
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Check if checkpoint file exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("Loaded state_dict from checkpoint['state_dict']")
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("Loaded state_dict from checkpoint['model_state_dict']")
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
                print("Loaded state_dict from checkpoint['model']")
            else:
                # Assume the entire dict is the state_dict
                state_dict = checkpoint
                print("Using entire checkpoint as state_dict")
        else:
            # Assume it's directly the state_dict
            state_dict = checkpoint
            print("Checkpoint is directly a state_dict")
        
        # Clean state_dict keys - remove 'module.' prefix if present (from DataParallel)
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            # Remove 'module.' prefix
            clean_key = key.replace('module.', '')
            cleaned_state_dict[clean_key] = value
        
        # Get model's state dict for comparison
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(cleaned_state_dict.keys())
        
        # Check for missing and unexpected keys
        missing_keys = model_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - model_keys
        
        if missing_keys:
            print(f"WARNING: Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"WARNING: Unexpected keys in checkpoint: {unexpected_keys}")
        
        # Load the state dict
        model.load_state_dict(cleaned_state_dict, strict=False)
        print("Model state_dict loaded successfully!")
        
        # Print loading statistics
        total_model_params = len(model_keys)
        loaded_params = len(checkpoint_keys & model_keys)
        print(f"Loaded {loaded_params}/{total_model_params} parameters ({loaded_params/total_model_params*100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False

def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_model_size_mb(model):
    """Calculate model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def measure_inference_time(model, input_tensor, num_runs=100, warmup_runs=10):
    """Measure inference time with GPU and CPU timing"""
    model.eval()
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
    
    # GPU timing
    torch.cuda.synchronize()
    gpu_times = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            _ = model(input_tensor)
            end_event.record()
            
            torch.cuda.synchronize()
            gpu_times.append(start_event.elapsed_time(end_event))
    
    # CPU timing
    cpu_times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(input_tensor)
            torch.cuda.synchronize()
            end_time = time.time()
            cpu_times.append((end_time - start_time) * 1000)  # Convert to ms
    
    return {
        'gpu_mean': np.mean(gpu_times),
        'gpu_std': np.std(gpu_times),
        'cpu_mean': np.mean(cpu_times),
        'cpu_std': np.std(cpu_times),
        'fps': 1000 / np.mean(gpu_times)
    }

def measure_memory_usage(model, input_tensor):
    """Measure GPU memory usage during inference"""
    torch.cuda.empty_cache()
    
    # Measure memory before inference
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated()
    
    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
    
    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated()
    mem_peak = torch.cuda.max_memory_allocated()
    
    return {
        'memory_before_mb': mem_before / 1024**2,
        'memory_after_mb': mem_after / 1024**2,
        'memory_peak_mb': mem_peak / 1024**2,
        'memory_used_mb': (mem_after - mem_before) / 1024**2
    }

def analyze_model_complexity(model, input_size=(3, 384, 384), device='cuda'):
    """Comprehensive model complexity analysis"""
    print("="*80)
    print("MODEL COMPLEXITY ANALYSIS")
    print("="*80)
    
    # Create dummy input
    dummy_input = torch.randn(1, *input_size).to(device)
    model = model.to(device)
    model.eval()
    
    # 1. Parameter Count
    total_params, trainable_params = count_parameters(model)
    print(f"\n1. PARAMETER ANALYSIS:")
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"   Non-trainable Parameters: {total_params - trainable_params:,}")
    
    # 2. Model Size
    model_size_mb = get_model_size_mb(model)
    print(f"\n2. MODEL SIZE:")
    print(f"   Model Size: {model_size_mb:.2f} MB")
    
    # 3. FLOPs and MACs using thop
    try:
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops_formatted, params_formatted = clever_format([flops, params], "%.3f")
        print(f"\n3. COMPUTATIONAL COMPLEXITY (THOP):")
        print(f"   FLOPs: {flops_formatted}")
        print(f"   Parameters: {params_formatted}")
        print(f"   MACs: {flops/2:.2e} (FLOPs/2)")
    except Exception as e:
        print(f"\n3. THOP Analysis failed: {e}")
    
    # 4. FLOPs using ptflops
    try:
        macs, params = get_model_complexity_info(model, input_size, print_per_layer_stat=False, verbose=False)
        print(f"\n4. COMPUTATIONAL COMPLEXITY (PTFLOPS):")
        print(f"   MACs: {macs}")
        print(f"   Parameters: {params}")
    except Exception as e:
        print(f"\n4. PTFLOPS Analysis failed: {e}")
    
    # 5. Inference Time Analysis
    print(f"\n5. INFERENCE TIME ANALYSIS:")
    timing_results = measure_inference_time(model, dummy_input)
    print(f"   GPU Time: {timing_results['gpu_mean']:.2f} ± {timing_results['gpu_std']:.2f} ms")
    print(f"   CPU Time: {timing_results['cpu_mean']:.2f} ± {timing_results['cpu_std']:.2f} ms")
    print(f"   FPS: {timing_results['fps']:.2f}")
    
    # 6. Memory Usage Analysis
    print(f"\n6. MEMORY USAGE ANALYSIS:")
    memory_results = measure_memory_usage(model, dummy_input)
    print(f"   Memory Before: {memory_results['memory_before_mb']:.2f} MB")
    print(f"   Memory After: {memory_results['memory_after_mb']:.2f} MB")
    print(f"   Memory Peak: {memory_results['memory_peak_mb']:.2f} MB")
    print(f"   Memory Used: {memory_results['memory_used_mb']:.2f} MB")
    
    # 7. Model Summary (if torchsummary works)
    # try:
    #     print(f"\n7. MODEL SUMMARY:")
    #     summary(model, input_size, device=device.split(':')[0] if ':' in device else device)
    # except Exception as e:
    #     print(f"\n7. Model Summary failed: {e}")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': model_size_mb,
        'timing': timing_results,
        'memory': memory_results
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=512, help='testing size')
    parser.add_argument('--pth_path', type=str, default='./results/Net_epoch_160.pth')
    parser.add_argument('--test_dataset_path', type=str, default=r'D:\BCNet\data/Testset/')
    parser.add_argument('--analyze_only', action='store_true', help='Only analyze model complexity without testing')
    parser.add_argument('--channels', type=int, default=32, help='Number of channels for the model')
    opt = parser.parse_args()
    
    # Initialize model
    model = Network(channels=opt.channels)
    
    # Load pretrained weights if available
    if os.path.exists(opt.pth_path):
        if not load_model_state_dict(model, opt.pth_path, 'cuda'):
            print(f"Failed to load weights from: {opt.pth_path}")
            print("Continuing with random weights...")
    else:
        print(f"Warning: Weight file {opt.pth_path} not found. Using random weights.")
    
    model.cuda()
    
    # Analyze model complexity
    complexity_results = analyze_model_complexity(
        model, 
        input_size=(3, opt.testsize, opt.testsize),
        device='cuda'
    )
    
    # If analyze_only flag is set, skip testing
    if opt.analyze_only:
        print("\nAnalysis complete. Skipping dataset testing.")
        return
    
    # Original testing code
    print("\n" + "="*80)
    print("DATASET TESTING")
    print("="*80)
    
    for _data_name in ['CAMO','COD10K','CHAMELEON','NC4K']:
        data_path = opt.test_dataset_path+'/{}/'.format(_data_name)
        save_path = './output_results/{}/{}/'.format(opt.pth_path.split('/')[-2], _data_name)
        os.makedirs(save_path, exist_ok=True)
        
        image_root = '{}/Imgs/'.format(data_path)
        gt_root = '{}/GT/'.format(data_path)
        
        # Check if dataset exists
        if not os.path.exists(image_root) or not os.path.exists(gt_root):
            print(f"Dataset {_data_name} not found at {data_path}")
            continue
        
        test_loader = test_dataset(image_root, gt_root, opt.testsize)
        
        # print(f"\nProcessing dataset: {_data_name}")
        # print(f"Number of test images: {test_loader.size}")
        
        model.eval()
        total_time = 0
        
        for i in range(test_loader.size):
            image, gt, name, img_for_post = test_loader.load_data()
            # print('> {} - {} ({}/{})'.format(_data_name, name, i+1, test_loader.size))
            
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            
            # Measure inference time for each image
            start_time = time.time()
            with torch.no_grad():
                result = model(image)
            torch.cuda.synchronize()
            inference_time = time.time() - start_time
            total_time += inference_time
            
            # Handle different result formats
            if isinstance(result, (list, tuple)):
                res = result[4] if len(result) > 4 else result[-1]
            else:
                res = result
                
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            # cv2.imwrite(save_path+name, res*255)
        
        avg_time = total_time / test_loader.size
        print(f"Average inference time per image: {avg_time*1000:.2f} ms")
        print(f"Average FPS: {1/avg_time:.2f}")
        print(f"Results saved to: {save_path}")
    
    # Summary
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print(f"Model: GLCONet")
    print(f"Input Size: {opt.testsize}x{opt.testsize}")
    print(f"Total Parameters: {complexity_results['total_params']:,}")
    print(f"Model Size: {complexity_results['model_size_mb']:.2f} MB")
    print(f"Average Inference Time: {complexity_results['timing']['gpu_mean']:.2f} ms")
    print(f"FPS: {complexity_results['timing']['fps']:.2f}")
    print(f"Peak Memory Usage: {complexity_results['memory']['memory_peak_mb']:.2f} MB")

if __name__ == '__main__':
    main()