import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import cv2
import json
from lib.GLCONet_PVT import Network, AblationConfig
from utils.data_val import test_dataset

# Set GPU device
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

def calculate_mae(pred, gt):
    """Calculate Mean Absolute Error"""
    return np.mean(np.abs(pred - gt))

def test_single_model(model_type, weight_path, test_datasets, opt):
    """Test a single model on all datasets"""
    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print(f"Testing {model_type} model")
    print(f"Weight path: {weight_path}")
    print(f"{'='*60}")
    
    # Create model based on type
    if model_type == 'original':
        model = Network(channels=opt.channels)
    else:
        model = AblationConfig.get_model(model_type, channels=opt.channels)
    
    model = model.to(device)
    
    # Load model weights
    if not load_model_state_dict(model, weight_path, device):
        print(f"Failed to load model weights for {model_type}")
        return None
    
    model.eval()
    
    # Test on each dataset
    results = {}
    
    for dataset_name in test_datasets:
        print(f"\nTesting on {dataset_name}...")
        
        # Setup paths
        data_path = os.path.join(opt.test_dataset_path, dataset_name)
        image_root = os.path.join(data_path, 'Imgs/')
        gt_root = os.path.join(data_path, 'GT/')
        
        # Check if paths exist and validate dataset structure
        if not os.path.exists(image_root) or not os.path.exists(gt_root):
            print(f"Dataset path not found for {dataset_name}")
            print(f"  Image root: {image_root} (exists: {os.path.exists(image_root)})")
            print(f"  GT root: {gt_root} (exists: {os.path.exists(gt_root)})")
            continue
        
        # Check if there are actually images in the directories
        try:
            image_files = [f for f in os.listdir(image_root) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            gt_files = [f for f in os.listdir(gt_root) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"Found {len(image_files)} images and {len(gt_files)} GT files in {dataset_name}")
            
            if len(image_files) == 0 or len(gt_files) == 0:
                print(f"No valid image files found in {dataset_name}, skipping...")
                continue
                
        except Exception as e:
            print(f"Error checking dataset {dataset_name}: {e}")
            continue
        
        # Create save directory
        save_path = os.path.join('./test_results', model_type, dataset_name)
        os.makedirs(save_path, exist_ok=True)
        
        # Create test loader
        test_loader = test_dataset(image_root, gt_root, opt.testsize)
        
        print(f"Found {test_loader.size} test images")
        
        # Process images
        mae_sum = 0
        processed_count = 0
        
        with torch.no_grad():
            for i in range(test_loader.size):
                name = f"image_{i}"  # Default name in case loading fails
                try:
                    # Load data
                    image, gt, name, img_for_post = test_loader.load_data()
                    
                    # Process ground truth
                    gt = np.asarray(gt, np.float32)
                    gt /= (gt.max() + 1e-8)
                    
                    # Move image to device
                    image = image.to(device)
                    
                    # Forward pass
                    result = model(image)
                    
                    # Process result
                    if isinstance(result, (list, tuple)):
                        # Use the final prediction (index 4 based on training script)
                        res = result[4] if len(result) > 4 else result[-1]
                    else:
                        res = result
                    
                    # Resize to match ground truth
                    res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
                    res = res.sigmoid().data.cpu().numpy().squeeze()
                    
                    # Normalize result
                    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                    
                    # Calculate MAE
                    mae = calculate_mae(res, gt)
                    mae_sum += mae
                    processed_count += 1
                    
                    # Save result
                    output_path = os.path.join(save_path, name)
                    cv2.imwrite(output_path, res * 255)
                    
                    if (i + 1) % 50 == 0:
                        print(f'Processed {i+1}/{test_loader.size} images')
                    
                except Exception as e:
                    print(f"Error processing {name}: {e}")
                    print(f"Skipping image {i+1}/{test_loader.size}")
                    continue
        
        # Calculate average MAE
        avg_mae = mae_sum / processed_count if processed_count > 0 else float('inf')
        results[dataset_name] = {
            'mae': avg_mae,
            'processed_images': processed_count,
            'total_images': test_loader.size,
            'save_path': save_path
        }
        
        print(f"{dataset_name} - MAE: {avg_mae:.4f} ({processed_count}/{test_loader.size} images)")
    
    return results

def test_ablation_models():
    """Main testing function for ablation study"""
    parser = argparse.ArgumentParser(description='Test Ablation Models')
    parser.add_argument('--testsize', type=int, default=512, help='testing size')
    parser.add_argument('--ablation_results_path', type=str, default='./ablation_results/',
                        help='path to ablation results directory')
    parser.add_argument('--test_dataset_path', type=str, default=r'D:\BCNet\data/Testset/',
                        help='path to test datasets')
    parser.add_argument('--channels', type=int, default=32, help='model channels parameter')
    parser.add_argument('--device', type=str, default='cuda', help='device to use')
    parser.add_argument('--weight_type', type=str, default='best_model', 
                        choices=['best_model', 'final_model', 'final_checkpoint'],
                        help='which weights to use for testing')
    parser.add_argument('--test_datasets', nargs='*', default=['COD10K'],
                        help='datasets to test on')
    
    # Updated to match your training arguments
    parser.add_argument(
        '--models_to_test', 
        nargs='*', 
        default=[
            'baseline_ecg',
            'baseline_attention',
            'baseline_local',
            'baseline_global',
            'baseline_ecg_attention',
            'baseline_ecg_local_global',
            'full_network'
        ],
        help='ablation models to test'
    )
    
    opt = parser.parse_args()
    
    # Setup device
    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"\n{'='*80}")
    print("ABLATION MODEL TESTING")
    print(f"{'='*80}")
    print(f"Test datasets: {opt.test_datasets}")
    print(f"Models to test: {opt.models_to_test}")
    print(f"Weight type: {opt.weight_type}")
    print(f"Results path: {opt.ablation_results_path}")
    
    # Test each ablation model
    all_test_results = {}
    failed_models = []
    
    for model_type in opt.models_to_test:
        # Find weight file
        model_dir = os.path.join(opt.ablation_results_path, model_type)
        
        if not os.path.exists(model_dir):
            print(f"Model directory not found: {model_dir}")
            failed_models.append(model_type)
            continue
        
        # Determine weight file path
        if opt.weight_type == 'final_checkpoint':
            weight_file = os.path.join(model_dir, 'weights', 'final_checkpoint.pth')
        else:
            weight_file = os.path.join(model_dir, 'weights', f'{opt.weight_type}.pth')
        
        if not os.path.exists(weight_file):
            print(f"Weight file not found: {weight_file}")
            failed_models.append(model_type)
            continue
        
        # Test this model
        try:
            test_results = test_single_model(model_type, weight_file, opt.test_datasets, opt)
            
            if test_results:
                all_test_results[model_type] = test_results
                print(f"✓ {model_type} tested successfully")
            else:
                failed_models.append(model_type)
                print(f"✗ {model_type} testing failed")
        except Exception as e:
            print(f"✗ Error testing {model_type}: {e}")
            failed_models.append(model_type)
            continue
    
    # Save and print comprehensive results
    if all_test_results:
        # Save results
        results_file = os.path.join('./test_results', 'ablation_test_results.json')
        os.makedirs('./test_results', exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(all_test_results, f, indent=2)
        
        # Print summary table
        print(f"\n{'='*80}")
        print("TESTING RESULTS SUMMARY")
        print(f"{'='*80}")
        
        # Print header
        datasets = opt.test_datasets
        header = f"{'Model':<25}"
        for dataset in datasets:
            header += f" {dataset:<10}"
        header += f" {'Avg MAE':<10}"
        print(header)
        print("-" * len(header))
        
        # Print results for each model
        for model_type, model_results in all_test_results.items():
            row = f"{model_type:<25}"
            maes = []
            
            for dataset in datasets:
                if dataset in model_results:
                    mae = model_results[dataset]['mae']
                    row += f" {mae:<10.4f}"
                    maes.append(mae)
                else:
                    row += f" {'N/A':<10}"
            
            # Calculate average MAE
            avg_mae = np.mean(maes) if maes else float('inf')
            row += f" {avg_mae:<10.4f}"
            print(row)
        
        # Find best model
        avg_maes = {}
        for model_type, model_results in all_test_results.items():
            maes = [model_results[dataset]['mae'] for dataset in datasets 
                   if dataset in model_results and model_results[dataset]['mae'] != float('inf')]
            avg_maes[model_type] = np.mean(maes) if maes else float('inf')
        
        if avg_maes:
            best_model = min(avg_maes.keys(), key=lambda x: avg_maes[x])
            print(f"\n🏆 Best performing model: {best_model} (Avg MAE: {avg_maes[best_model]:.4f})")
        
        print(f"\n📁 Detailed results saved to: {results_file}")
        print(f"📁 Output images saved to: ./test_results/")
        
        # Load training results for comparison
        comparison_file = os.path.join(opt.ablation_results_path, 'ablation_comparison.json')
        if os.path.exists(comparison_file):
            try:
                with open(comparison_file, 'r') as f:
                    training_results = json.load(f)
                
                print(f"\n{'='*60}")
                print("TRAINING vs TESTING COMPARISON")
                print(f"{'='*60}")
                print(f"{'Model':<25} {'Train MAE':<12} {'Test MAE':<12} {'Difference':<12}")
                print("-" * 80)
                
                for model_type in opt.models_to_test:
                    if model_type in training_results and model_type in all_test_results:
                        train_mae = training_results[model_type]['best_mae']
                        test_maes = [all_test_results[model_type][dataset]['mae'] 
                                   for dataset in datasets if dataset in all_test_results[model_type]]
                        test_mae = np.mean(test_maes) if test_maes else float('inf')
                        diff = test_mae - train_mae
                        
                        print(f"{model_type:<25} {train_mae:<12.4f} {test_mae:<12.4f} {diff:<12.4f}")
            except Exception as e:
                print(f"Could not load training results: {e}")
    
    else:
        print("No models were successfully tested.")
    
    if failed_models:
        print(f"\n❌ Failed models: {failed_models}")
    
    print(f"\n{'='*80}")
    print("TESTING COMPLETED!")
    print(f"{'='*80}")

if __name__ == '__main__':
    test_ablation_models()