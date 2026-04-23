import os
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from utils.data_val import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, get_coef, cal_ual
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from torch import optim
import json

# Import ablation models
from lib.LiteCOD_PVT import AblationConfig

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def train(train_loader, model, optimizer, epoch, save_path, writer, step_name):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    
    for i, (images, gts, edges) in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images = images.to(device)
        gts = gts.to(device)

        preds = model(images)

        ual_coef = get_coef(iter_percentage=i/total_step, method='cos')
        ual_loss = cal_ual(seg_logits=preds[4], seg_gts=gts)
        ual_loss *= ual_coef

        loss_init = structure_loss(preds[0], gts)*0.25 + structure_loss(preds[1], gts)*0.25 + structure_loss(preds[2], gts)*0.25 + structure_loss(preds[3], gts)*0.5                        
        loss_final = structure_loss(preds[4], gts)

        loss = loss_init + loss_final + 4 * ual_loss
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        step += 1
        epoch_step += 1
        loss_all += loss.data

        if i % 20 == 0 or i == total_step or i == 1:
            print('{} [{}] Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.format(
                datetime.now(), step_name, epoch, opt.epoch, i, total_step, loss.data))
            
            # Log to tensorboard
            writer.add_scalars('Loss_Statistics', 
                             {'Loss_init': loss_init.data, 'Loss_final': loss_final.data, 'Loss_total': loss.data},
                             global_step=step)

    loss_all /= epoch_step
    writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
    
    return loss_all.item()

def val(test_loader, model, epoch, step_name, writer):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.to(device)

            result = model(image)

            res = F.interpolate(result[4], size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', mae, global_step=epoch)
        
        print('[{}] Epoch: {}, MAE: {:.4f}, Best MAE: {:.4f}'.format(step_name, epoch, mae, best_mae))
        
        if mae < best_mae:
            best_mae = mae
            best_epoch = epoch
            
        return mae

def run_ablation_step(step_name):
    global best_mae, best_epoch, step
    
    print(f"\n{'='*60}")
    print(f"Running {step_name}: {AblationConfig.get_description(step_name)}")
    print(f"{'='*60}")
    
    # Create model
    model = AblationConfig.get_model(step_name, channels=opt.channels)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=30, eta_min=1e-6)
    
    # Create save directory
    step_save_path = os.path.join(opt.save_path, step_name)
    os.makedirs(step_save_path, exist_ok=True)
    
    # Create weights subdirectory
    weights_path = os.path.join(step_save_path, 'weights')
    os.makedirs(weights_path, exist_ok=True)
    
    writer = SummaryWriter(os.path.join(step_save_path, 'summary'))
    
    # Setup logging for this step
    log_file = os.path.join(step_save_path, f'{step_name}_training.log')
    logging.basicConfig(
        filename=log_file,
        format='[%(asctime)s-%(levelname)s:%(message)s]',
        level=logging.INFO,
        filemode='w',
        datefmt='%Y-%m-%d %I:%M:%S %p'
    )
    
    logging.info(f"Starting {step_name} ablation experiment")
    logging.info(f"Model: {AblationConfig.get_description(step_name)}")
    logging.info(f"Total Parameters: {total_params:,}")
    logging.info(f"Configuration: lr={opt.lr}, batch_size={opt.batchsize}, epochs={opt.epoch}")
    
    # Reset variables
    step = 0
    best_mae = 1.0
    best_epoch = 0
    
    results = {
        'epochs': [],
        'train_losses': [],
        'val_maes': [],
        'learning_rates': []
    }
    
    print(f"Training for {opt.epoch} epochs...")
    
    # Training loop
    for epoch in range(1, opt.epoch + 1):
        # Update learning rate
        cosine_schedule.step()
        current_lr = cosine_schedule.get_last_lr()[0]
        writer.add_scalar('learning_rate', current_lr, global_step=epoch)
        
        # Train
        train_loss = train(train_loader, model, optimizer, epoch, step_save_path + '/', writer, step_name)
        
        # Validate every few epochs
        if epoch % max(1, opt.epoch // 10) == 0 or epoch == opt.epoch:
            mae = val(val_loader, model, epoch, step_name, writer)
            
            # Store results
            results['epochs'].append(epoch)
            results['train_losses'].append(train_loss)
            results['val_maes'].append(mae)
            results['learning_rates'].append(current_lr)
            
            # Save best model
            if mae == best_mae:
                torch.save(model.state_dict(), os.path.join(weights_path, 'best_model.pth'))
                logging.info(f'New best model saved at epoch {epoch} with MAE: {mae:.4f}')
                print(f'[{step_name}] New best model saved! MAE: {mae:.4f}')
            
            # Save checkpoint every 20% of training
            if epoch % max(1, opt.epoch // 5) == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': cosine_schedule.state_dict(),
                    'best_mae': best_mae,
                    'mae': mae,
                    'step_name': step_name
                }
                torch.save(checkpoint, os.path.join(weights_path, f'checkpoint_epoch_{epoch}.pth'))
                
            logging.info(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val MAE: {mae:.4f}, Best MAE: {best_mae:.4f}')
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(weights_path, 'final_model.pth'))
    
    # Save final checkpoint
    final_checkpoint = {
        'epoch': opt.epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': cosine_schedule.state_dict(),
        'best_mae': best_mae,
        'best_epoch': best_epoch,
        'step_name': step_name,
        'total_params': total_params,
        'training_history': results
    }
    torch.save(final_checkpoint, os.path.join(weights_path, 'final_checkpoint.pth'))
    
    writer.close()
    
    # Save results
    final_results = {
        'step_name': step_name,
        'description': AblationConfig.get_description(step_name),
        'components': AblationConfig.get_components(step_name),
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': total_params * 4 / (1024**2),  # Approximate size in MB
        'best_mae': best_mae,
        'best_epoch': best_epoch,
        'final_mae': results['val_maes'][-1] if results['val_maes'] else None,
        'training_history': results,
        'weights_saved': {
            'best_model': os.path.join(weights_path, 'best_model.pth'),
            'final_model': os.path.join(weights_path, 'final_model.pth'),
            'final_checkpoint': os.path.join(weights_path, 'final_checkpoint.pth'),
            'checkpoints': [os.path.join(weights_path, f'checkpoint_epoch_{e}.pth') 
                          for e in range(40, opt.epoch + 1, 40)]
        }
    }
    
    # Save results as JSON
    with open(os.path.join(step_save_path, 'results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n[{step_name}] COMPLETED:")
    print(f"  Best MAE: {best_mae:.4f} at epoch {best_epoch}")
    print(f"  Final MAE: {final_results['final_mae']:.4f}" if final_results['final_mae'] else "  Final MAE: N/A")
    print(f"  Parameters: {total_params/1e6:.2f}M")
    print(f"  Model Size: {final_results['model_size_mb']:.1f} MB")
    print(f"  Weights saved in: {weights_path}")
    
    # Log completion
    logging.info(f"Training completed for {step_name}")
    logging.info(f"Best MAE: {best_mae:.4f} at epoch {best_epoch}")
    logging.info(f"Final MAE: {final_results['final_mae']:.4f}" if final_results['final_mae'] else "Final MAE: N/A")
    logging.info(f"Weights saved in: {weights_path}")
    
    # Clean up memory
    del model
    torch.cuda.empty_cache()
    
    return final_results

def print_system_info():
    """Print system information"""
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"GPU Count: {gpu_count}")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("GPU: Not available")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Ablation Study for LiteCOD')
    parser.add_argument('--epoch', type=int, default=162, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=512, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.2, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=60, help='every n epochs decay learning rate')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
    parser.add_argument('--train_root', type=str, default=r'D:\BCNet\data/Trainset/',
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default=r'D:\BCNet\data/Testset/',
                        help='the test rgb images root')
    parser.add_argument('--save_path', type=str, default='./ablation_results/',
                        help='the path to save results')
    parser.add_argument('--channels', type=int, default=32, help='number of channels')

    # Updated step names to match AblationConfig keys
    parser.add_argument(
        '--steps',
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
        help='ablation steps to run'
    )

    parser.add_argument(
        '--single_step',
        type=str,
        default=None,
        help='run only a single ablation step'
    )

    opt = parser.parse_args()

    # Setup device
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    cudnn.benchmark = True
    
    print(f'Using GPU {opt.gpu_id}')
    print_system_info()
    
    # Create save directory
    os.makedirs(opt.save_path, exist_ok=True)

    # Load data
    print('\nLoading data...')
    train_loader = get_loader(
        image_root=opt.train_root + 'Imgs/',
        gt_root=opt.train_root + 'GT/',
        edge_root=opt.train_root + 'Edge/',
        batchsize=opt.batchsize,
        trainsize=opt.trainsize,
        num_workers=4
    )
    val_loader = test_dataset(
        image_root=opt.val_root + 'COD10K/Imgs/',
        gt_root=opt.val_root + 'COD10K/GT/',
        testsize=opt.trainsize
    )
    total_step = len(train_loader)
    
    print(f"Training samples: {len(train_loader)} batches")
    print(f"Validation samples: {val_loader.size} images")
    
    # Print plan
    # AblationConfig.print_ablation_plan()
    
    # Determine steps to run
    if opt.single_step:
        steps_to_run = [opt.single_step]
        print(f"\nRunning single step: {opt.single_step}")
    else:
        steps_to_run = opt.steps
        print(f"\nRunning ablation steps: {steps_to_run}")
    
    print(f"Epochs per step: {opt.epoch}")
    print(f"Channels: {opt.channels}")
    print(f"Batch size: {opt.batchsize}")
    print(f"Learning rate: {opt.lr}")
    
    # Run ablation study
    print(f"\n{'='*80}")
    print("STARTING ABLATION STUDY")
    print(f"{'='*80}")
    
    all_results = {}
    failed_steps = []
    
    for i, step_name in enumerate(steps_to_run):
        try:
            print(f"\n[{i+1}/{len(steps_to_run)}] Starting {step_name}...")
            results = run_ablation_step(step_name)
            all_results[step_name] = results
            print(f"✓ {step_name} completed successfully")
        except Exception as e:
            print(f"✗ Error in {step_name}: {e}")
            failed_steps.append(step_name)
            import traceback
            traceback.print_exc()
            continue
    
    # Save comprehensive comparison
    comparison_file = os.path.join(opt.save_path, 'ablation_comparison.json')
    with open(comparison_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("ABLATION STUDY SUMMARY")
    print(f"{'='*80}")
    
    if all_results:
        print(f"{'Step':<15} {'Best MAE':<10} {'Final MAE':<10} {'Params(M)':<10} {'Size(MB)':<10} {'Description'}")
        print("-" * 80)
        
        for step_name, results in all_results.items():
            best_mae = results['best_mae']
            final_mae = results.get('final_mae', 'N/A')
            params = results['total_params'] / 1e6
            size = results['model_size_mb']
            desc = results['description'][:25] + "..." if len(results['description']) > 25 else results['description']
            
            final_mae_str = f"{final_mae:.4f}" if final_mae != 'N/A' else "N/A"
            print(f"{step_name:<15} {best_mae:<10.4f} {final_mae_str:<10} {params:<10.1f} {size:<10.1f} {desc}")
        
        # Find best performing model
        best_step = min(all_results.keys(), key=lambda x: all_results[x]['best_mae'])
        best_mae_overall = all_results[best_step]['best_mae']
        print(f"\n🏆 Best performing model: {best_step} (MAE: {best_mae_overall:.4f})")
        
        # Calculate improvements
        if 'baseline' in all_results and 'full' in all_results:
            baseline_mae = all_results['baseline']['best_mae']
            full_mae = all_results['full']['best_mae']
            improvement = ((baseline_mae - full_mae) / baseline_mae) * 100
            print(f"📈 Overall improvement over baseline: {improvement:.1f}%")
        
    else:
        print("No successful ablation experiments completed.")
    
    if failed_steps:
        print(f"\n❌ Failed steps: {failed_steps}")
    
    print(f"\n📁 All results saved to: {opt.save_path}")
    print(f"📊 Comparison file: {comparison_file}")
    print(f"\n{'='*80}")
    print("ABLATION STUDY COMPLETED!")
    print(f"{'='*80}")