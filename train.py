import hdf5storage
import torch
import argparse
import os
import time
import torch.multiprocessing as mp
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from getdataset import TrainDataset_V1, ValidDataset_V1
from my_utils import AverageMeter, initialize_logger, save_checkpoint, Loss_RMSE, Loss_PSNR, Loss_TV, Loss_MRAE, Loss_SAM
from DataProcess import Data_Process
import torch.utils.data
from architecture import model_generator
import numpy as np
import torch.nn as nn
import shutil
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


parser = argparse.ArgumentParser(description="Model training of HyperspecI-V1")
parser.add_argument("--method", type=str, default='srnet', help='Model')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument("--end_epoch", type=int, default=200, help="number of epochs")
parser.add_argument("--epoch_sam_num", type=int, default=4000, help="per_epoch_iteration")
parser.add_argument("--init_lr", type=float, default=4e-4, help="initial learning rate")
parser.add_argument("--gpu_id", type=int, nargs='+', default=[0,1,2], help='select gpu')
parser.add_argument("--load_check_point", type=lambda x: x.lower() == 'true', default=False, help='loading for training more epochs')
parser.add_argument("--pretrained_model_path", type=str, default=None, help='pre-trained model path')

parser.add_argument("--sigma", type=float, default=(0, 1/ 255, 2/255, 3/255), help="Sigma of Gaussian Noise")
parser.add_argument("--mask_path", type=str, default=None, help='path of calibrated sensing matrix')

parser.add_argument("--output_folder", type=str, default=None, help='output path')
parser.add_argument("--validate_inference_dir", type=str, default=None, help='path of inference result')

parser.add_argument("--start_dir", type=int, default=(0, 0), help="size of test image coordinate")
parser.add_argument("--image_size", type=int, default=(2048, 2048), help="size of test image")
parser.add_argument("--train_patch_size", type=int, default=(512, 512), help="size of patch")
parser.add_argument("--valid_patch_size", type=int, default=(512, 512), help="size of patch")

parser.add_argument("--train_data_path", type=str, default=None, help='path datasets')
parser.add_argument("--valid_data_path", type=str, default=None, help='path datasets')
parser.add_argument("--poisson_flag", type=lambda x: x.lower() == 'true', default=False, help='poisson nosie model')
parser.add_argument("--gaussion_flag", type=lambda x: x.lower() == 'true', default=True, help='gassion nosie model')



def worker_main(rank, world_size, opt, gpu_ids):
    local_rank = gpu_ids[rank]
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    device = torch.device("cuda", local_rank)

    # 创建输出目录
    output_path = opt.output_folder
    if rank == 0:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        log_dir = os.path.join(output_path, 'train.log')
        logger = initialize_logger(log_dir)
        print(f"Working on GPU IDs: {gpu_ids}")
    else:
        logger = None

    criterion_rmse = Loss_RMSE().to(device)
    criterion_psnr = Loss_PSNR().to(device)
    criterion_mrae = Loss_MRAE().to(device)
    criterion_sam = Loss_SAM().to(device)
    criterion_tv = Loss_TV(TVLoss_weight=float(0.5)).to(device)  # 这里TVLoss_weight是该loss的权重，默认给0.5
    data_processing = Data_Process()


    mask_init = hdf5storage.loadmat(opt.mask_path)['mask']
    mask = mask_init[:, opt.start_dir[0]:opt.start_dir[0]+opt.image_size[0], opt.start_dir[1]:opt.start_dir[1] + opt.image_size[1]]
    mask = np.maximum(mask, 0)
    mask = mask / mask.max()
    mask = torch.from_numpy(mask)
    mask = mask.float().to(device)
    if rank == 0:
        print('mask:', mask.shape)
        print('mask:', mask.dtype, mask.shape, mask.max(), mask.mean(), mask.min())

    cudnn.benchmark = True

    
    train_data = TrainDataset_V1(data_path=opt.train_data_path, patch_size=opt.train_patch_size, arg=True)
    val_data = ValidDataset_V1(data_path=opt.valid_data_path, patch_size=opt.valid_patch_size, arg=True)
    if rank == 0:
        print("\nloading dataset ...")
        print('Train dataset:', opt.train_data_path)
        print('len(train_data):', len(train_data))
        print(f"Iteration per epoch: {len(train_data)}")
        print('Valid dataset:', opt.valid_data_path)
        print('len(valid_data):', len(val_data))
        print('poisson model:', opt.poisson_flag, 'gassion model:', opt.gaussion_flag)
    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_data, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size,
                              num_workers=8, pin_memory=True, drop_last=True, sampler=train_sampler)
    val_loader = DataLoader(dataset=val_data, batch_size=1, 
                            num_workers=4, pin_memory=True, sampler=val_sampler)

    script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在目录
    train_py_path = os.path.join(script_dir, 'train.py')
    train_sh_path = os.path.join(script_dir, 'train.sh')
    
    if rank ==0:
        print("Backing up training scripts...")
        if os.path.exists(train_py_path):
            shutil.copy(train_py_path, os.path.join(output_path, 'train.py'))
            print(f"Copied train.py to {output_path}")
        if os.path.exists(train_sh_path):
            shutil.copy(train_sh_path, os.path.join(output_path, 'train.sh'))
            print(f"Copied train.sh to {output_path}")

    # 添加记录每个epoch的损失值的列表
    epochs = []
    train_losses = []
    rmse_losses = []
    psnr_losses = []
    mrae_losses = []
    sam_losses = []
    
    per_epoch_iteration = opt.epoch_sam_num // opt.batch_size
    total_iteration = per_epoch_iteration*opt.end_epoch

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model = model_generator(opt.method, opt.pretrained_model_path).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    start_epoch = 0
    iteration = 0
    iteration = start_epoch * per_epoch_iteration
    
    from collections import OrderedDict # 确保引入了这个库

    if opt.pretrained_model_path is not None and opt.load_check_point == True:
        if rank == 0:
            print(f"Loading checkpoint from {opt.pretrained_model_path}")
        
        checkpoint = torch.load(opt.pretrained_model_path, map_location=device)

        raw_state_dict = None
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            raw_state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
            raw_state_dict = checkpoint['model']
        elif isinstance(checkpoint, dict) and 'net' in checkpoint:
            raw_state_dict = checkpoint['net']
        else:
            raw_state_dict = checkpoint

        new_state_dict = OrderedDict()
        for k, v in raw_state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        try:
            # 这里的 model.module 指向被 DDP 包裹前的原始模型
            model.module.load_state_dict(new_state_dict)
            if rank == 0:
                print("State dict loaded successfully.")
        except Exception as e:
            print(f"Error loading state dict: {e}")
            raise e

        if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
            iteration = start_epoch * per_epoch_iteration
            if rank == 0:
                print(f"Resuming from epoch {start_epoch}")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.init_lr,
                                 betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration - iteration, eta_min=1e-6)

    record_rmse_loss = 10000.0
    strat_time = time.time()
    
    for epoch in range(start_epoch, opt.end_epoch):
        model.train()
        train_sampler.set_epoch(epoch)
        if 'set_epoch' in dir(val_sampler):
            val_sampler.set_epoch(epoch)
        losses = AverageMeter()
        epoch_start_time = time.time()

        for i, (HSIs, sample_name) in enumerate(train_loader):
            batch_start_time = time.time()
            if i >= per_epoch_iteration:
                break
            HSIs = HSIs.to(device, non_blocking=True)
            HSIs = HSIs[:,:,:,:]
            mask_patch = data_processing.get_random_mask_patches(mask=mask, 
                        image_size=opt.image_size, patch_size=opt.train_patch_size, 
                        batch_size=opt.batch_size)
            inputs, targets = data_processing.get_mos_poisson_gaussion_hsi(hsi=HSIs, mask=mask_patch, 
                            sigma=opt.sigma, mos_size=opt.train_patch_size[0], 
                            hsi_input_size=opt.train_patch_size[0], 
                            hsi_target_size=opt.train_patch_size[0],
                            poisson_flag=opt.poisson_flag, gaussion_flag=opt.gaussion_flag)

            inputs = Variable(inputs)
            targets = Variable(targets)
 
            lr = optimizer.param_groups[0]['lr']
            outputs = model(inputs, mask_patch)

            loss_rmse = criterion_rmse(outputs, targets)
            loss_tv = criterion_tv(outputs, targets) 
            loss_mrae = criterion_mrae(outputs, targets)
            loss = loss_rmse  + loss_tv + 0.02 * loss_mrae 

            loss_sam = criterion_sam(outputs, targets)
            if rank == 0 and i % 10 == 0:
                print(f"Train loss: {loss.data:.9f}, RMSE: {loss_rmse.data:.9f}, TV: {loss_tv.data:.9f}, MRAE: {loss_mrae.data:.9f}, SAM: {loss_sam.mean().data:.9f}")

            loss.backward()
            optimizer.step() 
            optimizer.zero_grad() 
            scheduler.step() 
            losses.update(loss.data)
            iteration = iteration + 1
            batch_time = time.time() - batch_start_time

        epoch_time = time.time() - epoch_start_time
        rmse_loss, psnr_loss, mrae_loss, sam_loss = Validate(val_loader, model, mask, epoch, opt.validate_inference_dir,
                                                                data_processing, opt, device, world_size, rank,
                                                                criterion_rmse, criterion_psnr, criterion_mrae, criterion_sam)
        
        # 保存模型
        if rank == 0:
            # Use Python abs() for float comparison
            if abs(record_rmse_loss - rmse_loss) < 0.0001 or rmse_loss < record_rmse_loss:
                print(f'Saving to {output_path}')
                save_checkpoint(output_path, epoch, iteration, model, optimizer)
                if rmse_loss < record_rmse_loss:
                    record_rmse_loss = rmse_loss

            print("Epoch[%06d/%06d], Time[%06d], Learning rate: %.9f, Train Loss: %.9f, "
                "Test RMSE: %.9f, Test PSNR: %.9f, Test MRAE: %.9f, Test SAM: %.9f"
                % (epoch, opt.end_epoch, epoch_time, lr, losses.avg, 
                    rmse_loss, psnr_loss, mrae_loss, sam_loss))

            logger.info("Epoch[%06d/%06d], Time[%06d], Learning rate: %.9f, Train Loss: %.9f, "
                    "Test RMSE: %.9f, Test PSNR: %.9f, Test MRAE: %.9f, Test SAM: %.9f"
                    % (epoch, opt.end_epoch, epoch_time, lr, losses.avg, 
                        rmse_loss, psnr_loss, mrae_loss, sam_loss))
                
            # 记录每个epoch的损失值
            epochs.append(epoch)
            train_losses.append(losses.avg.item())
            rmse_losses.append(rmse_loss)          # was rmse_loss.item()
            psnr_losses.append(psnr_loss)          # was psnr_loss.item()
            mrae_losses.append(mrae_loss)          # was mrae_loss.item()
            sam_losses.append(sam_loss)            # was sam_loss.item()
            
            # 重置loss统计
            losses = AverageMeter()
            # 记录新epoch的开始时间
            epoch_start_time = time.time()
    
    if rank == 0:
        save_loss_fig_csv(output_path, epochs, train_losses, rmse_losses, psnr_losses, mrae_losses, sam_losses)
    if dist.is_initialized():
        dist.destroy_process_group()
    

def Validate(val_loader, model, mask, epoch, valid_dir, data_processing, opt, device, world_size, rank,
             criterion_rmse, criterion_psnr, criterion_mrae, criterion_sam):
    model.eval()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    losses_mrae = AverageMeter()
    losses_sam = AverageMeter()
    num = 0

    random_index = np.random.randint(0, len(val_loader))

    for i, (HSIs) in enumerate(val_loader):
        HSIs = HSIs.to(device, non_blocking=True)
        HSIs = HSIs[:,:,:,:]

        mask_patch = data_processing.get_random_mask_patches(
            mask=mask, image_size=opt.image_size, patch_size=opt.valid_patch_size, batch_size=1
        )
        
        inputs, targets = data_processing.get_mos_poisson_gaussion_hsi(
            hsi=HSIs, mask=mask_patch, sigma=opt.sigma, mos_size=opt.valid_patch_size[0],
            hsi_input_size=opt.valid_patch_size[0], hsi_target_size=opt.valid_patch_size[0],
            poisson_flag=opt.poisson_flag, gaussion_flag=opt.gaussion_flag
        )

        with torch.no_grad():
            outputs = model(inputs, mask_patch)

            loss_rmse = criterion_rmse(outputs, targets)
            loss_psnr = criterion_psnr(outputs, targets)
            loss_mrae = criterion_mrae(outputs, targets)
            loss_sam = criterion_sam(outputs, targets)
            losses_psnr.update(loss_psnr.data)
            losses_rmse.update(loss_rmse.data)
            losses_mrae.update(loss_mrae.data)
            losses_sam.update(loss_sam.data)

        if rank == 0:
            if i == random_index:
                bmp_data = inputs.squeeze()
                bmp_data = bmp_data.cpu().numpy()
                bmp_data_scaled = (bmp_data / (bmp_data.max()) * 255).astype(np.uint8)
                base_name = 'epoch' + str(epoch) + '_val_' + str(i) + '_mos' 
                save_folder_bmp = valid_dir
                if not os.path.exists(save_folder_bmp):
                    os.makedirs(save_folder_bmp)
                bmp_path = os.path.join(save_folder_bmp, f"{base_name}.bmp")
                cv2.imwrite(bmp_path, bmp_data_scaled)
                print(f"Saved BMP to: {bmp_path}")

                rgb_output = get_rgb_from_hsi(outputs.detach().cpu().numpy().squeeze())
                rgb_output = (rgb_output * 255).astype(np.uint8)
                base_name = 'epoch' + str(epoch) + '_val_' + str(i) + '_output' 
                png_path = os.path.join(save_folder_bmp, f"{base_name}.png")
                plt.imsave(png_path, rgb_output)

                rgb_target = get_rgb_from_hsi(targets.detach().cpu().numpy().squeeze())
                rgb_target = (rgb_target * 255).astype(np.uint8)
                base_name = 'epoch' + str(epoch) + '_val_' + str(i) + '_target' 
                png_path = os.path.join(save_folder_bmp, f"{base_name}.png")
                plt.imsave(png_path, rgb_target)

    metrics = torch.tensor([losses_rmse.avg, losses_psnr.avg, losses_mrae.avg, losses_sam.avg]).to(device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    final_metrics = metrics / world_size


    return final_metrics[0].item(), final_metrics[1].item(), final_metrics[2].item(), final_metrics[3].item()

def get_rgb_from_hsi(hsi):
        if hsi.shape[0] < 26:
            print(f"Warning: HSI has only {hsi.shape[0]} channels, using alternative RGB conversion")
            # 使用前三个通道作为RGB
            if hsi.shape[0] >= 3:
                r = hsi[0]
                g = hsi[1]
                b = hsi[2]
            else:
                r = g = b = hsi[0]
        else:
            r_idx = min(20, hsi.shape[0]-1)  # 670nm
            g_idx = min(12, hsi.shape[0]-1)  # 600nm
            b_idx = min(5, hsi.shape[0]-1)   # 510nm
            
            r = hsi[r_idx]
            g = hsi[g_idx]
            b = hsi[b_idx]
        
        rgb = np.stack([r, g, b], axis=-1)
        
        max_val = np.max(rgb)
        if max_val > 0:
            rgb = rgb / max_val
        
        rgb = np.clip(rgb, 0, 1)
        
        # 提亮显示
        brightness_factor = 1.0 # b 5.0
        rgb = np.clip(rgb * brightness_factor, 0, 1)
        
        return rgb

def save_loss_fig_csv(output_path, epochs, train_losses, rmse_losses, psnr_losses, mrae_losses, sam_losses):
    loss_data = {
        'Epoch': epochs,
        'Train_Loss': train_losses,
        'RMSE': rmse_losses,
        'PSNR': psnr_losses,
        'MRAE': mrae_losses,
        'SAM': sam_losses
    }
    df = pd.DataFrame(loss_data)
    csv_path = os.path.join(output_path, 'training_losses.csv')
    df.to_csv(csv_path, index=False)
    print(f"保存训练损失数据到: {csv_path}")
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 3, 2)
    plt.plot(epochs, rmse_losses, 'r-', label='RMSE')
    plt.title('RMSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 3, 3)
    plt.plot(epochs, psnr_losses, 'g-', label='PSNR')
    plt.title('PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 3, 4)
    plt.plot(epochs, mrae_losses, 'm-', label='MRAE')
    plt.title('MRAE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MRAE')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 3, 5)
    plt.plot(epochs, sam_losses, 'c-', label='SAM')
    plt.title('SAM Loss')
    plt.xlabel('Epoch')
    plt.ylabel('SAM')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(output_path, 'training_curves.png')
    plt.savefig(plot_path, dpi=300)
    print(f"保存训练曲线图到: {plot_path}")
    plt.close()  # close multi-plot figure
    
    metrics = {
        'train_loss': train_losses,
        'rmse': rmse_losses,
        'psnr': psnr_losses,
        'mrae': mrae_losses,
        'sam': sam_losses
    }
    
    for name, values in metrics.items():
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, values, '-o')
        plt.title(f'{name.upper()} Curve')
        plt.xlabel('Epoch')
        plt.ylabel(name.upper())
        plt.grid(True)
        plt.tight_layout()
        single_plot_path = os.path.join(output_path, f'{name}_curve.png')
        plt.savefig(single_plot_path, dpi=300)
        print(f"保存 {name} 曲线图到: {single_plot_path}")
        plt.close()  # close per-metric figure

if __name__ == '__main__':

    opt = parser.parse_args()
    if opt.gpu_id is None:
        raise ValueError(f"错误: 未指定gpu id!")
    else:
        gpu_ids = opt.gpu_id
        max_gpu_id = torch.cuda.device_count() - 1
        for gpu_id in gpu_ids:
            if gpu_id > max_gpu_id:
                raise ValueError(f"错误：请求的GPU ID {gpu_id} 无效。此服务器上可用的最大GPU ID为 {max_gpu_id}。")

    world_size = len(gpu_ids)
    if world_size == 0:
        raise ValueError("没有可用的或指定的GPU用于训练。")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345' 
    
    # Ensure multiprocessing uses 'spawn' everywhere and avoid semaphore-heavy sharing
    import multiprocessing as py_mp  # local import to avoid global side effects
    try:
        py_mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    try:
        torch.multiprocessing.set_sharing_strategy('file_system')
    except Exception:
        pass

    mp.spawn(worker_main,
            args=(world_size, opt, gpu_ids),
            nprocs=world_size,
            join=True)

    # Make sure all resources are cleaned up even if errors occur
    try:
        del train_loader
    except Exception:
        pass
    try:
        del val_loader
    except Exception:
        pass
    try:
        import gc
        gc.collect()
    except Exception:
        pass
    torch.cuda.empty_cache()
    dist.destroy_process_group()




