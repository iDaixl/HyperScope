import torch
import torch.backends.cudnn as cudnn
import numpy as np
from architecture import model_generator
import h5py
import hdf5storage
import cv2
import os
from tqdm import tqdm
import time
from DataProcess import Data_Process
import pandas as pd
import cupy as cp
import matplotlib.pyplot as plt
import tifffile
import imageio


class HSIInference:
    def __init__(self, model_path, mask_path, method='srnet', device='cuda'):
        self.device = device
        self.model = model_generator(method, model_path)
        if device == 'cuda':
            cudnn.benchmark = True
            self.model.cuda()
        self.model.eval()
        
        raw_mask = self.load_mask(mask_path)
        self.mask = self.preprocess_mask(raw_mask)
        self.data_process = Data_Process()
        print(f"Loaded and preprocessed mask shape: {self.mask.shape}")

    def preprocess_mask(self, mask):
        if isinstance(mask, np.ndarray):
            mask = mask / mask.max()  # 归一化
            mask = mask.astype(np.float32)
            mask = torch.from_numpy(mask)
        
        if self.device == 'cuda':
            mask = mask.cuda()
        
        if len(mask.shape) == 3:
            mask = mask.unsqueeze(0)
        
        return mask

    def preprocess_image(self, image):
        if isinstance(image, np.ndarray):
            image = image.astype(np.float32)
            image = image / image.max()  # 归一化到0-1范围
            image = torch.from_numpy(image).float()
        
        if self.device == 'cuda':
            image = image.cuda()
            
        if len(image.shape) == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif len(image.shape) == 3:
            image = image.unsqueeze(0)
            
        return image

    def get_rgb_from_hsi(self, hsi, enhance=False):
        # for multicolor clice
        # # Original: R: 650nm (25), G: 550nm (15), B: 450nm (5)
        r_idx = 28  # 670nm
        g_idx = 18  # 600nm
        b_idx = 5   # 510nm
        
        # 获取对应波段的值
        r = hsi[r_idx]
        g = hsi[g_idx]
        b = hsi[b_idx]
        
        # 堆叠成RGB图像
        rgb = np.stack([r, g, b], axis=-1)
        
        if enhance is not None:
            # 提亮处理
            max_val = rgb.max()
            rgb = (rgb / max_val) * 255.0 / 255.0
            rgb = rgb * enhance
        else:
            # 普通归一化
            rgb = rgb / rgb.max()
            
        rgb = np.clip(rgb, 0, 1)
        return rgb

    @torch.no_grad()
    def inference(self, image, start_dir=(0, 0), image_size=(2048, 2048)):
        h, w = image.shape
        start_dir=np.array(start_dir)
        image_size=np.array(image_size)
        end_dir = start_dir + image_size
        if end_dir[0]>h:
            end_dir[0] = h
            print('Assigned image size larger than original image size\n')
        if end_dir[1]>w:
            end_dir[1] = w
            print('Assigned image size larger than original image size\n')

        print('Image start from', start_dir,', end with', end_dir)
        image = image[start_dir[0]:end_dir[0], start_dir[1]:end_dir[1]]
        print('Image size is: ', image.shape)

        mask = self.mask[:, :, start_dir[0]:end_dir[0], start_dir[1]:end_dir[1]]
        mask = mask / mask.max()
        
        image = self.preprocess_image(image)
        
        outputs = self.model(image, mask)
        
        hsi = outputs.squeeze().cpu().numpy()
        rgb = self.get_rgb_from_hsi(hsi, enhance=1.0)
        
        return hsi, rgb
    

    def load_mask(self, mask_path):
        mask = hdf5storage.loadmat(mask_path)['mask']
        return mask


    def load_image(self, image_path):
        if '.tif' in image_path:
            if 'wo_bg.tif' in image_path:
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                print('dtype:', image.dtype)   # 应该是 uint16
                print('image min max:', image.min(), image.max())
            else:
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                print('dtype:', image.dtype)   # 应该是 uint16
                print('before:', image.min(), image.max())
                image = image - 100 # offset
                image = image.clip(0, None)
                print('after:', image.min(), image.max())
                print('Meet scmos image, -offset 100')
        else:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        print('image:', image.min(), image.max())
        image = image.astype(np.float32)

        image = image / image.max()
        # image = image / 255 * image.max()
        print(f"Loaded image shape: {image.shape}")
        return image

    def load_image_wt_max(self, image_path):
        if '.tif' in image_path:
            if 'wo_bg' in image_path:
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                print('dtype:', image.dtype)   # 应该是 uint16
                print('image min max:', image.min(), image.max())
            else:
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                print('dtype:', image.dtype)   # 应该是 uint16
                print('before:', image.min(), image.max())
                image = image - 100 # offset
                image = image.clip(0, None)
                print('after:', image.min(), image.max())
                print('Meet scmos image, -offset 100')
        else:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        print('image:', image.min(), image.max())
        image = image.astype(np.float32)

        img_max = image.max()
        image = image / img_max
        # image = image / 255 * image.max()
        print(f"Loaded image shape: {image.shape}")
        return image, img_max

    def save_results(self, hsi, rgb, save_dir='./', prefix='output'):
        h5_path = f'{save_dir}/{prefix}_hsi.h5'
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('hsi', data=hsi)
        
        rgb_path = f'{save_dir}/{prefix}_rgb.png'
        rgb_bgr = (rgb * 255).astype(np.uint8)
        rgb_bgr = cv2.cvtColor(rgb_bgr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(rgb_path, rgb_bgr)
        
        print(f"Results saved to '{h5_path}' and '{rgb_path}'")
        return rgb_bgr

    def visualize_rgb(self, rgb, scale=0.25):
        rgb_bgr = (rgb * 255).astype(np.uint8)
        rgb_bgr = cv2.cvtColor(rgb_bgr, cv2.COLOR_RGB2BGR)
        
        # 缩放图像
        h, w = rgb_bgr.shape[:2]
        display_size = (int(w*scale), int(h*scale))
        rgb_bgr_small = cv2.resize(rgb_bgr, display_size, interpolation=cv2.INTER_AREA)
        
        # 显示结果
        cv2.imshow('RGB Result', rgb_bgr_small)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process_single_image(self, image_path, save_dir='./', prefix='output', 
                           start_dir=(0, 0), image_size=(2048, 2048), visualize=True):
        image = self.load_image(image_path)
        
        hsi, rgb = self.inference(image, start_dir, image_size)
        print(f"Output HSI shape: {hsi.shape}")
        print(f"Output RGB shape: {rgb.shape}")
        
        rgb_bgr = self.save_results(hsi, rgb, save_dir, prefix)
        
        if visualize:
            self.visualize_rgb(rgb)
        
        return hsi, rgb 

    def process_folder(self, image_folder, save_dir, start_dir=(0, 0), image_size=(2048, 2048), save_hsi=False, save_gray_image=False):
        hsi_dir = os.path.join(save_dir, 'hsi')
        rgb_dir = os.path.join(save_dir, 'rgb')
        os.makedirs(hsi_dir, exist_ok=True)
        os.makedirs(rgb_dir, exist_ok=True)
        
        image_files = [f for f in os.listdir(image_folder) 
                      if f.lower().endswith(('.bmp', '.jpg', '.png', '.tif'))]
        image_files.sort()
        
        print(f"Found {len(image_files)} images in {image_folder}")
        
        total_time = 0
        load_time = 0
        infer_time = 0
        save_time = 0
        
        for image_file in tqdm(image_files, desc="Processing images"):
            try:
                t_start = time.time()
                
                t0 = time.time()
                image_path = os.path.join(image_folder, image_file)
                image = self.load_image(image_path)
                t1 = time.time()
                print("load image time", t1-t0)
                load_time += t1 - t0
                t0 = time.time()
                hsi, rgb = self.inference(image, start_dir, image_size)
                print(f"推理结果hsi参数 尺寸:{hsi.shape}, 最小值:{hsi.min()}, 最大值:{hsi.max()}")
                hsi_max = hsi.max()
                hsi = hsi / hsi_max
                hsi = hsi.clip(0, 1)
                print(f"推理结果hsi参数 尺寸:{hsi.shape}, 99%归一化后 最小值:{hsi.min()}, 最大值:{hsi.max()}")
                t1 = time.time()
                print("inference time", t1-t0)
                infer_time += t1 - t0
                
                t0 = time.time()
                base_name = os.path.splitext(image_file)[0]
                
                # 保存HSI结果
                h5_path = os.path.join(hsi_dir, f'HSI_R_{base_name}.h5')
                print('hsi shape is:', hsi.shape)
                if save_hsi:
                    with h5py.File(h5_path, 'w') as f:
                        f.create_dataset('hsi_R', data=hsi)
                    t1 = time.time()
                    print("save h5 time", t1-t0)

                if save_gray_image==True:
                    channel_list=[18, 28]
                    spectral_channel_list=[570, 670]
                    for i in range(len(channel_list)):
                        channel = channel_list[i]
                        spectral_channel = spectral_channel_list[i]
                        gray_img = hsi[channel,:,:] / hsi.max()
                        channel_path = os.path.join(rgb_dir, f'{base_name}_channel_{str(spectral_channel)}.png')
                        gray_img = (gray_img * 255).astype(np.uint8)
                        imageio.imwrite(channel_path, gray_img)


                rgb_path = os.path.join(rgb_dir, f'{base_name}_rgb.png')
                rgb_bgr = (rgb * 255).astype(np.uint8)
                rgb_bgr = cv2.cvtColor(rgb_bgr, cv2.COLOR_RGB2BGR)
                cv2.imwrite(rgb_path, rgb_bgr)
                
                t1 = time.time()
                print("save time", t1-t0)
                save_time += t1 - t0
                
                total_time += time.time() - t_start
                
                
            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")
                continue
        
        # 打印性能统计
        print("\nPerformance Statistics:")
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Average time per image: {total_time/len(image_files):.2f}s")
        print(f"Time breakdown:")
        #print(f"  - Loading: {load_time:.2f}s ({100*load_time/total_time:.1f}%)")

        print(f"  - Inference: {infer_time:.2f}s ({100*infer_time/total_time:.1f}%)")
        print(f"  - Saving: {save_time:.2f}s ({100*save_time/total_time:.1f}%)")
        
        print(f"\nResults saved to:")
        print(f"HSI: {hsi_dir}")
        print(f"RGB: {rgb_dir}") 


    def process_folder_with_unmixing_fluor_list(self, image_folder, save_dir, start_dir=(0, 0), image_size=(2048, 2048), save_hsi=False, save_gray_image=False,
                                        save_unmix=True, save_tif=False, endmember_path=None, fluor_list=None):
        if save_hsi:
            hsi_dir = os.path.join(save_dir, 'hsi')
            os.makedirs(hsi_dir, exist_ok=True)

        if save_unmix:
            unmix_dir = os.path.join(save_dir, 'unmix')
            os.makedirs(unmix_dir, exist_ok=True)

        rgb_dir = os.path.join(save_dir, 'rgb')
        os.makedirs(rgb_dir, exist_ok=True)
        
        image_files = [f for f in os.listdir(image_folder) 
                    if f.lower().endswith(('.bmp', '.jpg', '.png', '.tif'))]
        image_files.sort()
        
        print(f"Found {len(image_files)} images in {image_folder}")
        
        total_time = 0
        load_time = 0
        infer_time = 0
        save_time = 0
        
        for image_file in tqdm(image_files, desc="Processing images"):
            try:
                t_start = time.time()
                
                t0 = time.time()
                image_path = os.path.join(image_folder, image_file)
                image = self.load_image(image_path)
                t1 = time.time()
                print("load image time", t1-t0)
                load_time += t1 - t0
                t0 = time.time()
                hsi, rgb = self.inference(image, start_dir, image_size)
                print(f"推理结果hsi参数 尺寸:{hsi.shape}, 最小值:{hsi.min()}, 最大值:{hsi.max()}")
                hsi = hsi.clip(0, 1)
                t1 = time.time()
                print("inference time", t1-t0)
                infer_time += t1 - t0
                
                # 保存结果
                t0 = time.time()
                base_name = os.path.splitext(image_file)[0]
                
                # 保存HSI结果
                if save_hsi:
                    h5_path = os.path.join(hsi_dir, f'HSI_R_{base_name}.h5')
                    print('hsi shape is:', hsi.shape)
                    with h5py.File(h5_path, 'w') as f:
                        f.create_dataset('hsi_R', data=hsi)
                    t1 = time.time()
                    print("save h5 time", t1-t0)

                if save_gray_image==True:
                    channel_list=[6, 11, 18, 21, 27, 29]
                    spectral_channel_list=[450, 510, 570, 600, 650, 700]
                    for i in range(len(channel_list)):
                        channel = channel_list[i]
                        spectral_channel = spectral_channel_list[i]
                        gray_img = hsi[channel,:,:] / hsi.max()
                        channel_path = os.path.join(rgb_dir, f'{base_name}_channel_{str(spectral_channel)}.png')
                        gray_img = (gray_img * 255).astype(np.uint8)
                        imageio.imwrite(channel_path, gray_img)

                rgb_path = os.path.join(rgb_dir, f'{base_name}_rgb.png')
                rgb_bgr = (rgb * 255).astype(np.uint8)
                rgb_bgr = cv2.cvtColor(rgb_bgr, cv2.COLOR_RGB2BGR)
                cv2.imwrite(rgb_path, rgb_bgr)

                if save_unmix==True:
                    recon, inverse = self.spectral_unmixing(hsi, endmember_path, num_iters=2000, inverse=True)
                    unmixed_file = os.path.join(unmix_dir, f'{base_name}_unmixed.tif')
                    if fluor_list == None:
                        fluor_list=['dapi_450', 'opal_480', 'opal_520', 'opal_570', 'opal_620', 'opal_650', 'opal_690']
                        print('使用默认的fluor list: dapi 480 520 570 620 650 690')
                    else:
                        print('fluor list:', fluor_list)
                    
                    fluor_list_len = len(fluor_list)
                    num_unmixing = recon.shape[0]
                    assert fluor_list_len == num_unmixing, '输入的荧光端元和fluor list不匹配'

                    if save_tif:
                        tifffile.imwrite(
                            unmixed_file,
                            recon,
                            # bigtiff=True,
                            imagej=True,
                            metadata={'axes': 'CYX'}   # 明确指定维度顺序：Channel, Y, X
                        )
                        print("\nUnmixed data written to %s" % unmixed_file)
                    else:
                        print('Only save png image.')

                    for c in range(recon.shape[0]): # [z, c, y, x] 改为 [c, y, x]
                        fluor = recon[c]
                        # fluor = (fluor / fluor.max() * 255).astype(np.uint8)
                        fluor = (fluor / recon.max() * 255).astype(np.uint8)
                        # imageio.imwrite(unmixed_file.replace('.ome.tif', f'_channel_{c}.png'), fluor)
                        imageio.imwrite(unmixed_file.replace('.tif', f'_{fluor_list[c]}.png'), fluor)
                                
                    # inversed_file = os.path.join(unmix_dir, f'{base_name}_inversed.ome.tif')
                    # if inverse is not None:
                        # tifffile.imwrite(
                        # inversed_file,
                        # inverse,
                        # # bigtiff=True,
                        # imagej=True,
                        # metadata={'axes': 'CYX'}   # 明确指定维度顺序：Channel, Y, X
                        # )
                        # print("Linearly unmixed data written to %s" % inversed_file)

                        # for c in range(inverse.shape[0]): # [z, c, y, x] 改为 [c, y, x]
                        #     fluor = inverse[c]
                        #     # fluor = (fluor / fluor.max() * 255).astype(np.uint8)
                        #     fluor = (fluor / inverse.max() * 255).astype(np.uint8)
                        #     # imageio.imwrite(inversed_file.replace('.ome.tif', f'_channel_{c}.png'), fluor)
                        #     imageio.imwrite(inversed_file.replace('.ome.tif', f'_{fluor_list[c]}.png'), fluor)
                
                t1 = time.time()
                print("save time", t1-t0)
                save_time += t1 - t0
                
                total_time += time.time() - t_start
                

                
            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")
                continue
        
        # 打印性能统计
        print("\nPerformance Statistics:")
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Average time per image: {total_time/len(image_files):.2f}s")
        print(f"Time breakdown:")
        #print(f"  - Loading: {load_time:.2f}s ({100*load_time/total_time:.1f}%)")

        print(f"  - Inference: {infer_time:.2f}s ({100*infer_time/total_time:.1f}%)")
        print(f"  - Saving: {save_time:.2f}s ({100*save_time/total_time:.1f}%)")
        
        print(f"\nResults saved to:")
        if save_hsi:
            print(f"HSI: {hsi_dir}")
        if save_gray_image:
            print(f"RGB: {rgb_dir}") 
        if save_unmix:
            print(f"Unmixing: {unmix_dir}") 


    def load_hsi(self, h5_path):
        hsi = None
        with h5py.File(h5_path, 'r') as f:
            if 'hsi' in f:
                hsi = f['hsi'][:]
            elif 'hsi_R' in f:
                hsi = f['hsi_R'][:]
            else:
                print(f"No valid dataset found in {h5_path}")
        hsi = hsi.astype(np.float32)
        output_hsi = hsi / hsi.max()
        # 转换为tensor
        output_hsi = np.ascontiguousarray(output_hsi)
        output_hsi = torch.from_numpy(hsi).cuda()
        output_hsi = output_hsi.unsqueeze(0)  # 添加batch维度
        return output_hsi

    def convert_numpy_to_tensor(self, numpy_array):
        pass
    
    def spectral_unmixing(self, hsi, endmember_path, num_iters=2000, inverse=True):
        # Read endmember matrix and calculate transpose
        M = pd.read_csv(endmember_path, header=0)
        M = cp.array(M, dtype=cp.float32, order='F')
        M = M / M.sum(axis=0, keepdims=True)
        MT = cp.array(cp.transpose(M), dtype=cp.float32)
        # if inverse is not None:
        if inverse:
            Mpinv = np.linalg.pinv(M)
            Mpinv = Mpinv.get()

        # mixed = hsi.cpu().numpy()
        mixed = hsi
        mixed = mixed.astype('float32')

        print('Unmixing Input shape: %s' % (mixed.shape, ))
        print('Unmixing endmember shape: %s' % (M.shape, ))
        print('Number of iterations: %d' % num_iters)
        print('Whether using inverse method:', inverse)
        
        if mixed.ndim == 3:
            mixed = np.expand_dims(mixed, axis=0)

        # Reshape mixed data into vector
        num_z = mixed.shape[0]
        num_c = mixed.shape[1]
        num_x = mixed.shape[2]
        num_y = mixed.shape[3]
        mixed = mixed.reshape(num_z, num_c, num_x * num_y) # vectorize x, y -> x * y

        # Precompute HTones
        HTones = cp.matmul(MT, cp.ones_like(mixed[0]),)

        # Calculate Richardson-Lucy iterations
        recon = np.ones((num_z, M.shape[1], num_x * num_y))
        if inverse:
            # if inverse is not None:
            inverse = np.ones_like(recon)
        # start_time = timeit.default_timer()
        for z in range(num_z):
            slice = cp.array(mixed[z])
            recon_slice = cp.ones((M.shape[1], slice.shape[1]), dtype=cp.float32)

            if inverse is not None:
                inverse[z] = np.matmul(Mpinv, mixed[z])
                inverse = inverse.clip(0, np.inf)

            for iter in range(num_iters):
                Hu = cp.matmul(M, recon_slice)
                ratio = slice / (Hu + 1E-12)
                HTratio = cp.matmul(MT, ratio)
                recon_slice = recon_slice * HTratio / HTones
                # recon_slice *= (cp.matmul(MT, slice / (Hu + 1E-12)) / HTones)
            
            recon[z] = recon_slice.get()
            # calc_time = timeit.default_timer() - iter_start_time
            # print("Slice %03d: %d iterations completed in %f s." % (z, num_iters, calc_time))

        # calc_time = timeit.default_timer() - start_time
        # print("Finished in %f s" % calc_time)

        recon = recon.reshape(M.shape[1], num_x, num_y)
        recon = recon.astype(np.float32)

        inverse = inverse.reshape(M.shape[1], num_x, num_y)
        inverse = inverse.astype(np.float32)
        
        return recon, inverse

