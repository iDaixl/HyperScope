import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random

class Data_Process(object):
    def __init__(self):
        self.noise_sigma = 0
        self.hsi_max = []

    def add_noise(self, inputs, sigma):
        noise = torch.zeros_like(inputs)
        noise.normal_(0, sigma)
        noisy = inputs + noise
        noisy = torch.clamp(noisy, 0, 1.0)
        return noisy
    
    def add_poisson_noise(self, inputs):
        scaled_inputs = inputs * 65535.
        
        noisy_scaled = torch.poisson(scaled_inputs)
    
        noisy = noisy_scaled / 65535.
        
        noisy = torch.clamp(noisy, 0, 1.0)
        
        return noisy

    def add_uniform_noise(self, inputs, pepper_sigma, prob):
        noise_gauss = torch.zeros_like(inputs)#生成噪声
        noise_gauss.normal_(0.2, pepper_sigma)

        noise_mask = torch.zeros_like(inputs)#选择噪声
        noise_mask.uniform_(0, 1)

        index = noise_mask < prob

        noise_gauss = noise_gauss * index

        outputs = inputs  + noise_gauss

        outputs = torch.clip(outputs, 0, 1)

        return outputs

    def hsi_add_noise(self, hsi, sigma):
        noise = torch.zeros_like(hsi)
        noise.normal_(0, sigma)
        noisy = hsi + noise
        noisy = torch.clamp(noisy, 0, 1.0)
        return noisy
    
    
    def get_random_mask_patches(self, mask, image_size, patch_size, batch_size):
        masks = []
        for i in range(batch_size):            
            random_h = random.randint(0, image_size[0] - patch_size[0]) 
            random_w = random.randint(0, image_size[1] - patch_size[1])
            mask_patch = mask[:, random_h:random_h + patch_size[0], random_w:random_w + patch_size[1]]

            masks.append(mask_patch)
            
        mask_patches = torch.stack(masks, dim=0)
        return mask_patches
            
        
    def get_mos_hsi(self, hsi, mask, sigma=0, mos_size=2048, hsi_input_size=512, hsi_target_size=512, init_div_rat=8):
        if not hsi_input_size == hsi_target_size:
            hsi_out = self.extend_spatial_resolution(hsi, extend_rate=hsi_target_size / hsi_input_size)
        else:
            hsi_out=hsi

        if not mos_size == hsi_input_size:
            hsi_expand = self.extend_spatial_resolution(hsi, extend_rate=mos_size / hsi_input_size)
        else:
            hsi_expand=hsi


        mos = torch.sum(hsi_expand * mask, dim=1).unsqueeze(1)
        mos_max = torch.max(mos.view(mos.shape[0], -1), 1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)


        output_hsi = hsi_out # / mos_max * init_div_rat
        input_mos = mos / mos_max

        if isinstance(sigma, tuple):
            select_noise_sigma = sigma[random.randint(0, len(sigma) - 1)]
        else: 
            select_noise_sigma = sigma

        input_mos = self.add_noise(input_mos, select_noise_sigma)
        input_mos = input_mos / input_mos.max()



        return input_mos, output_hsi
    
    def get_mos_hsi_low_snr(self, hsi, mask, sigma=0, mos_size=2048, hsi_input_size=512, hsi_target_size=512, init_div_rat=8):
        if not hsi_input_size == hsi_target_size:
            hsi_out = self.extend_spatial_resolution(hsi, extend_rate=hsi_target_size / hsi_input_size)
        else:
            hsi_out=hsi

        if not mos_size == hsi_input_size:
            hsi_expand = self.extend_spatial_resolution(hsi, extend_rate=mos_size / hsi_input_size)
        else:
            hsi_expand=hsi

        hsi_expand = hsi_expand * 0.1

        mos = torch.sum(hsi_expand * mask, dim=1).unsqueeze(1)

        mos_max = torch.max(mos.view(mos.shape[0], -1), 1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)


        output_hsi = hsi_out # / mos_max * init_div_rat

        input_mos = mos / mos_max * 0.15


        if isinstance(sigma, tuple):
            select_noise_sigma = sigma[random.randint(0, len(sigma) - 1)]
        else: 
            select_noise_sigma = sigma

        input_mos = self.add_noise(input_mos, select_noise_sigma)
        input_mos = input_mos.clip(0, 1)
        # input_mos = input_mos / input_mos.max()



        return input_mos, output_hsi

    def get_mos_poisson_gaussion_hsi(self, hsi, mask, sigma=0, mos_size=2048, hsi_input_size=512, hsi_target_size=512, init_div_rat=8,
                                    poisson_flag=False, gaussion_flag=True):
        if not hsi_input_size == hsi_target_size:
            hsi_out = self.extend_spatial_resolution(hsi, extend_rate=hsi_target_size / hsi_input_size)
        else:
            hsi_out=hsi

        if not mos_size == hsi_input_size:
            hsi_expand = self.extend_spatial_resolution(hsi, extend_rate=mos_size / hsi_input_size)
        else:
            hsi_expand=hsi


        mos = torch.sum(hsi_expand * mask, dim=1).unsqueeze(1)
        mos_max = torch.max(mos.view(mos.shape[0], -1), 1)[0].unsqueeze(1).unsqueeze(1).unsqueeze(1)


        output_hsi = hsi_out # / mos_max * init_div_rat
        input_mos = mos / mos_max


        if isinstance(sigma, tuple):
            select_noise_sigma = sigma[random.randint(0, len(sigma) - 1)]
        else: 
            select_noise_sigma = sigma

        if gaussion_flag and not poisson_flag:
            input_mos = self.add_noise(input_mos, select_noise_sigma)
        elif poisson_flag and not gaussion_flag:
            input_mos = self.add_poisson_noise(input_mos)
        elif gaussion_flag and poisson_flag:
            if random.random() < 0.3:
                input_mos = self.add_poisson_noise(input_mos)
            else:
                input_mos = self.add_noise(input_mos, select_noise_sigma)
        input_mos = input_mos / input_mos.max()



        return input_mos, output_hsi
    

    def extend_spatial_resolution(self, hsi, extend_rate):
        hsi_extend = torch.nn.functional.interpolate(hsi, recompute_scale_factor=True, scale_factor=extend_rate)
        return hsi_extend



class Image_Cut(object):
    def __init__(self, image_size, patch_size, stride):
        self.patch_size = patch_size
        self.stride = stride #初始化归一化系数
        self.image_size = image_size

        self.patch_number = []
        self.hsi_max = []

    def image2patch(self, image):
        '''
        image_size = C, H, W
        '''
        patch_size = self.patch_size
        stride = self.stride

        c, h, w = image.shape
        image = image.unsqueeze(0)

        #构建图像块索引
        range_h = np.arange(0, h-patch_size[0], stride)
        range_w = np.arange(0, w-patch_size[1], stride)

        

        range_h = np.append(range_h, h-patch_size[0])
        range_w = np.append(range_w, w-patch_size[1])

        patch_num = len(range_h)*len(range_w)

        # print(range_h)
        # print(range_w)

        # if range_h[-1] != h - patch_size[0]:
        #     range_h = np.append(range_h, h-patch_size[0])
        # if range_w[-1] != w - patch_size[1]:
        #     range_w = np.append(range_w, h-patch_size[1])
        # patch_num = len(range_h)*len(range_w)

        patches = []
        for m in range_h:
            for n in range_w:
                patches.append(image[:, :, m : m + patch_size[0], n : n + patch_size[1]])

        return torch.cat(patches, 0)


        # patch_dir = []
        # data = np.expand_dims(data, axis=0)
        # # print(data.shape)
        # hight = data.shape[2]
        # width = data.shape[3]
        # bootom_width = 0
        # bootom_hight = 0
        # width_flag = False
        # hight_flag = False
        # out = []

        # while hight - bootom_hight > 0:
        #     if hight - bootom_hight <= size:
        #         bootom_hight = hight - size
        #         hight_flag = True
        #     while width - bootom_width > 0:
        #         if width - bootom_width <= size:
        #             bootom_width = width - size
        #             width_flag = True
        #         temp_data = data[:, :, bootom_hight:bootom_hight + size, bootom_width:bootom_width+ size]
        #         patch_dir.append([bootom_hight, bootom_width])
        #         out.append(temp_data)
        #         bootom_width += stride

        #         if width_flag == True:
        #             width_flag = False
        #             bootom_width = 10000000
        #     bootom_hight += stride
        #     bootom_width = 0
        #     if hight_flag == True:
        #         hight_flag = False
        #         bootom_hight = 10000000

        # return np.concatenate(out, 0), patch_dir

    def patch2image(self, patches):

        patch_size = self.patch_size
        stride = self.stride
        c = patches.shape[1]
        h, w = self.image_size

        res = torch.zeros((c, h, w)).to(patches.device)
        weight = torch.zeros((c, h, w)).to(patches.device)

        range_h = np.arange(0, h-patch_size[0], stride)
        range_w = np.arange(0, w-patch_size[1], stride)


        range_h = np.append(range_h, h-patch_size[0])
        range_w = np.append(range_w, w-patch_size[1])

        index = 0

        for m in range_h:
            for n in range_w:

                # print([m, n])

                res[:, m : m + patch_size[0], n : n + patch_size[1]] = res[:, m : m + patch_size[0], n : n + patch_size[1]] + patches[index, ...]

                weight[:, m : m + patch_size[0], n : n + patch_size[1]] = weight[:, m : m + patch_size[0], n : n + patch_size[1]] + 1
                index = index+1

        image = res / weight
        return image
