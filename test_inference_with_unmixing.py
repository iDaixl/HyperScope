# [2025-11-14] 此代码和test_inference_mosaic_save_img_only.py区别在于把load_model写到了外环
# [2025-11-14] 不需要每次都重新load_model


import numpy as np
import h5py
import cv2
import hdf5storage
from inference import HSIInference
import os
import time  # 
import shutil

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def load_model(model_path, mask_path, model_method='srnet'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    model_load_start = time.time()
    inferencer = HSIInference(model_path, mask_path, method=model_method)
    model_load_time = time.time() - model_load_start
    print(f"模型加载时间: {model_load_time:.2f}秒")
    return inferencer

def inference_image_folder(inferencer_input, image_folder, save_dir, endmember_path, fluor_list):
    inferencer = inferencer_input
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Input folder not found: {image_folder}")
    
    os.makedirs(save_dir, exist_ok=True)

    try:
        test_file = os.path.join(save_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
    except Exception as e:
        raise PermissionError(f"Cannot write to output directory: {save_dir}. Error: {str(e)}")
    
    try:
        inference_start = time.time()
        
        # 处理文件夹中的所有图像
        inferencer.process_folder_with_unmixing_fluor_list(
            image_folder=image_folder,
            save_dir=save_dir,
            start_dir=(0, 0),
            image_size=(2048, 2048),
            save_hsi=False,
            save_gray_image=True,
            save_unmix=True,
            save_tif=True,
            endmember_path=endmember_path,
            fluor_list=fluor_list
        )
        
        # 计算推理时间
        inference_time = time.time() - inference_start
        print(f"\n推理处理时间: {inference_time:.2f}秒")
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()
    
    
    # 统计处理的图像数量
    image_count = len([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', 'tif','.png', '.bmp'))])
    if image_count > 0 and 'inference_time' in locals():
        print(f"平均每张图像处理时间: {inference_time/image_count:.2f}秒")


if __name__ == '__main__':
    model_path = "./model_zoo/net.pth"
    save_name = "result_wt_unmixing/"

    mask_path = "./mask_dir/6504pro_10x_flipud.mat"

    endmember_path = "./spectral_unmixing/Cell_unmixing.csv"
    fluor_list = ['Nuclei', 'Lyso', 'Mito', 'Tublin']

    image_folder_list = ['./input_img/']

    inferencer = load_model(model_path, mask_path, model_method='s2rnet')
    

    for image_folder in image_folder_list:
        image_folder = image_folder + '/'
        print('processing image folder:', image_folder)

        save_dir = image_folder + save_name  # 结果保存根目录
        
        try:
            inference_image_folder(inferencer, image_folder, save_dir, endmember_path, fluor_list)
        except Exception as e:
            print(f"Fatal error: {str(e)}")
            import traceback
            traceback.print_exc()