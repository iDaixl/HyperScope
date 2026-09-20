#!/bin/bash


python train.py --batch_size 4 \
    --end_epoch 200 \
    --epoch_sam_num 4000 \
    --method s2rnet \
    --load_check_point False \
    --gpu_id 0 \
    --init_lr 1e-4 \
    --mask_path ./mask_dir/6504pro_10x_flipud.mat \
    --output_folder ./output/S2RNet_10fluor/ \
    --validate_inference_dir ./inference_result/S2RNet_10fluor/ \
    --train_data_path ./dataset/hsi_dataset_10fluor/train_data/ \
    --valid_data_path ./dataset/hsi_dataset_10fluor/valid_data/ \
    --poisson_flag False \
    --gaussion_flag True \





