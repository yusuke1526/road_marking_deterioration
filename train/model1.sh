python train.py --eval_interval 1 --img_height 256 --img_width 256 --batch_size 16 --epochs 20 --num_workers 2 \
--learning_rate 1e-4 --train_data_dir ~/dataset/semantic_segmentation_for_self_driving_cars/dataA/dataA/ \
--test_data_dir ~/dataset/semantic_segmentation_for_self_driving_cars/dataB/dataB/ \
--img_dir CameraRGB --mask_dir CameraSeg --log_dir ./output/train/model1