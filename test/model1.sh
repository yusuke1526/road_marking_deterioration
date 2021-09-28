python test.py --img_height 384 --img_width 512 --train_data_dir ~/dataset/semantic_segmentation_for_self_driving_cars/dataA/dataA/ \
--test_data_dir ~/dataset/semantic_segmentation_for_self_driving_cars/dataB/dataB/ \
--img_dir CameraRGB --mask_dir CameraSeg --log_dir ./output/test/model1 --checkpoint ./output/train/model1/model_epoch19.pth \
--dataset semantic_segmentation_for_self_driving_cars