python test.py --img_height 192 --img_width 256 --train_data_dir ~/dataset/semantic_segmentation_for_self_driving_cars/dataA/dataA/ \
--test_data_dir ~/dataset/semantic_segmentation_for_self_driving_cars/dataB/dataB/ \
--img_dir CameraRGB --mask_dir CameraSeg --log_dir ./output/test/model2 --checkpoint ./output/train/model2/model_epoch19.pth