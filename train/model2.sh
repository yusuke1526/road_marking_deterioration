python train.py --eval_interval 1 --img_height 384 --img_width 512 --batch_size 8 --epochs 20 --num_workers 2 \
--learning_rate 1e-4 --train_data_dir ~/dataset/DAVID/train/ \
--test_data_dir ~/dataset/DAVID/test/ --log_dir ./output/train/model2 --dataset DAVID