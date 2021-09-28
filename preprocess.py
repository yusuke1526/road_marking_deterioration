import os
import cv2
import glob
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def scale_to_width(img, width):
    """幅が指定した値になるように、アスペクト比を固定して、リサイズする。
    """
    h, w = img.shape[:2]
    height = round(h * (width / w))
    dst = cv2.resize(img, dsize=(width, height))

    return dst

def trim(img):
    h, w = img.shape[:2]
    return img[h//2:, :, :]

def main():
    paths = sorted(glob.glob(os.path.join(args.source_dir, '*.jpg')))
    os.makedirs(args.target_dir)

    if 0 < args.img_width:
        for filename in tqdm(paths):
            img = cv2.imread(filename)
            img = scale_to_width(img, args.img_width)
            cv2.imwrite(os.path.join(args.target_dir, filename.split('/')[-1]), trim(img))
    else:
        for filename in tqdm(paths):
            img = cv2.imread(filename)
            cv2.imwrite(os.path.join(args.target_dir, filename.split('/')[-1]), trim(img))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Config
    parser.add_argument("--source_dir", "-s", type=str, required=True)
    parser.add_argument("--target_dir", "-t", type=str, required=True)
    parser.add_argument("--img_width", "-w", type=int, default=-1)
    args = parser.parse_args()
    print(args)
    main()