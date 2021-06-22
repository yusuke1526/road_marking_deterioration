import os
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
import random
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score
from unet import UNet
import numpy as np
from dataset import MyDataset, DAVIDDataset
from torch.utils.data import DataLoader
import cv2
import argparse

def fix_seed(seed=418810):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def dataset():
    transform1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([args.img_height, args.img_width]),
    ])
    transform2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([args.img_height, args.img_width]),
    ])
    if args.dataset == 'semantic_segmentation_for_self_driving_cars':
        train_dataset = MyDataset(
            data_dir=args.train_data_dir,
            img_dir=args.img_dir,
            mask_dir=args.mask_dir,
            transform1=transform1,
            transform2=transform2,
        )
        test_dataset = MyDataset(
            data_dir=args.test_data_dir,
            img_dir=args.img_dir,
            mask_dir=args.mask_dir,
            transform1=transform1,
            transform2=transform2,
        )
    elif args.dataset == 'DAVID':
        train_dataset = DAVIDDataset(
            data_dir=args.train_data_dir,
            transform1=transform1,
            transform2=transform2,
        )
        test_dataset = DAVIDDataset(
            data_dir=args.test_data_dir,
            transform1=transform1,
            transform2=transform2,
        )
    return train_dataset, test_dataset

def init_model():
    model = UNet(n_classes=1)
    model.to(device)
    return model

def expand_score(y_true, y_pred, score_func):
    assert y_true.shape[0] == y_pred.shape[0], f"y_true size {y_true.shape[0]} and y_pred size {y_pred.shape[0]} doesn't match"
    if len(y_true.shape) == len(y_pred.shape) == 1:
        return score_func(y_true, y_pred, zero_division=0)
    else:
        y_true = y_true.reshape(y_true.shape[0], -1)
        y_pred = y_pred.reshape(y_pred.shape[0], -1)
        scores = np.array([score_func(t, p, zero_division=0) for t, p in zip(y_true, y_pred)])
        return scores.mean()

def metrics(pred, mask):
    pred = pred.detach().cpu().numpy()
    pred = np.where(pred > 0.5, 1, 0)
    mask = mask.detach().cpu().numpy().astype(int)
    return [expand_score(mask, pred, precision_score),
            expand_score(mask, pred, precision_score),
            expand_score(mask, pred, precision_score)]

def test(model, dataloader, criterion, path_pred):
    model.eval()
    loss_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    batch_iter = tqdm(dataloader, leave=False, desc='Batch')
    with torch.no_grad():
        for img, mask, path in batch_iter:
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            loss = criterion(pred, mask)
            batch_iter.set_postfix(loss=loss.item())
            loss_list.append(loss.item())

            # calculate metrics
            precision, recall, f1 = metrics(pred, mask)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            
            # save images
            img = img.detach().cpu().numpy()[0].transpose(1, 2, 0)
            mask = mask.detach().cpu().numpy()[0].transpose(1, 2, 0)
            pred = pred.detach().cpu().numpy()[0].transpose(1, 2, 0)
            plt.figure(figsize=(18, 6))
            plt.suptitle(f'loss:{loss.item()}')
            
            plt.subplot(1, 3, 1)
            plt.title('input')
            plt.imshow(img)
            plt.axis("off")
            
            plt.subplot(1, 3, 2)
            plt.title('ground truth')
            plt.imshow(mask)
            plt.axis("off")
            
            plt.subplot(1, 3, 3)
            plt.title('pred')
            plt.imshow(pred>0.5)
            plt.axis("off")
            
            plt.tight_layout()
            
            path = path[0].split('/')[-1].split('.')[0]
            file_name = f'loss_{loss.item()}_{path}.png'
            plt.savefig(os.path.join(path_pred, file_name))
            plt.close()

def main():
    train_dataset, test_dataset = dataset()
    train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=args.num_workers)
    
    # create dirs
    path_pred_train = os.path.join(args.log_dir, 'pred_train')
    path_pred_test = os.path.join(args.log_dir, 'pred_test')
    os.makedirs(path_pred_train, exist_ok=True)
    os.makedirs(path_pred_test, exist_ok=True)
    
    # load weight
    model = init_model()
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint)
    print(f'loaded model: {args.checkpoint}')
    model.to(device)
    
    criterion = nn.BCELoss()

    print('start prediction for test data')
    test(model, test_dataloader, criterion, path_pred_test)
    
    print('start prediction for train data')
    test(model, train_dataloader, criterion, path_pred_train)
            
if __name__ == '__main__':
    log = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument("--img_height", type=int, default=256)
    parser.add_argument("--img_width", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--dataset", type=str, default='DAVID', choices=["DAVID", "semantic_segmentation_for_self_driving_cars"])
    parser.add_argument("--train_data_dir", default='./data/dataA/dataA/')
    parser.add_argument("--test_data_dir", default='./data/dataB/dataB/')
    parser.add_argument("--img_dir", default='CameraRGB')
    parser.add_argument("--mask_dir", default='CameraSeg')
    
    # model
    parser.add_argument("--checkpoint", type=str, required=True)
    
    parser.add_argument('--log_dir', type=str, required=True)

    args = parser.parse_args()

    fix_seed()

    print(args)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    main()
    