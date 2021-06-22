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
            expand_score(mask, pred, recall_score),
            expand_score(mask, pred, f1_score)]

def main():
    train_dataset, test_dataset = dataset()
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    model = init_model()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # train loop
    for epoch in tqdm(range(args.epochs), desc='Epoch'):
        model.train()
        loss_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        batch_iter = tqdm(train_dataloader, leave=False, desc='Train Batch')
        for img, mask, _ in batch_iter:
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            loss = criterion(pred, mask)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_iter.set_postfix(loss=loss.item())
            loss_list.append(loss.item())

            # calculate metrics
            precision, recall, f1 = metrics(pred, mask)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

        # writer
        writer.add_scalar('loss/train', np.array(loss_list).mean(), epoch)
        writer.add_scalar('precision/train', np.array(precision_list).mean(), epoch)
        writer.add_scalar('recall/train', np.array(recall_list).mean(), epoch)
        writer.add_scalar('f1/train', np.array(f1_list).mean(), epoch)

        if (epoch % args.eval_interval == 0) or (epoch == args.epochs - 1):
            model.eval()
            loss_list = []
            precision_list = []
            recall_list = []
            f1_list = []
            batch_iter = tqdm(test_dataloader, leave=False, desc='test Batch')
            with torch.no_grad():
                for img, mask, _ in batch_iter:
                    img, mask = img.to(device), mask.to(device)
                    pred = model(img)
                    loss = criterion(pred, mask).item()
                    batch_iter.set_postfix(loss=loss)
                    loss_list.append(loss)
                    
                    # calculate metrics
                    precision, recall, f1 = metrics(pred, mask)
                    precision_list.append(precision)
                    recall_list.append(recall)
                    f1_list.append(f1)

            # writer
            writer.add_scalar('loss/test', np.array(loss_list).mean(), epoch)
            writer.add_scalar('precision/test', np.array(precision_list).mean(), epoch)
            writer.add_scalar('recall/test', np.array(recall_list).mean(), epoch)
            writer.add_scalar('f1/test', np.array(f1_list).mean(), epoch)

            torch.save(model.state_dict(), os.path.join(args.log_dir, f'model_epoch{epoch}.pth'))
            
if __name__ == '__main__':
    log = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    # Config
    parser.add_argument("--eval_interval", type=int, default=1)

    # Training Parameters
    parser.add_argument("--img_height", type=int, default=256)
    parser.add_argument("--img_width", type=int, default=256)
    parser.add_argument("--batch_size", '-B', type=int, default=32)
    parser.add_argument("--epochs", '-E', type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=0)

    # Optimizer
    parser.add_argument("--learning_rate", '-lr', type=float, default=1e-4)

    parser.add_argument("--dataset", type=str, default='DAVID', choices=["DAVID", "semantic_segmentation_for_self_driving_cars"])
    parser.add_argument("--train_data_dir", default='./data/dataA/dataA/')
    parser.add_argument("--test_data_dir", default='./data/dataB/dataB/')
    parser.add_argument("--img_dir", default='CameraRGB')
    parser.add_argument("--mask_dir", default='CameraSeg')
    

    parser.add_argument('--log_dir', type=str, required=True)

    args = parser.parse_args()

    fix_seed()

    writer = SummaryWriter(log_dir=args.log_dir)

    print(args)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    main()
    writer.close()
    