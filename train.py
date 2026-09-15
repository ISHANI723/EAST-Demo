import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from dataset import custom_dataset
from model import EAST
from loss import Loss
import os
import time


def train(train_img_path, train_gt_path, pths_path,
          batch_size, lr, num_workers, epoch_iter, interval):

    os.makedirs(pths_path, exist_ok=True)

    trainset = custom_dataset(train_img_path, train_gt_path)

    train_loader = data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False
    )

    criterion = Loss()

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)
    print("Training images:", len(trainset))
    print("Batch size:", batch_size)
    print("Epochs:", epoch_iter)

    model = EAST()

    data_parallel = False

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        data_parallel = True

    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    scheduler = lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[epoch_iter // 2],
        gamma=0.1
    )

    for epoch in range(epoch_iter):

        model.train()

        epoch_loss = 0.0
        epoch_time = time.time()

        for i, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):

            start_time = time.time()

            img = img.to(device)
            gt_score = gt_score.to(device)
            gt_geo = gt_geo.to(device)
            ignored_map = ignored_map.to(device)

            pred_score, pred_geo = model(img)

            loss = criterion(
                gt_score,
                pred_score,
                gt_geo,
                pred_geo,
                ignored_map
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

            print(
                "Epoch [{}/{}], Batch [{}/{}], "
                "Time {:.2f}s, Loss {:.6f}".format(
                    epoch + 1,
                    epoch_iter,
                    i + 1,
                    len(train_loader),
                    time.time() - start_time,
                    loss.item()
                )
            )

        scheduler.step()

        average_loss = epoch_loss / len(train_loader)

        print("=" * 60)
        print(
            "Epoch [{}/{}] completed | Average Loss: {:.6f} | Time: {:.2f}s".format(
                epoch + 1,
                epoch_iter,
                average_loss,
                time.time() - epoch_time
            )
        )
        print(time.asctime(time.localtime(time.time())))
        print("=" * 60)

        if (epoch + 1) % interval == 0:

            state_dict = (
                model.module.state_dict()
                if data_parallel
                else model.state_dict()
            )

            save_path = os.path.join(
                pths_path,
                "model_epoch_{}.pth".format(epoch + 1)
            )

            torch.save(state_dict, save_path)

            print("Checkpoint saved:", save_path)


if __name__ == '__main__':
        train_img_path = r'D:\Ishani\DEAKIN UNI\T2-2026\SIT789_ROBOTICS\5.2HD_Datasets\ICDAR2015\ICDAR2015_CUSTOM\train_img'
        train_gt_path  = r'D:\Ishani\DEAKIN UNI\T2-2026\SIT789_ROBOTICS\5.2HD_Datasets\ICDAR2015\ICDAR2015_CUSTOM\train_gt'

        pths_path      = './pths'

        batch_size     = 2
        lr             = 1e-4
        num_workers    = 0
        epoch_iter     = 20
        save_interval  = 5

        os.makedirs(pths_path, exist_ok=True)

        train(
            train_img_path,
            train_gt_path,
            pths_path,
            batch_size,
            lr,
            num_workers,
            epoch_iter,
            save_interval
        )