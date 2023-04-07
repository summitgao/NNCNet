from setup import option
from utils import set_random_seed, AverageMeter, ProgressMeter, accuracy, adjust_learning_rate
from builder import NNCNet, TwoCropsTransform
from model import Net1
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from add_noise import AddNoiseBlind
from dataset import AllDS
import time

windowSize = 11


def main():
    args = option()
    if args.seed is not None:
        set_random_seed(args.seed)
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    model = NNCNet(Net1, args.dim, args.k, args.m, args.t)

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    cudnn.benchmark = True

    normalize = transforms.Normalize(
        (2382.7, 2368.9, 2688.8, 382590, -15.279, 29.642, -5.3076),
        (1260.8, 780.56, 1356.5, 134780, 0.22072, 125.31, 19.850))

    transform_crop = [
        transforms.RandomResizedCrop(windowSize, scale=(0.5, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ]
    augmentation_lidar = [normalize]
    augmentation_hsi = []
    transform_noise = [
        AddNoiseBlind([10, 20, 30]),
    ]

    train_sampler = None
    trainsetall = AllDS(
        TwoCropsTransform(transforms.Compose(transform_crop), 0.5),
        transforms.Compose(augmentation_lidar),
        transforms.Compose(augmentation_hsi),
        transforms.Compose(transform_noise))

    train_loader = torch.utils.data.DataLoader(trainsetall,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last=True)

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        loss, acc1 = train(train_loader, model, criterion, optimizer, epoch,
                           args)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':6.6f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))

    model.train()
    running_loss = 0.0
    running_corrects = 0
    end = time.time()
    for i, (hsi, lidar, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            lidar[0] = lidar[0].type(torch.FloatTensor).cuda(args.gpu,
                                                             non_blocking=True)
            lidar[1] = lidar[1].type(torch.FloatTensor).cuda(args.gpu,
                                                             non_blocking=True)
            hsi[0] = hsi[0].type(torch.FloatTensor).cuda(args.gpu,
                                                         non_blocking=True)
            hsi[1] = hsi[1].type(torch.FloatTensor).cuda(args.gpu,
                                                         non_blocking=True)

        output, target, k = model(hsi[0].unsqueeze(1), hsi[1].unsqueeze(1),
                                  lidar[0], lidar[1])
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), lidar[0].size(0))
        top1.update(acc1[0], lidar[0].size(0))
        top5.update(acc5[0], lidar[0].size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        running_loss += loss.data.item()
        running_corrects += acc1.item()
    epoch_loss = running_loss / (len(train_loader) * args.batch_size)
    epoch_acc1 = running_corrects / len(train_loader)

    torch.save(
        {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, "./checkpoint.pth.tar")

    return epoch_loss, epoch_acc1


if __name__ == '__main__':
    main()
