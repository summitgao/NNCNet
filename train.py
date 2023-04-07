from setup import option
from utils import set_random_seed
from model import Net2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from dataset import TrainDS, TestDS
import os
import numpy as np
from sklearn.metrics import accuracy_score
import torch.optim as optim


def main():
    args = option()
    if args.seed is not None:
        set_random_seed(args.seed)
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    model = Net2()

    if os.path.isfile(args.pretrained):
        print("=> loading checkpoint '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('encoder_q'):
                # remove prefix
                state_dict[k[len("encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        print("=> loaded pre-trained model '{}'".format(args.pretrained))
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = model.to(device)

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    normalize = transforms.Normalize(
        (2382.7, 2368.9, 2688.8, 382590, -15.279, 29.642, -5.3076),
        (1260.8, 780.56, 1356.5, 134780, 0.22072, 125.31, 19.850))
    augmentation_lidar = transforms.Compose([
        normalize,
    ])

    mytrainset = TrainDS(augmentation_lidar)
    mytestset = TestDS(augmentation_lidar)

    train_loader = torch.utils.data.DataLoader(dataset=mytrainset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset=mytestset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=2)

    total_loss = 0
    Loss_list = []
    Accuracy_list = []
    max_acc = 0

    for epoch in range(args.epochs):
        net.train()
        epoch_loss = 0
        data_length = 0

        for i, (hsi, lidar, tr_labels) in enumerate(train_loader):
            hsi = hsi.to(device)
            lidar = lidar.to(device)
            tr_labels = tr_labels.to(device)

            output = net(hsi, lidar)

            loss = criterion(output, tr_labels)

            epoch_loss += loss.item() * hsi.size(0)
            data_length += hsi.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss /= data_length
        total_loss += epoch_loss
        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' %
              (epoch + 1, total_loss / (epoch + 1), epoch_loss))

        net.eval()

        count = 0
        epoch_loss = 0
        data_length = 0
        for hsi, lidar, gtlabels in test_loader:
            hsi = hsi.to(device)
            lidar = lidar.to(device)
            gtlabels = gtlabels.to(device)

            outputs = net(hsi, lidar)
            loss = criterion(outputs, gtlabels)
            epoch_loss += loss.item() * hsi.size(0)
            data_length += hsi.size(0)
            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            if count == 0:
                y_pred_test = outputs
                gty = gtlabels.cpu()
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, outputs))  #
                gty = np.concatenate((gty, gtlabels.cpu()))
        epoch_loss /= data_length
        cur_acc = accuracy_score(gty, y_pred_test)
        Loss_list.append(epoch_loss)
        Accuracy_list.append(cur_acc * 100)
        modelname = 'houston2018_' + str(cur_acc)
        torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict()
        }, './houston2018res/' + modelname + '.pth')
        if (max_acc < cur_acc):
            max_acc = cur_acc
            torch.save({
                'epoch': epoch,
                'state_dict': net.state_dict()
            }, './model_best.pth')
        print("cur_loss:", epoch_loss, " cur_acc:", cur_acc, " max_acc:",
              max_acc)


if __name__ == '__main__':
    main()
