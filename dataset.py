from skimage import io
import numpy as np
from utils import applyPCA, getAllPos, getTrainPos, getTestPos, createImgCubeAll, createImgCube, createImgPatch
import torch

windowSize = 11


class AllDS(torch.utils.data.Dataset):
    def __init__(self, transform_crop, transform_lidar, transform_hsi,
                 transform_noise):
        hsiall_im = io.imread("./houston2018/HSI/HSI.tif")[:, :, :48]
        lidarall_im = np.load("./houston2018/Lidar/lidarall.npy")
        hsiall_pca = applyPCA(hsiall_im, 30)

        self.pos = getAllPos(lidarall_im)
        self.len = self.pos.shape[0]
        self.hsi_cube, self.gt = createImgCubeAll(hsiall_pca,
                                                  self.pos,
                                                  windowSize=windowSize)
        self.hsi_cube = torch.from_numpy(self.hsi_cube.transpose(
            (0, 3, 1, 2))).float()
        self.gt = torch.from_numpy(self.gt - 1)
        self.lidar_patch = createImgPatch(lidarall_im,
                                          self.pos,
                                          windowSize=windowSize)
        self.lidar_patch = torch.from_numpy(
            self.lidar_patch.transpose((0, 3, 1, 2))).float()

        self.transform_lidar = transform_lidar
        self.transform_hsi = transform_hsi
        self.transform_crop = transform_crop
        self.transform_noise = transform_noise

    def __getitem__(self, index):
        lidarhsi = torch.cat([self.lidar_patch[index], self.hsi_cube[index]],
                             0)
        lidarhsi = self.transform_crop(lidarhsi)
        lidar_ = [
            self.transform_noise(self.transform_lidar(lidarhsi[0][:7])),
            self.transform_noise(self.transform_lidar(lidarhsi[1][:7]))
        ]
        hsi_ = [
            self.transform_noise(self.transform_hsi(lidarhsi[0][7:])),
            self.transform_noise(self.transform_hsi(lidarhsi[1][7:]))
        ]
        return hsi_, lidar_, self.gt[index]

    def __len__(self):
        return self.len


class TrainDS(torch.utils.data.Dataset):
    def __init__(self, augmentation_lidar):
        hsi_im = np.load("./houston2018/HSI/hsi.npy")[:, :, :48]
        lidar_im = np.load("./houston2018/Lidar/lidar.npy")
        train_im = np.load("./houston2018/GT/mask_train.npy")
        hsi_pca = applyPCA(hsi_im, 30)
        self.pos = getTrainPos(train_im)
        self.len = self.pos.shape[0]
        self.hsi_cube, self.gt = createImgCube(hsi_pca,
                                               self.pos,
                                               windowSize=windowSize)
        self.hsi_cube = torch.from_numpy(self.hsi_cube.transpose(
            (0, 3, 1, 2))).unsqueeze(1).float()
        self.gt = torch.from_numpy(self.gt - 1)
        self.lidar_patch = createImgPatch(lidar_im,
                                          self.pos,
                                          windowSize=windowSize)
        self.lidar_patch = torch.from_numpy(
            self.lidar_patch.transpose((0, 3, 1, 2))).float()

        self.aug = augmentation_lidar

    def __getitem__(self, index):
        lidar_ = self.aug(self.lidar_patch[index]).float()
        return self.hsi_cube[index], lidar_, self.gt[index]

    def __len__(self):
        return self.len


class TestDS(torch.utils.data.Dataset):
    def __init__(self, augmentation_lidar):
        hsi_im = np.load("./houston2018/HSI/hsi.npy")[:, :, :48]
        lidar_im = np.load("./houston2018/Lidar/lidar.npy")
        test_im = np.load("./houston2018/GT/mask_test.npy")
        hsi_pca = applyPCA(hsi_im, 30)
        self.pos = getTestPos(test_im)
        self.len = self.pos.shape[0]
        self.hsi_cube, self.gt = createImgCube(hsi_pca,
                                               self.pos,
                                               windowSize=windowSize)
        self.hsi_cube = torch.from_numpy(self.hsi_cube.transpose(
            (0, 3, 1, 2))).unsqueeze(1).float()
        self.gt = torch.from_numpy(self.gt - 1)
        self.lidar_patch = createImgPatch(lidar_im,
                                          self.pos,
                                          windowSize=windowSize)
        self.lidar_patch = torch.from_numpy(
            self.lidar_patch.transpose((0, 3, 1, 2))).float()

        self.aug = augmentation_lidar

    def __getitem__(self, index):
        lidar_ = self.aug(self.lidar_patch[index]).float()
        return self.hsi_cube[index], lidar_, self.gt[index]

    def __len__(self):
        return self.len
