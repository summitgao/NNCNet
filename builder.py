import torch
import torch.nn as nn
import random


class NNCNet(nn.Module):
    def __init__(self, base_encoder, dim=128, K=2048, m=0.9, T=0.07):
        super(NNCNet, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    def forward(self, hsi_q, hsi_k, lidar_q, lidar_k):
        """
        Input:
            q: a batch of query images
            k: a batch of key images
        Output:
            logits, targets
        """

        lidar_q1 = lidar_q[:int(len(lidar_q) / 2)]
        lidar_q2 = lidar_q[int(len(lidar_q) / 2):]
        hsi_q1 = hsi_q[:int(len(hsi_q) / 2)]
        hsi_q2 = hsi_q[int(len(hsi_q) / 2):]
        q1 = self.encoder_q(hsi_q1, lidar_q1)
        q2 = self.encoder_q(hsi_q2, lidar_q2)
        q = nn.functional.normalize(torch.cat((q1, q2), 0), dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            lidar_k1 = lidar_k[0::2]
            lidar_k2 = lidar_k[1::2]
            hsi_k1 = hsi_k[0::2]
            hsi_k2 = hsi_k[1::2]
            k1 = self.encoder_k(hsi_k1, lidar_k1)
            k2 = self.encoder_k(hsi_k2, lidar_k2)
            k = torch.cat((k1, k2))
            for i in range(len(k1)):
                k[2 * i] = k1[i]
                k[2 * i + 1] = k2[i]
            k = nn.functional.normalize(k, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels, k


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""
    def __init__(self, base_transform, ratio=1):
        self.base_transform = base_transform
        self.ratio = ratio

    def __call__(self, x):
        q = self.base_transform(x)
        if (random.random() < self.ratio):
            k = self.base_transform(x)
        else:
            k = q
        return [q, k]
