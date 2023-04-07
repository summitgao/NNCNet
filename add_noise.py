import numpy as np
import threading
import torch


class SequentialSelect(object):
    def __pos(self, n):
        i = 0
        while True:
            yield i
            i = (i + 1) % n

    def __init__(self, transforms):
        self.transforms = transforms
        self.pos = LockedIterator(self.__pos(len(transforms)))

    def __call__(self, img):
        out = self.transforms[next(self.pos)](img)
        return out


class LockedIterator(object):
    def __init__(self, it):
        self.lock = threading.Lock()
        self.it = it.__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        self.lock.acquire()
        try:
            return next(self.it)
        finally:
            self.lock.release()


class AddNoise(object):
    """add gaussian noise to the given numpy array (B,H,W)"""
    def __init__(self, sigma):
        self.sigma_ratio = sigma / 255.

    def __call__(self, img):
        noise = np.random.randn(*img.shape) * self.sigma_ratio
        return img + noise


class AddNoiseBlind(object):
    """add blind gaussian noise to the given numpy array (B,H,W)"""
    def __pos(self, n):
        i = 0
        while True:
            yield i
            i = (i + 1) % n

    def __init__(self, sigmas):
        self.sigmas = np.array(sigmas) / 255.
        self.pos = LockedIterator(self.__pos(len(sigmas)))

    def __call__(self, img):
        noise = np.random.randn(*img.shape) * self.sigmas[next(self.pos)]
        return img + noise


class AddNoiseBlindv2(object):
    """add blind gaussian noise to the given numpy array (B,H,W)"""
    def __init__(self, min_sigma, max_sigma):
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def __call__(self, img):
        noise = np.random.randn(*img.shape) * np.random.uniform(
            self.min_sigma, self.max_sigma) / 255
        return img + noise


class AddNoiseNoniid(object):
    """add non-iid gaussian noise to the given numpy array (B,H,W)"""
    def __init__(self, sigmas):
        self.sigmas = np.array(sigmas) / 255.

    def __call__(self, img):
        bwsigmas = np.reshape(
            self.sigmas[np.random.randint(0, len(self.sigmas), img.shape[0])],
            (-1, 1, 1))
        noise = np.random.randn(*img.shape) * bwsigmas
        return img + noise


class AddNoiseMixed(object):
    """add mixed noise to the given numpy array (B,H,W)
    Args:
        noise_bank: list of noise maker (e.g. AddNoiseImpulse)
        num_bands: list of number of band which is corrupted by each item in noise_bank"""
    def __init__(self, noise_bank, num_bands):
        assert len(noise_bank) == len(num_bands)
        self.noise_bank = noise_bank
        self.num_bands = num_bands

    def __call__(self, img):
        B, H, W = img.shape
        all_bands = np.random.permutation(range(B))
        pos = 0
        for noise_maker, num_band in zip(self.noise_bank, self.num_bands):
            if 0 < num_band <= 1:
                num_band = int(np.floor(num_band * B))
            bands = all_bands[pos:pos + num_band]
            pos += num_band
            img = noise_maker(img, bands)
        return img


class _AddNoiseImpulse(object):
    """add impulse noise to the given numpy array (B,H,W)"""
    def __init__(self, amounts, s_vs_p=0.5):
        self.amounts = np.array(amounts)
        self.s_vs_p = s_vs_p

    def __call__(self, img, bands):
        # bands = np.random.permutation(range(img.shape[0]))[:self.num_band]
        bwamounts = self.amounts[np.random.randint(0, len(self.amounts),
                                                   len(bands))]
        for i, amount in zip(bands, bwamounts):
            self.add_noise(img[i, ...],
                           amount=amount,
                           salt_vs_pepper=self.s_vs_p)
        return img

    def add_noise(self, image, amount, salt_vs_pepper):
        out = image
        p = amount
        q = salt_vs_pepper
        flipped = np.random.choice([True, False],
                                   size=image.shape,
                                   p=[p, 1 - p])
        salted = np.random.choice([True, False],
                                  size=image.shape,
                                  p=[q, 1 - q])
        peppered = ~salted

        out[torch.tensor(flipped & salted)] = 1
        out[torch.tensor(flipped & peppered)] = 0

        return out


class _AddNoiseStripe(object):
    """add stripe noise to the given numpy array (B,H,W)"""
    def __init__(self, min_amount, max_amount):
        assert max_amount > min_amount
        self.min_amount = min_amount
        self.max_amount = max_amount

    def __call__(self, img, bands):
        B, H, W = img.shape
        # bands = np.random.permutation(range(img.shape[0]))[:len(bands)]
        num_stripe = np.random.randint(np.floor(self.min_amount * W),
                                       np.floor(self.max_amount * W),
                                       len(bands))
        for i, n in zip(bands, num_stripe):
            loc = np.random.permutation(range(W))
            loc = loc[:n]

            stripe = np.random.uniform(0, 1, size=(len(loc), )) * 0.5 - 0.25
            img[i, :, loc] -= np.reshape(stripe, (-1, 1))
        return img


class _AddNoiseDeadline(object):
    """add deadline noise to the given numpy array (B,H,W)"""
    def __init__(self, min_amount, max_amount):
        assert max_amount > min_amount
        self.min_amount = min_amount
        self.max_amount = max_amount

    def __call__(self, img, bands):
        B, H, W = img.shape
        # bands = np.random.permutation(range(img.shape[0]))[:len(bands)]
        num_deadline = np.random.randint(np.ceil(self.min_amount * W),
                                         np.ceil(self.max_amount * W),
                                         len(bands))
        for i, n in zip(bands, num_deadline):
            loc = np.random.permutation(range(W))
            loc = loc[:n]
            img[i, :, loc] = 0
        return img


class AddNoiseImpulse(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [_AddNoiseImpulse([0.1, 0.3, 0.5, 0.7])]
        self.num_bands = [1 / 3]


class AddNoiseStripe(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [_AddNoiseStripe(0.05, 0.25)]
        self.num_bands = [1 / 3]


class AddNoiseDeadline(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [_AddNoiseDeadline(0.05, 0.25)]
        self.num_bands = [1 / 3]


class AddNoiseComplex(AddNoiseMixed):
    def __init__(self):
        self.noise_bank = [
            _AddNoiseStripe(0.05, 0.25),
            _AddNoiseDeadline(0.05, 0.25),
            _AddNoiseImpulse([0.1, 0.3, 0.5, 0.7])
        ]
        self.num_bands = [1 / 3, 1 / 3, 1 / 3]
