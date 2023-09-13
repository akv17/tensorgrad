try:
    import torchvision
except ImportError:
    msg = 'torchvision is required to train MNIST'
    raise Exception(msg)

import numpy as np

import tensorgrad


class Dataset:

    def __init__(self, patch_size, is_train=True, truncate=None, root='.datasets'):
        self.patch_size = patch_size
        self.is_train = is_train
        self.truncate = truncate
        self.dataset = torchvision.datasets.MNIST(root=root, download=True, train=is_train)
        self.size = self.truncate or len(self.dataset)

    @property
    def seq_len(self):
        ph, pw = self.patch_size
        lh = 28 // ph
        lw = 28 // pw
        sl = lh * lw
        return sl

    @property
    def num_features(self):
        ph, pw = self.patch_size
        n = ph * pw
        return n

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image).astype('float32')
        
        ih, iw = image.shape
        sh, sw = image.strides
        ph, pw = self.patch_size
        oh = ih // ph
        ow = iw // pw
        patches =  np.lib.stride_tricks.as_strided(
            image,
            [oh, ow, ph, pw],
            strides=[ph * sh, pw * sw, sh, sw]
        )
        patches = patches.reshape(oh * ow, ph * pw)
        patches = tensorgrad.tensor(patches)
        return patches, label
