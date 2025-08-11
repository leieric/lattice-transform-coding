import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision import transforms

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')
import tensorflow_probability as tfp

tfm = tf.math
tfpb = tfp.bijectors
tfpd = tfp.distributions


def get_loader(args, n_samples, drop_last=False, train=False):
    if args.data_name == 'Gaussian':
        return Gaussian(args.d, args.batch_size, n_samples)
    elif args.data_name == 'Uniform':
        return Uniform(args.d, args.batch_size, n_samples)
    elif args.data_name == 'Laplace':
        return Laplace(args.d, args.batch_size, n_samples)
    elif args.data_name == 'Banana1d':
        return Banana1d(args.d, args.batch_size, n_samples)
    elif args.data_name == 'Sawbridge':
        return Sawbridge(args.batch_size, n_samples, 1024)
    elif args.data_name == 'SawbridgeBlock':
        return SawbridgeBlock(args.n, args.batch_size, n_samples, 1024, drop_last)
    elif args.data_name == 'Banana':
        return Banana(args.batch_size, n_samples)
    elif args.data_name == 'BananaBlock':
        return BananaBlock(args.n, args.batch_size, n_samples, drop_last)
    elif args.data_name == 'Physics':
        return Physics(args.batch_size, train, args.data_root)
    elif args.data_name == 'PhysicsBlock':
        return PhysicsBlock(args.n, args.batch_size, train, args.data_root)
    elif args.data_name == 'Speech':
        return Speech(args.batch_size, train, args.data_root)
    elif args.data_name == 'SpeechBlock':
        return SpeechBlock(args.n, args.batch_size, train, args.data_root)
    else:
        raise Exception("invalid data_name")

def Gaussian(n, batch_size, n_samples=1000000):
    dset = torch.randn(n_samples, n) 
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(dset), batch_size=batch_size, num_workers=4)
    return loader

def Uniform(n, batch_size, n_samples=1000000):
    dset = torch.rand(n_samples, n) - 0.5
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(dset), batch_size=batch_size, num_workers=4)
    return loader

def Laplace(n, batch_size, n_samples=1000000):
    # dset = torch.rand(n_samples, n) - 0.5
    dist = torch.distributions.laplace.Laplace(0.0, 1.0)
    dset = dist.sample((n_samples, n))
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(dset), batch_size=batch_size, num_workers=4)
    return loader

def Sawbridge(batch_size, n_samples=1000000, sampling_rate=1024):
    t = torch.linspace(0, 1, sampling_rate)
    torch.manual_seed(123)
    U = torch.rand((n_samples, 1))
    X = t - (t >= U).to(torch.get_default_dtype())
    dset = torch.utils.data.TensorDataset(X) # [n_samples, sampling_rate]
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(dset), batch_size=batch_size, num_workers=4)
    return loader

def SawbridgeBlock(n, batch_size, n_samples=1000000, sampling_rate=1024, drop_last=False):
    dset = SawbridgeBlockDataset(n, n_samples, sampling_rate)
    loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, num_workers=4, drop_last=drop_last)
    return loader


class SawbridgeBlockDataset(Dataset):
    def __init__(self, n, n_samples=1000000, sampling_rate=1024):
        self.n = n
        self.n_samples = n_samples
        self.sampling_rate = sampling_rate

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        t = torch.linspace(0, 1, self.sampling_rate)
        U = torch.rand((self.n, 1))
        Xi = t - (t >= U).to(torch.get_default_dtype())
        return Xi, 0
    
def _rotation_2d(degrees):
    phi = tf.convert_to_tensor(degrees / 180 * np.pi, dtype=tf.float32)
    rotation = [[tfm.cos(phi), -tfm.sin(phi)], [tfm.sin(phi), tfm.cos(phi)]]
    rotation = tf.linalg.LinearOperatorFullMatrix(
        rotation, is_non_singular=True, is_square=True)
    return rotation

def get_banana():
    return tfpd.TransformedDistribution(
        tfpd.Independent(tfpd.Normal(loc=[0, 0], scale=[3, .5]), 1),
        tfpb.Invert(tfpb.Chain([
            tfpb.RealNVP(
                num_masked=1,
                shift_and_log_scale_fn=lambda x, _: (.1 * x ** 2, None)),
            tfpb.ScaleMatvecLinearOperator(_rotation_2d(240)),
            tfpb.Shift([1, 1]),
        ])),
    )

def Banana(batch_size, n_samples=1000000):
    source = get_banana()
    # with tf.device('/cpu:0'):
    X = source.sample(n_samples).numpy()
    X[:, [1, 0]] = X[:, [0, 1]]
    X = torch.tensor(X)
    X = (X - X.mean(dim=0)[None, :]) / X.std()
    dset = torch.utils.data.TensorDataset(X) # [n_samples, 2]
    loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, num_workers=4)
    return loader

def Banana1d(n, batch_size, n_samples=1000000):
    source = get_banana()
    # with tf.device('/cpu:0'):
    X = source.sample(n_samples * n).numpy()
    X[:, [1, 0]] = X[:, [0, 1]]
    X = torch.tensor(X)
    X = (X - torch.mean(X)) / torch.std(X) # make 0-mean, 1-var
    X = X[:, 0] # take first marginal
    X = X.reshape(n_samples, n)
    dset = torch.utils.data.TensorDataset(X) # [n_samples, 2]
    loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, num_workers=4)
    return loader



# def BananaBlock(n, batch_size, n_samples=1000000, drop_last=False):
#     dset = BananaBlockDataset(n, n_samples)
#     loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, num_workers=8, pin_memory=True, drop_last=drop_last)
#     return loader

def BananaBlock(n, batch_size, n_samples=1000000, drop_last=False):
    source = get_banana()
    X = source.sample(n * n_samples).numpy()
    X[:, [1, 0]] = X[:, [0, 1]]
    X = torch.tensor(X).to(torch.get_default_dtype())
    X = (X - X.mean(dim=0)[None, :]) / X.std()
    X = X.reshape(n_samples, n, 2)
    dset = torch.utils.data.TensorDataset(X) # [n_samples, 2]
    loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, num_workers=4)
    return loader

class BananaBlockDataset(Dataset):
    def __init__(self, n, n_samples=1000000):
        self.n = n
        self.n_samples = n_samples
        self.source = get_banana()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        Xi = self.source.sample(self.n).numpy()
        Xi[:, [1, 0]] = Xi[:, [0, 1]]
        Xi = torch.tensor(Xi)
        return Xi, 0
    

def Physics(batch_size, train=True, data_root='.'):
    X_train = np.load(f'{data_root}/physics/ppzee-split=train.npy')
    # mean = np.mean(X_train, axis=0)
    # std = np.std(X_train, axis=0)
    if train:
        X = X_train
    else:
        X = np.load(f'{data_root}/physics/ppzee-split=test.npy')
    X = torch.tensor(X).float()
    print(X.std())
    X = (X - X.mean(dim=0)[None, :]) / X.std() 
    dset = torch.utils.data.TensorDataset(X) # [n_samples, 2]
    loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, num_workers=4)
    return loader

def PhysicsBlock(n, batch_size, train=True, data_root='.'):
    X_train = np.load(f'{data_root}/physics/ppzee-split=train.npy')
    # mean = np.mean(X_train, axis=0)
    # std = np.std(X_train, axis=0)
    if train:
        X = X_train
    else:
        X = np.load(f'{data_root}/physics/ppzee-split=test.npy')
    X = torch.tensor(X).float()
    X = (X - X.mean(dim=0)[None, :]) / X.std()
    N, dim = X.shape
    n_samples = N // n
    X = X[0:n*n_samples].reshape(n_samples, n, dim)
    dset = torch.utils.data.TensorDataset(X) # [n_samples, 2]
    loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, num_workers=4)
    return loader
    

def Speech(batch_size, train=True, data_root='.'):
    X_train = np.load(f'{data_root}/speech/stft-split=train.npy')
    # mean = np.mean(X_train, axis=0)
    if train:
        X = X_train
    else:
        X = np.load(f'{data_root}/speech/stft-split=test.npy')
    # X = X - mean[None, :]
    X = torch.tensor(X).float()
    print(X.std())
    X = (X - X.mean(dim=0)[None, :])# / X.std()
    dset = torch.utils.data.TensorDataset(X) # [n_samples, 2]
    loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, num_workers=4)
    return loader

def SpeechBlock(n, batch_size, train=True, data_root='.'):
    X_train = np.load(f'{data_root}/speech/stft-split=train.npy')
    # mean = np.mean(X_train, axis=0)
    if train:
        X = X_train
    else:
        X = np.load(f'{data_root}/speech/stft-split=test.npy')
    # X = X - mean[None, :]
    X = torch.tensor(X).float()
    # print(X.std())
    X = (X - X.mean(dim=0)[None, :])# / X.std()
    N, dim = X.shape
    n_samples = N // n
    X = X[0:n*n_samples].reshape(n_samples, n, dim)
    dset = torch.utils.data.TensorDataset(X) # [n_samples, 2]
    loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, num_workers=4)
    return loader
