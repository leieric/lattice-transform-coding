# Training script is taken from CompressAI repository and slightly modified for handling grayscale images + MS-SSIM loss.

# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import random
import shutil
import sys
import os

import torch
torch.set_num_threads(4)
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder, Vimeo90kDataset
# from compressai.losses import RateDistortionLoss
# from compressai.optimizers import net_aux_optimizer
from LTC.net_aux import net_aux_optimizer
from compressai.zoo import image_models
from PIL import Image

from pytorch_msssim import ms_ssim
import math
import tqdm
import wandb

import LTC.models_compressai as models_compressai

from compressai.zoo import image_models as pretrained_models

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=0.01, metric="mse", return_type="all"):
        super().__init__()
        if metric == "mse":
            self.metric = nn.MSELoss()
        elif metric == "ms-ssim":
            self.metric = ms_ssim
        else:
            raise NotImplementedError(f"{metric} is not implemented!")
        self.lmbda = lmbda
        self.return_type = return_type

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        if self.metric == ms_ssim:
            out["ms_ssim_loss"] = self.metric(output["x_hat"], target, data_range=1)
            distortion = 1 - out["ms_ssim_loss"]
        else:
            out["mse_loss"] = self.metric(output["x_hat"], target)
            distortion = 255**2 * out["mse_loss"]

        out["loss"] = self.lmbda * distortion + out["bpp_loss"]
        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]

class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        "net": {"type": "Adam", "lr": args.learning_rate},
        "aux": {"type": "Adam", "lr": args.aux_learning_rate},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]

def psnr(mse):
    return 10*torch.log10(1 / mse)

def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, args
):
    model.train()
    device = next(model.parameters()).device

    pbar = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader), dynamic_ncols=True)
    for i, d in pbar:
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % (len(train_dataloader) // 4) == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\t{args.dist_metric} loss: {out_criterion[f"{args.dist_metric}_loss"].item():.3f} |'
                f'\tPSNR: {psnr(out_criterion[f"{args.dist_metric}_loss"]).item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.4f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )
            wandb.log({"loss":out_criterion["loss"].item(),
               "dist": out_criterion[f"{args.dist_metric}_loss"].item(),
               "psnr": psnr(out_criterion[f"{args.dist_metric}_loss"]).item(),
               "bpp": out_criterion["bpp_loss"].item(),
               "aux_loss": aux_loss.item()})
        pbar.set_description(f'loss={out_criterion["loss"].item():.4f}, bpp={out_criterion["bpp_loss"].item():.4f}, psnr={psnr(out_criterion[f"{args.dist_metric}_loss"]).item():.3f}')


def test_epoch(epoch, test_dataloader, model, criterion, args):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    psnr_loss = AverageMeter()

    with torch.no_grad():
        for d in tqdm.tqdm(test_dataloader, total=len(test_dataloader), dynamic_ncols=True):
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion[f"{args.dist_metric}_loss"])
            psnr_loss.update(psnr(out_criterion[f"{args.dist_metric}_loss"]))

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.4f} |"
        f"\tDistort loss: {mse_loss.avg:.4f} |"
        f'\tPSNR: {psnr_loss.avg:.4f} |'
        f"\tBpp loss: {bpp_loss.avg:.4f} |"
        f"\tAux loss: {aux_loss.avg:.4f}\n"
    )
    wandb.log({"test_loss":loss.avg,
               "test_dist": mse_loss.avg,
               "test_psnr": psnr_loss.avg,
               "test_bpp": bpp_loss.avg,
               "test_aux_loss": aux_loss.avg})

    return loss.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f"{filename[:-3]}_best.pt")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    # parser.add_argument(
    #     "-m",
    #     "--model",
    #     default="bmshj2018-factorized",
    #     choices=image_models.keys(),
    #     help="Model architecture (default: %(default)s)",
    # )
    parser.add_argument(
        "--lattice_name", type=str, default="E8Product", required=True, help="Lattice name"
    )
    parser.add_argument(
        "--channels", type=int, default=128, required=True, help="latent channel dimension"
    )
    parser.add_argument(
        "--N_integral", type=int, default=2048, required=True, help="Monte-Carlo number of samples"
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument("--overwrite_lr", action="store_true", help="Overwrite saved LR with --learning-rate when loading a checkpoint")
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--dist_metric",
        dest="dist_metric",
        type=str,
        default="mse",
        help="Either mse or ms-ssim (default: %(default)s)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=4,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        type=float,
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--use_data_parallel", action="store_true", help="Use data parallel (requires multiple GPUs)")
    parser.add_argument("--load_ntc", action="store_true", help="initialize with NTC baseline")
    parser.add_argument(
        "--ntc_quality",
        type=int,
        default=4,
        help="NTC baseline quality",
    )
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument("--seed", type=int, help="Set random seed for reproducibility")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    print(f"patch_size={args.patch_size}")
    wandb.init(project="compressai")
    wandb.config.update(args)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = transforms.Compose(
        [
            transforms.RandomCrop(args.patch_size), 
            transforms.ToTensor()
            ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.CenterCrop(args.patch_size), 
         transforms.ToTensor()]
    )
    
    print(f"Dataset: {os.path.basename(args.dataset)}")
    if os.path.basename(args.dataset) == "vimeo_septuplet":
        train_dataset = Vimeo90kDataset(args.dataset, split="train", transform=train_transforms, tuplet=7)
        test_dataset = Vimeo90kDataset(args.dataset, split="valid", transform=test_transforms, tuplet=7)
    else:
        train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
        test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = models_compressai.Cheng2020AttentionLattice(N=args.channels, N_integral=args.N_integral, lattice_name=args.lattice_name)
    if args.load_ntc:
        print("Loading NTC models...")
        model_name = "cheng2020-attn"
        model_NTC = pretrained_models[model_name](
            quality=args.ntc_quality, metric=args.dist_metric, pretrained=True, progress=False
        )
        state_dict = {k: v for k, v in model_NTC.state_dict().items() if k not in ["gaussian_conditional._offset", "gaussian_conditional._quantized_cdf", "gaussian_conditional._cdf_length", "gaussian_conditional.scale_table"]}
        net.load_state_dict(state_dict, strict=False)
    net = net.to(device)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateDistortionLoss(lmbda=args.lmbda, metric=args.dist_metric)
    args.dist_metric = "ms_ssim" if args.dist_metric == "ms-ssim" else args.dist_metric

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        # net = net.module
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if args.overwrite_lr:
            for g in optimizer.param_groups:
                g['lr'] = args.learning_rate
    
    if args.cuda and args.use_data_parallel and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            args
        )
        loss = test_epoch(epoch, test_dataloader, net, criterion, args)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_dir = f"trained_compressai/{os.path.basename(args.dataset)}"
            os.makedirs(save_dir, exist_ok=True)
            if args.load_ntc:
                fname = f"{save_dir}/Cheng2020Lattice_fromNTC_q{args.ntc_quality}_{args.lattice_name}_{args.dist_metric}_lmbda{args.lmbda}.pt"
            else:
                fname = f"{save_dir}/Cheng2020Lattice_{args.lattice_name}_{args.dist_metric}_lmbda{args.lmbda}.pt"
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.module.state_dict() if args.use_data_parallel else net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                filename=fname
            )


if __name__ == "__main__":
    main(sys.argv[1:])