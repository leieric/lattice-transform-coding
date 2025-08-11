#!/bin/bash

# One-shot coding
# Square
python train_model.py --d 33 --dy 33 --d_hidden 500 --model_name "NTC" --data_name "Speech" --lam_sweep 16 32 64 128 256 512 1024 --epochs 10 --save --batch_size 64 --lr 5e-5

# D_n^*
python train_model.py --d 33 --dy 4 --d_hidden 500 --model_name "LatticeCompanderDither" --data_name "Speech" --transform_name "MLP" --eb_name "Flow_RealNVP" --lattice_name "DnDualUnitVol" --lam_sweep 16 32 64 128 256 --N_integral 4096 --epochs 2 --lr 5e-5 --batch_size 32 

# E8
python train_model.py --d 33 --dy 8 --d_hidden 500 --model_name "LatticeCompanderDither" --data_name "Speech" --transform_name "MLP" --eb_name "FactorizedPrior" --lattice_name "E8" --lam_sweep 16 32 64 128 256 512 1024 --N_integral 4096 --epochs 10 --lr 1e-4 --batch_size 32 
