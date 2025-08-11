#!/bin/bash

# 1. One-shot coding
# Square
python train_model.py --d 2 --dy 2 --n_train_samples 1000000 --num_eval_samples 200000 --model_name "NTC" --data_name "Banana" --lam_sweep 0.1 0.25 0.5 1 2 4 8 16 32 64 --epochs 25 --lr 4e-5 --save

python train_model.py --n 1 --d 2 --dy 2 --n_train_samples 1000000 --num_eval_samples 1000000 --model_name "LTC" --data_name "Banana" --transform_name "MLP" --eb_name "Flow_RealNVP" --lattice_name "Square" --lam_sweep 0.1 1 2 4 8 16 32 64 --epochs 5 --save

# Hexagonal A_2 Lattice
python train_model.py --n 1 --d 2 --dy 2 --n_train_samples 1000000 --num_eval_samples 1000000 --model_name "LTCDither" --data_name "Banana" --transform_name "MLP2" --eb_name "FactorizedPrior" --lattice_name "HexagonalUnitVol" --lam_sweep 16 32 64 --epochs 5 --save

# 2. Block coding (BLTC)
# E_8 Lattice
python train_model.py --n 8 --d 2 --dy 1 --n_train_samples 1000000 --num_eval_samples 1000000 --model_name "BlockLTCZeroMean" --data_name "BananaBlock" --transform_name "LinearNoBias" --eb_name "Flow_RealNVP" --lattice_name "E8" --lam_sweep 1 2 3 4 6 8 12 16 --epochs 10 --save

# Leech Lattice
python train_model.py --n 24 --d 2 --dy 1 --n_train_samples 1000000 --num_eval_samples 1000000 --model_name "BlockLTCZeroMean" --data_name "BananaBlock" --transform_name "LinearNoBias" --eb_name "Flow_RealNVP" --lattice_name "Leech2UnitVol" --lam_sweep 0.1 0.25 0.5 1 2 4 8 16 32 64 --epochs 2 --save --N_integral 7000