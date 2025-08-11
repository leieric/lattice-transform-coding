#!/bin/bash

# 1. One-shot Coding
# Square
python train_model.py --d 16 --dy 16 --d_hidden 500 --n_train_samples 1000000 --num_eval_samples 1000000 --model_name "NTC" --data_name "Physics" --lam_sweep 0.05 0.5 1 2 4 8 --epochs 10

# Hexagonal
python train_model.py --d 16 --dy 2 --d_hidden 500 --model_name "LTCDither" --data_name "Physics" --transform_name "MLP" --eb_name "FactorizedPrior" --lattice_name "HexagonalUnitVol" --lam_sweep 1 2 4 8 16 32 64 --N_integral 4096 --epochs 10 --lr 1e-4 --batch_size 16

# D_n^*
python train_model.py --d 16 --dy 4 --d_hidden 500 --model_name "LTCDither" --data_name "Physics" --transform_name "MLP" --eb_name "FactorizedPrior" --lattice_name "DnDualUnitVol" --lam_sweep 1 2 4 8 16 32 64 --N_integral 4096 --epochs 10 --lr 1e-4 --batch_size 16

# 2. Block Coding
# E_8 Lattice
python train_model.py --n 8 --d 16 --dy 2 --batch_size 32 --model_name "BlockLTCZeroMean" --data_name "PhysicsBlock" --transform_name "LinearNoBias" --eb_name "FactorizedPrior" --lattice_name "E8" --lam_sweep 0.5 1 2 4 8 16 32 64 --epochs 10 --save --lr 5e-5
python train_model.py --n 8 --d 16 --dy 4 --batch_size 32 --model_name "BlockLTCZeroMean" --data_name "PhysicsBlock" --transform_name "LinearNoBias" --eb_name "Flow_RealNVP2" --lattice_name "E8" --lam_sweep 8 16 32 64 --epochs 10 --save --lr 1e-4 --N_integral 4096 --N_integral_eval 4096 --MC_method "sobol_scrambled"

# Leech Lattice
python train_model.py --n 24 --d 16 --dy 2 --batch_size 16 --model_name "BlockLTCZeroMean" --data_name "PhysicsBlock" --transform_name "LinearNoBias" --eb_name "Flow_RealNVP2" --lattice_name "Leech2UnitVol" --lam_sweep 2 4 8 16 32 64 --epochs 10 --save --lr 5e-5 --N_integral 4096 --N_integral_eval 4096 --MC_method "sobol_scrambled"
python train_model.py --n 24 --d 16 --dy 4 --batch_size 16 --model_name "BlockLTCZeroMean" --data_name "PhysicsBlock" --transform_name "LinearNoBias" --eb_name "FactorizedPrior" --lattice_name "Leech2UnitVol" --lam_sweep 8 16 32 64 --epochs 40 --save --lr 5e-5 --N_integral 4096 --N_integral_eval 4096 --MC_method "sobol_scrambled"
