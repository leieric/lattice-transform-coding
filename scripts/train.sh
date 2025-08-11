#!/bin/bash
python LTC/train_compander.py --n 2 --d 2 --dy 2 --n_train_samples 1000000 --num_eval_samples 1000000 --model_name "SquareCompanderNTC" --data_name "Gaussian" --lam_sweep 0.5 0.75 1 1.5 2 4 8 --epochs 2 --save

# Hexagonal A_2 Lattice
python LTC/train_compander.py --n 2 --d 2 --dy 2 --n_train_samples 1000000 --num_eval_samples 1000000 --model_name "LatticeCompander" --data_name "Gaussian" --transform_name "MLP" --eb_name "Flow_RealNVP" --lattice_name "HexagonalUnitVol" --lam_sweep 0.5 0.75 1 1.5 2 4 8 --epochs 2 --save

# D^*_4 Lattice
# python train_compander.py --n 4 --num_eval_samples 1000000 --transform_name "LinearNoBias" --lattice_name "DnDualUnitVol" --lam_sweep 0.5 0.75 1 1.5 2 4 8
python LTC/train_compander.py --n 4 --d 4 --dy 4 --n_train_samples 1000000 --num_eval_samples 200000 --model_name "LatticeCompander" --data_name "Gaussian" --transform_name "LinearNoBias" --eb_name "Flow_RealNVP" --lattice_name "DnDualUnitVol" --lam_sweep 0.5 0.75 1 1.5 2 4 8 --epochs 2 --save

# E_8 Lattice
python LTC/train_compander.py --n 8 --d 8 --dy 8 --n_train_samples 1000000 --num_eval_samples 200000 --model_name "LatticeCompander" --data_name "Gaussian" --transform_name "LinearNoBias" --eb_name "Flow_RealNVP" --lattice_name "E8" --lam_sweep 0.5 0.75 1 1.5 2 4 8 --epochs 2 --save

# n = 8 with Square
python LTC/train_compander.py --n 8 --d 8 --dy 8 --n_train_samples 1000000 --num_eval_samples 200000 --model_name "LatticeCompander" --data_name "Gaussian" --transform_name "LinearNoBias" --eb_name "Flow_RealNVP" --lattice_name "Square" --lam_sweep 0.5 0.75 1 1.5 2 4 8 --epochs 2 --save

# Barnes Wall Lattice
python LTC/train_compander.py --n 16 --d 16 --dy 16 --n_train_samples 1000000 --num_eval_samples 200000 --model_name "LatticeCompander" --data_name "Gaussian" --transform_name "LinearNoBias" --eb_name "Flow_RealNVP" --lattice_name "BarnesWallUnitVol" --lam_sweep 0.5 0.75 1 1.5 2 4 8 --epochs 10 --loss_func "L2" --save

# Leech Lattice
python LTC/train_compander.py --n 24 --d 24 --dy 24 --n_train_samples 1000000 --num_eval_samples 1000000 --model_name "LatticeCompander" --data_name "Gaussian" --transform_name "LinearNoBias" --eb_name "Flow_RealNVP" --lattice_name "Leech2UnitVol" --lam_sweep 1.5 2 4 8 16 --epochs 10 --loss_func "L2" --save --N_integral 8192
