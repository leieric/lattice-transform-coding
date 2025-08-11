#!/bin/bash
python train_model.py --d 1 --dy 1 --n_train_samples 1000000 --num_eval_samples 200000 --model_name "NTC" --data_name "Laplace" --lam_sweep 0.5 0.75 1 1.5 2 4 8 --epochs 25

# E_8 Lattice
python train_model.py --n 8 --d 8 --dy 8 --n_train_samples 1000000 --num_eval_samples 1000000 --model_name "LTC" --data_name "Laplace" --transform_name "LinearNoBias" --eb_name "Flow_RealNVP" --lattice_name "E8" --lam_sweep 0.5 0.75 1 1.5 2 2.5 3 3.5 4 6 8 --epochs 10 --save

# Barnes Wall Lattice
python train_model.py --n 16 --d 16 --dy 16 --n_train_samples 1000000 --num_eval_samples 1000000 --model_name "LTC" --data_name "Laplace" --transform_name "LinearNoBias" --eb_name "Flow_RealNVP" --lattice_name "BarnesWallUnitVol" --lam_sweep 0.5 0.75 1 1.5 2 4 8 --epochs 10 --loss_func "L1" --save --N_integral 4096

# Leech Lattice
python train_model.py --n 24 --d 24 --dy 24 --n_train_samples 1000000 --num_eval_samples 1000000 --model_name "LTC" --data_name "Laplace" --transform_name "LinearNoBias" --eb_name "Flow_RealNVP" --lattice_name "Leech2UnitVol" --lam_sweep 1.5 2 2.5 3 3.5 4 5 6 8 --epochs 10 --loss_func "L1" --save --N_integral 8192 --lr 5e-5
