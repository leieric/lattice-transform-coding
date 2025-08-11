#!/bin/bash

# only Block Coding here
# E_8 Lattice
python train_model.py --n 8 --d 1024 --n_train_samples 1000000 --num_eval_samples 200000 --model_name "BlockLTC" --data_name "SawbridgeBlock" --transform_name "Linear" --eb_name "Flow_RealNVP" --lattice_name "E8" --lam_sweep 128 64 32 16 8 4 2 1 --epochs 5 --save
# Barnes Wall Lattice
python train_model.py --n 16 --d 1024 --n_train_samples 1000000 --num_eval_samples 200000 --model_name "BlockLTC" --data_name "SawbridgeBlock" --transform_name "Linear" --eb_name "Flow_RealNVP" --lattice_name "BarnesWallUnitVol" --lam_sweep 128 64 32 16 8 4 2 1 --epochs 10 --save

# Leech Lattice
python train_model.py --n 24 --d 1024 --n_train_samples 500000 --num_eval_samples 200000 --model_name "BlockLTC" --data_name "SawbridgeBlock" --transform_name "Linear" --eb_name "Flow_RealNVP" --lattice_name "Leech2UnitVol" --lam_sweep 128 64 32 16 8 4 2 1 --epochs 2 --save
  