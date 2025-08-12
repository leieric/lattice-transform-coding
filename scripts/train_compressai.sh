#!/bin/bash

# DnDual
python train_compressai.py --dist_metric "mse" --lattice_name "DnDualUnitVol" --channels 128 --N_integral 2048 --dataset "data/vimeo_septuplet" -lr 5e-5 --batch-size 8 --save --cuda --lambda 0.0018 --epochs 60 
python train_compressai.py --dist_metric "mse" --lattice_name "DnDualUnitVol" --channels 128 --N_integral 2048 --dataset "data/vimeo_septuplet" -lr 5e-5 --batch-size 8 --save --cuda --lambda 0.0067 --epochs 60 
python train_compressai.py --dist_metric "mse" --lattice_name "DnDualUnitVol" --channels 192 --N_integral 2048 --dataset "data/vimeo_septuplet" -lr 5e-5 --batch-size 6 --save --cuda --lambda 0.0250 --epochs 60 
python train_compressai.py --dist_metric "mse" --lattice_name "DnDualUnitVol" --channels 192 --N_integral 2048 --dataset "data/vimeo_septuplet" -lr 5e-5 --batch-size 6 --save --cuda --lambda 0.0483 --epochs 60 

# E8
python train_compressai.py --dist_metric "mse" --lattice_name "E8Product" --channels 128 --N_integral 2048 --dataset "data/vimeo_septuplet" -lr 2.5e-5 --batch-size 8 --save --cuda --lambda 0.0018 --epochs 60 
python train_compressai.py --dist_metric "mse" --lattice_name "E8Product" --channels 128 --N_integral 2048 --dataset "data/vimeo_septuplet" -lr 2.5e-5 --batch-size 8 --save --cuda --lambda 0.0067 --epochs 60 
python train_compressai.py --dist_metric "mse" --lattice_name "E8Product" --channels 192 --N_integral 2048 --dataset "data/vimeo_septuplet" -lr 2.5e-5 --batch-size 6 --save --cuda --lambda 0.0250 --epochs 60 
python train_compressai.py --dist_metric "mse" --lattice_name "E8Product" --channels 192 --N_integral 2048 --dataset "data/vimeo_septuplet" -lr 2.5e-5 --batch-size 6 --save --cuda --lambda 0.0483 --epochs 60 


# Leech
python train_compressai.py --dist_metric "mse" --lattice_name "Leech2ProductUnitVol" --channels 120 --N_integral 2048 --dataset "data/vimeo_septuplet" -lr 2.5e-5 --batch-size 6 --save --cuda --lambda 0.0018 --epochs 60 
python train_compressai.py --dist_metric "mse" --lattice_name "Leech2ProductUnitVol" --channels 120 --N_integral 2048 --dataset "data/vimeo_septuplet" -lr 2.5e-5 --batch-size 6 --save --cuda --lambda 0.0067 --epochs 60 
python train_compressai.py --dist_metric "mse" --lattice_name "Leech2ProductUnitVol" --channels 120 --N_integral 2048 --dataset "data/vimeo_septuplet" -lr 2.5e-5 --batch-size 6 --save --cuda --lambda 0.0130 --epochs 60 
python train_compressai.py --dist_metric "mse" --lattice_name "Leech2ProductUnitVol" --channels 192 --N_integral 1024 --dataset "data/vimeo_septuplet" -lr 2.5e-5 --batch-size 6 --save --cuda --lambda 0.0250 --epochs 60 
python train_compressai.py --dist_metric "mse" --lattice_name "Leech2ProductUnitVol" --channels 192 --N_integral 1024 --dataset "data/vimeo_septuplet" -lr 2.5e-5 --batch-size 6 --save --cuda --lambda 0.0483 --epochs 60 


# finetune from NTC:

# E8
python train_compressai.py --dist_metric "mse" --lattice_name "E8Product" --channels 128 --N_integral 2048 --dataset "data/vimeo_septuplet" -lr 1e-4 --batch-size 8 --save --cuda --lambda 0.0018 --epochs 30 --load_ntc --ntc_quality 1
python train_compressai.py --dist_metric "mse" --lattice_name "E8Product" --channels 128 --N_integral 2048 --dataset "data/vimeo_septuplet" -lr 1e-4 --batch-size 8 --save --cuda --lambda 0.0067 --epochs 30 --load_ntc --ntc_quality 3
python train_compressai.py --dist_metric "mse" --lattice_name "E8Product" --channels 192 --N_integral 2048 --dataset "data/vimeo_septuplet" -lr 1e-4 --batch-size 6 --save --cuda --lambda 0.0250 --epochs 30 --load_ntc --ntc_quality 5
python train_compressai.py --dist_metric "mse" --lattice_name "E8Product" --channels 192 --N_integral 2048 --dataset "data/vimeo_septuplet" -lr 1e-4 --batch-size 6 --save --cuda --lambda 0.0483 --epochs 30 --load_ntc --ntc_quality 6