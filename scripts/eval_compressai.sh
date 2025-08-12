#!/bin/bash

# evaluate: pretrained NTC models, adjust quality q according to CompressAI library for R-D sweep
python -m compressai.utils.eval_model pretrained /home/Shared/image_datasets/Kodak/1 -a "cheng2020-attn" -q 5 --entropy-estimation

# evaluate: trained LTC models, adjust checkpoint path for R-D sweep
# D_n^* lattice
python eval_compressai.py checkpoint data/Kodak/1 --entropy-estimation --path trained_compressai/vimeo_septuplet/Cheng2020Lattice_DnDualUnitVol_mse_lmbda0.0067_best.pt --lattice_name "DnDualUnitVol" --channels 128 --N_integral 2048 --cuda

# E8 lattice
python eval_compressai.py checkpoint data/Kodak/1 --entropy-estimation --path trained_compressai/vimeo_septuplet/Cheng2020Lattice_E8Product_mse_lmbda0.0067_best.pt --lattice_name "E8Product" --channels 128 --N_integral 2048 --cuda

# Leech lattice
python eval_compressai.py checkpoint data/Kodak/1 --entropy-estimation --path trained_compressai/vimeo_septuplet/Cheng2020Lattice_Leech2ProductUnitVol_mse_lmbda0.0067_best.pt --lattice_name "Leech2ProductUnitVol" --channels 120 --N_integral 2048 --cuda



# evaluate: load pretrained NTC weights into LTC, adjust quality q according to CompressAI library for R-D sweep
# D_n^* lattice
python eval_compressai.py pretrained data/Kodak/1 -a "cheng2020-attn" -q 1 --entropy-estimation --lattice_name "DnDualUnitVol" --channels 128 --N_integral 2048 --cuda

# E8 lattice
python eval_compressai.py pretrained data/Kodak/1 -a "cheng2020-attn" -q 5 --entropy-estimation --lattice_name "E8Product" --channels 192 --N_integral 2048 --cuda



# evaluate: finetuned LTC model from NTC weights
python eval_compressai.py checkpoint data/Kodak/1 --entropy-estimation --path trained_compressai/vimeo_septuplet/Cheng2020Lattice_fromNTC_q1_E8Product_mse_lmbda0.0018_best.pt --lattice_name "E8Product" --channels 128 --N_integral 2048 --cuda

python eval_compressai.py checkpoint data/Kodak/1 --entropy-estimation --path trained_compressai/vimeo_septuplet/Cheng2020Lattice_fromNTC_q3_E8Product_mse_lmbda0.0067_best.pt --lattice_name "E8Product" --channels 128 --N_integral 2048 --cuda

python eval_compressai.py checkpoint data/Kodak/1 --entropy-estimation --path trained_compressai/vimeo_septuplet/Cheng2020Lattice_fromNTC_q5_E8Product_mse_lmbda0.025_best.pt --lattice_name "E8Product" --channels 192 --N_integral 2048 --cuda

python eval_compressai.py checkpoint data/Kodak/1 --entropy-estimation --path trained_compressai/vimeo_septuplet/Cheng2020Lattice_fromNTC_q6_E8Product_mse_lmbda0.0483_best.pt --lattice_name "E8Product" --channels 192 --N_integral 2048 --cuda