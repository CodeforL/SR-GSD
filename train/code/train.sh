save="SRGSD"
model="SRGSD"
CUDA_VISIBLE_DEVICES=1 nohup python main.py --model $model --save $save --scale 2 --n_resgroups 10 --n_resblocks 20 --reset --chop --save_results --print_model --patch_size 96  --ext sep > ../experiment/$save.txt 2>&1 &