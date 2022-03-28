# calculate the mean of attn on one dataset and replace the attn map with it
#save_dir="/mnt/models/Ziyu_model/M2M_ViT"
save_dir="."

devices="4"
port=4574

lr=1e-2

CUDA_VISIBLE_DEVICES=${devices} python3 utils/conv_approximate.py cifar100-lr${lr}-attnAsParam-approx --lr ${lr} \
--update_iter 10000 --conv_size 51 --attn_model_path checkpoints/cifar100-lr1e-4-attnAsParam/estimate.pth

# TODO: reset the root, gpu number and batch size
--train_batch_size 2 --eval_batch_size 2


