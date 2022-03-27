# calculate the mean of attn on one dataset and replace the attn map with it
#save_dir="/mnt/models/Ziyu_model/M2M_ViT"
save_dir="."

#devices="8,9,10,11"
devices="0"
port=4569
n_gpu=1

lr=1e-3

CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar10-lr${lr} --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset cifar100 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz --attn_replace "parameter" \
--train_batch_size 2 --eval_batch_size 2


# TODO: reset the root and

bash
conda activate mae
cd /home/t-xiaochen/ViT-pytorch-1