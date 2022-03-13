###################### baseline ######################
save_dir="/mnt/models/Ziyu_model/M2M_ViT"
#save_dir="."

#devices="8,9,10,11"
devices="12,13,14,15"
port=4569
n_gpu=4

lr=1e-3

CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar10-lr${lr} --learning_rate ${lr} --num_workers 0 \
--dataset cifar100 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz

###################### pruned ######################
#save_dir="/mnt/models/Ziyu_model/M2M_ViT"
save_dir="."

devices="0,1,2,3"
#devices="12,13,14,15"
port=4569
n_gpu=2

lr=1e-2
prune_dense_ratio=0.5
prune_death_rate=0.1
prune_avg_magni_var_alpha=0.1
prune_inv=500
prune_end=8000

CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar10-lr${lr} --learning_rate ${lr} --num_workers 0 \
--dataset cifar100 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--prune --prune_dense_ratio ${prune_dense_ratio} --prune_death_rate ${prune_death_rate} \
--prune_avg_magni_var_alpha ${prune_avg_magni_var_alpha} --prune_inv ${prune_inv} --prune_end ${prune_end} \
--train_batch_size 2 --eval_batch_size 2

##############################
--train_batch_size 2 --eval_batch_size 2