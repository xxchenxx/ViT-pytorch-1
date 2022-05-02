# fix mlps
#save_dir="/mnt/models/Ziyu_model/M2M_ViT"
save_dir="."

devices="0,1,2,3"
#devices="4,5,6,7"
#devices="8,9,10,11"
#devices="12,13,14,15"
port=4573
#n_gpu=4
n_gpu=2

lr=1e-3

CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar100-lr${lr}-fixmlps --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset cifar100 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz --fix_mlps \
--train_batch_size 4 --eval_batch_size 4

# fix mlps, prune the backward
#save_dir="/mnt/models/Ziyu_model/M2M_ViT"
save_dir="."

#devices="0,1,2,3"
#devices="4,5,6,7"
#devices="8,9,10,11"
devices="12,13,14,15"
port=4573
#n_gpu=4
n_gpu=2

pruneStoreAttn=0.5
pruneStoreAct=0.0

lr=1e-3

CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar100-lr${lr}-fixmlps-pruneStoreAttn${pruneStoreAttn}Act${pruneStoreAct} --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset cifar100 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz --fix_mlps \
--attn_store_prune --prune_ratio_attn_mat_store ${pruneStoreAttn} --prune_ratio_act_store ${pruneStoreAct} \
--train_batch_size 32 --eval_batch_size 32


# prune the backward (no fix mlp)
save_dir="/mnt/models/Ziyu_model/M2M_ViT"
#save_dir="."

#devices="0,1,2,3"
#devices="4,5,6,7"
devices="8,9,10,11"
#devices="12,13,14,15"
port=4781
#n_gpu=4
n_gpu=4

pruneStoreAttn=0.8
pruneStoreAct=0.8

lr=1e-2

CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar100-lr${lr}-pruneStoreAttn${pruneStoreAttn}Act${pruneStoreAct} --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset cifar100 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--attn_store_prune --prune_ratio_attn_mat_store ${pruneStoreAttn} --prune_ratio_act_store ${pruneStoreAct}

# prune the backward new
save_dir="/mnt/models/Ziyu_model/M2M_ViT"
#save_dir="."

#devices="0,1,2,3"
#devices="4,5,6,7"
#devices="8,9,10,11"
devices="12,13,14,15"
port=4799
n_gpu=4
#n_gpu=1

backPruneRatio=0.8

lr=1e-2

CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar100-lr${lr}-pruneAllR${backPruneRatio} --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset cifar100 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--new_backrazor --back_prune_ratio ${backPruneRatio}

# with layernorm
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar100-lr${lr}-pruneAllR${backPruneRatio}wLN --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset cifar100 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--new_backrazor --back_prune_ratio ${backPruneRatio} --backrazor_with_layernorm

# with gelu
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar100-lr${lr}-pruneAllR${backPruneRatio}wGELU --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset cifar100 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--new_backrazor --back_prune_ratio ${backPruneRatio} --backrazor_with_gelu



# prune the backward new
save_dir="/mnt/models/Ziyu_model/M2M_ViT"
#save_dir="."

devices="0,1,2,3"
#devices="4,5,6,7"
#devices="8,9,10,11"
#devices="12,13,14,15"
port=4999
#n_gpu=4
n_gpu=1

backPruneRatio=0.8

lr=1e-2

# with layernorm
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar100-lr${lr}-pruneAllR${backPruneRatio}wLN --learning_rate ${lr} --num_workers 5 --output_dir ${save_dir} \
--dataset cifar100 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--train_batch_size 128 --eval_batch_size 128  \
--new_backrazor --back_prune_ratio ${backPruneRatio} --backrazor_with_layernorm \

##############################
--train_batch_size 2 --eval_batch_size 2
