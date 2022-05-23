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
n_gpu=4
#n_gpu=1

backPruneRatio=0.8

lr=1e-2

# with layernorm
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar100-lr${lr}-pruneAllR${backPruneRatio}wLN-quantize --learning_rate ${lr} --num_workers 5 --output_dir ${save_dir} \
--dataset cifar100 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--new_backrazor --back_prune_ratio ${backPruneRatio} --backrazor_with_layernorm --quantize \
--train_batch_size 8 --eval_batch_size 8


###################### cifar10 B128 backRazor + half ######################
save_dir="/mnt/models/Ziyu_model/M2M_ViT"
#save_dir="."

#devices="8,9,10,11"
#devices="12,13,14,15"
devices="14"
port=7282
n_gpu=1

backPruneRatio=0.95

for lr in 0.03
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar10-lr${lr}-B128-pruneAllR${backPruneRatio}wLN-half --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset cifar10 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--new_backrazor --back_prune_ratio ${backPruneRatio} --backrazor_with_layernorm --backrazor_half \
--train_batch_size 128 --eval_batch_size 128 \
--num_steps 20000 --eval_every 1000
done


###################### cifar10 B128 backRazor quantize ######################
save_dir="/mnt/models/Ziyu_model/M2M_ViT"
#save_dir="."

#devices="8,9,10,11"
#devices="12,13,14,15"
devices="4"
port=7282
n_gpu=1

backPruneRatio=0.8

for lr in 0.01
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar10-lr${lr}-B128-pruneAllR${backPruneRatio}wLN-quantize --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset cifar10 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--new_backrazor --back_prune_ratio ${backPruneRatio} --backrazor_with_layernorm --quantize \
--train_batch_size 128 --eval_batch_size 128 \
--num_steps 20000 --eval_every 1000
done


###################### cifar100 B128 backRazor + half ######################
save_dir="/mnt/models/Ziyu_model/M2M_ViT"
#save_dir="."

#devices="8,9,10,11"
#devices="12,13,14,15"
devices="8"
port=4292
n_gpu=1

backPruneRatio=0.8

for lr in 0.01
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar100-lr${lr}-B128-pruneAllR${backPruneRatio}wLN-half --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset cifar100 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--new_backrazor --back_prune_ratio ${backPruneRatio} --backrazor_with_layernorm \
--train_batch_size 128 --eval_batch_size 128 --backrazor_half  \
--num_steps 20000 --eval_every 1000
done

###################### cifar100 B128 backRazor quantize ######################
#save_dir="/mnt/models/Ziyu_model/M2M_ViT"
save_dir="."

#devices="8,9,10,11"
#devices="12,13,14,15"
devices="5"
port=7296
n_gpu=1

backPruneRatio=0.8

for lr in 0.01
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar100-lr${lr}-B128-pruneAllR${backPruneRatio}wLN-quantize --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset cifar100 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--new_backrazor --back_prune_ratio ${backPruneRatio} --backrazor_with_layernorm \
--train_batch_size 128 --eval_batch_size 128 --quantize \
--num_steps 20000 --eval_every 1000
done


###################### pet37 B128 backRazor ######################
save_dir="/mnt/models/Ziyu_model/M2M_ViT"
#save_dir="."

#devices="8,9,10,11"
#devices="12,13,14,15"
devices="3"
port=5995
n_gpu=1

backPruneRatio=0.8

for lr in 3e-3
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name Pet37-lr${lr}-B128-coTuneTrans-pruneAllR${backPruneRatio}wLN --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset Pet37 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--new_backrazor --back_prune_ratio ${backPruneRatio} --backrazor_with_layernorm \
--train_batch_size 128 --eval_batch_size 128 --cotuning_trans \
--num_steps 20000 --eval_every 1000
done


###################### pet37 B128 backRazor + half ######################
save_dir="/mnt/models/Ziyu_model/M2M_ViT"
#save_dir="."

#devices="8,9,10,11"
#devices="12,13,14,15"
devices="8,9"
port=5992
n_gpu=2

backPruneRatio=0.8

for lr in 3e-3
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name Pet37-lr${lr}-B128-coTuneTrans-pruneAllR${backPruneRatio}wLN-half --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset Pet37 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--new_backrazor --back_prune_ratio ${backPruneRatio} --backrazor_with_layernorm --backrazor_half \
--train_batch_size 128 --eval_batch_size 128 --cotuning_trans \
--num_steps 20000 --eval_every 1000
done

###################### pet37 B128 backRazor + half + 2k ######################
save_dir="/mnt/models/Ziyu_model/M2M_ViT"
#save_dir="."

#devices="8,9,10,11"
#devices="12,13,14,15"
devices="8,9"
port=5926
n_gpu=2

backPruneRatio=0.95

for lr in 1e-2
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name Pet37-lr${lr}-B128-coTuneTrans-pruneAllR${backPruneRatio}wLN-half-2k --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset Pet37 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--new_backrazor --back_prune_ratio ${backPruneRatio} --backrazor_with_layernorm --backrazor_half \
--train_batch_size 128 --eval_batch_size 128 --cotuning_trans \
--num_steps 2000 --eval_every 1000
done

for lr in 3e-3 1e-2 3e-2
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name Pet37-lr${lr}-B128-coTuneTrans-pruneAllR${backPruneRatio}wLN-half-4k --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset Pet37 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--new_backrazor --back_prune_ratio ${backPruneRatio} --backrazor_with_layernorm --backrazor_half \
--train_batch_size 128 --eval_batch_size 128 --cotuning_trans \
--num_steps 4000 --eval_every 1000
done

###################### pet37 B128 backRazor + quantize + 2k ######################
save_dir="/mnt/models/Ziyu_model/M2M_ViT"
#save_dir="."

#devices="8,9,10,11"
#devices="12,13,14,15"
devices="10,11"
port=5936
n_gpu=2

backPruneRatio=0.8

for lr in 1e-2
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name Pet37-lr${lr}-B128-coTuneTrans-pruneAllR${backPruneRatio}wLN-quantize-2k --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset Pet37 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--new_backrazor --back_prune_ratio ${backPruneRatio} --backrazor_with_layernorm --quantize \
--train_batch_size 128 --eval_batch_size 128 --cotuning_trans \
--num_steps 2000 --eval_every 1000
done


##############################
--train_batch_size 2 --eval_batch_size 2
