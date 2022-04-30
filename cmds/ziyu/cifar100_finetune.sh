###################### baseline ######################
#save_dir="/mnt/models/Ziyu_model/M2M_ViT"
save_dir="."

#devices="8,9,10,11"
#devices="12,13,14,15"
devices="0"
port=4569
n_gpu=1

lr=1e-2

CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar10-lr${lr} --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset cifar100 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz --memory_cost_profile

###################### rigL prune ######################
save_dir="/mnt/models/Ziyu_model/M2M_ViT"
#save_dir="."

#devices="0,1,2,3"
devices="4,5,6,7"
#devices="8,9,10,11"
#devices="12,13,14,15"
port=5589
n_gpu=4

lr=1e-2
prune_dense_ratio=0.5
prune_death_rate=1.0
prune_avg_magni_var_alpha=1.0
prune_inv=200
prune_end=8000

CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar100-lr${lr}-rigL_D${prune_dense_ratio}Dth${prune_death_rate}Walpha${prune_avg_magni_var_alpha}Inv${prune_inv}To${prune_end} \
--learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset cifar100 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--prune --prune_dense_ratio ${prune_dense_ratio} --prune_death_rate ${prune_death_rate} \
--prune_avg_magni_var_alpha ${prune_avg_magni_var_alpha} --prune_inv ${prune_inv} --prune_end ${prune_end}

###################### init prune with taylor distance to the pretrained model, rigL prune ######################
save_dir="/mnt/models/Ziyu_model/M2M_ViT"
#save_dir="."

#devices="0,1,2,3"
devices="4,5,6,7"
#devices="8,9,10,11"
#devices="12,13,14,15"
port=5598
n_gpu=4

lr=1e-2
prune_dense_ratio=0.5
prune_death_rate=0.0
prune_avg_magni_var_alpha=1.0
prune_inv=10000
prune_end=8000

init_iter=20

CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar100-lr${lr}-rigL_D${prune_dense_ratio}Dth${prune_death_rate}Walpha${prune_avg_magni_var_alpha}Inv${prune_inv}To${prune_end}-initTaylorDistAlpha${prune_avg_magni_var_alpha}Iter${init_iter} \
--learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset cifar100 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--prune --prune_dense_ratio ${prune_dense_ratio} --prune_death_rate ${prune_death_rate} \
--prune_avg_magni_var_alpha ${prune_avg_magni_var_alpha} --prune_inv ${prune_inv} --prune_end ${prune_end} \
--prune_init_method taylor_change_magni_var --prune_init_iter_time ${init_iter}


###################### prune_after_softmax, init prune with taylor distance to the pretrained model, rigL prune ######################
save_dir="/mnt/models/Ziyu_model/M2M_ViT"
#save_dir="."

#devices="0,1,2,3"
#devices="4,5,6,7"
devices="8,9,10,11"
#devices="12,13,14,15"
port=6001
n_gpu=4

lr=1e-2
prune_dense_ratio=0.5
prune_death_rate=0.0
prune_avg_magni_var_alpha=1.0
prune_inv=10000
prune_end=8000

init_iter=10

CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar100-lr${lr}-rigL_D${prune_dense_ratio}Dth${prune_death_rate}Walpha${prune_avg_magni_var_alpha}Inv${prune_inv}To${prune_end}-initTaylorDistAlpha${prune_avg_magni_var_alpha}Iter${init_iter}-pruneAfterSM \
--learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset cifar100 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--prune --prune_dense_ratio ${prune_dense_ratio} --prune_death_rate ${prune_death_rate} \
--prune_avg_magni_var_alpha ${prune_avg_magni_var_alpha} --prune_inv ${prune_inv} --prune_end ${prune_end} \
--prune_init_method taylor_change_magni_var --prune_init_iter_time ${init_iter} --prune_after_softmax


###################### taylor score prune, init prune with taylor distance to the pretrained model, rigL prune ######################
save_dir="/mnt/models/Ziyu_model/M2M_ViT"
#save_dir="."

devices="0,1,2,3"
#devices="4,5,6,7"
#devices="8,9,10,11"
#devices="12,13,14,15"
port=6049
n_gpu=4

lr=1e-2
prune_dense_ratio=0.5
prune_death_rate=0.3
prune_avg_magni_var_alpha=1.0
prune_inv=200
prune_end=8000

init_iter=10

CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar100-lr${lr}-rigL_taylor_D${prune_dense_ratio}Dth${prune_death_rate}Walpha${prune_avg_magni_var_alpha}Inv${prune_inv}To${prune_end}-initTaylorDistAlpha${prune_avg_magni_var_alpha}Iter${init_iter} \
--learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset cifar100 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--prune --prune_dense_ratio ${prune_dense_ratio} --prune_death_rate ${prune_death_rate} \
--prune_avg_magni_var_alpha ${prune_avg_magni_var_alpha} --prune_inv ${prune_inv} --prune_end ${prune_end} \
--prune_init_method taylor_change_magni_var --prune_init_iter_time ${init_iter} --prune_death_mode taylor_magni_var

##############################
--train_batch_size 2 --eval_batch_size 2


bash
conda activate mae
cd /home/t-xiaochen/ViT-pytorch-1
