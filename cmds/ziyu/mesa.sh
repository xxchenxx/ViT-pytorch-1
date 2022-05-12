save_dir="/mnt/models/Ziyu_model/M2M_ViT"
#save_dir="."

#devices="0,1,2,3"
#devices="4,5,6,7"
#devices="8,9,10,11"
devices="12,13,14,15"
port=4579
n_gpu=4

lr=3e-3

CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar100-lr${lr}-mesa --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset cifar100 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz --mesa


# test
#save_dir="/mnt/models/Ziyu_model/M2M_ViT"
save_dir="."

devices="0,1,2,3"
#devices="4,5,6,7"
#devices="8,9,10,11"
#devices="12,13,14,15"
port=4579
n_gpu=1

lr=1e-2

CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar100-lr${lr}-mesa --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset cifar100 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz --mesa \
--train_batch_size 16 --eval_batch_size 16


###################### cifar10 B128 mesa ######################
save_dir="/mnt/models/Ziyu_model/M2M_ViT"
#save_dir="."

#devices="8,9,10,11"
#devices="12,13,14,15"
devices="13"
port=7579
n_gpu=1

for lr in 0.03 0.01 0.003
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar10-lr${lr}-B128-mesa --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset cifar10 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--train_batch_size 128 --eval_batch_size 128 --mesa \
--num_steps 20000 --eval_every 1000
done

###################### cifar100 B128 mesa ######################
save_dir="/mnt/models/Ziyu_model/M2M_ViT"
#save_dir="."

#devices="8,9,10,11"
#devices="12,13,14,15"
devices="14"
port=7595
n_gpu=1

for lr in 0.03 0.01 0.003
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar100-lr${lr}-B128-mesa --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset cifar100 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--train_batch_size 128 --eval_batch_size 128 --mesa \
--num_steps 20000 --eval_every 1000
done

###################### Pet37 B128 mesa ######################
save_dir="/mnt/models/Ziyu_model/M2M_ViT"
#save_dir="."

#devices="8,9,10,11"
#devices="12,13,14,15"
devices="3"
port=8295
n_gpu=1

for lr in 3e-4 1e-3 3e-3
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name Pet37-lr${lr}-B128-coTuneTrans-mesa --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset Pet37 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--train_batch_size 128 --eval_batch_size 128 --cotuning_trans --mesa \
--num_steps 20000 --eval_every 1000
done



##############################
--train_batch_size 2 --eval_batch_size 2
