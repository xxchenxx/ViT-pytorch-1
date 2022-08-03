save_dir="."

#devices="0"
#port=7296
#n_gpu=1

devices="0,1"
port=7296
n_gpu=2


for lr in 0.3 0.1 0.03
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
ViT/train.py --name aircraft-lr${lr}-B128E100 --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset aircraft --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--train_batch_size 128 --eval_batch_size 128 \
--num_steps 6000 --eval_every 1000
done


save_dir="."

#devices="0"
#port=7296
#n_gpu=1

devices="6,7"
port=5896
n_gpu=2


for lr in 0.1 0.03 0.01
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
ViT/train.py --name aircraft-lr${lr}-B128E100-colorDistort --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset aircraft --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--train_batch_size 128 --eval_batch_size 128 --color-distort \
--num_steps 6000 --eval_every 1000
done

#devices="0,1,2,3"
#port=7256
#n_gpu=4

backPruneRatio=0.8
lr=0.01

CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
ViT/train.py --name aircraft-lr${lr}-B128-BackRazor${backPruneRatio} --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset aircraft --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--new_backrazor --back_prune_ratio ${backPruneRatio} \
--train_batch_size 128 --eval_batch_size 128 \
--num_steps 20000 --eval_every 1000 \
--train_batch_size 32 --eval_batch_size 32


save_dir="."

devices="0,1"
port=7296
n_gpu=2


for lr in 0.3 0.1 0.03 0.01
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
ViT/train.py --name aircraft-lr${lr}-B128E100-coTuneTrans-colorJitter-lrHead10x --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset aircraft --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--train_batch_size 128 --eval_batch_size 128 \
--cotuning_trans --HeadLr10times \
--num_steps 6000 --eval_every 1000
done


################ backRazor #######################
save_dir="."

#devices="0"
#port=7296
#n_gpu=1

devices="2,3"
port=6777
n_gpu=2

backPruneRatio=0.8

for lr in 0.3 0.1 0.03 0.01
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
ViT/train.py --name aircraft-lr${lr}-B128E100-coTuneTrans-colorJitter-lrHead10x-BackRazor${backPruneRatio} --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset aircraft --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--new_backrazor --back_prune_ratio ${backPruneRatio} \
--train_batch_size 128 --eval_batch_size 128 \
--cotuning_trans --HeadLr10times \
--num_steps 6000 --eval_every 1000
done


###################### aircraft fix the backbone ######################
#save_dir="/mnt/models/Ziyu_model/M2M_ViT"
save_dir="."

#devices="8,9,10,11"
#devices="12,13,14,15"
devices="1"
port=4229
n_gpu=1

for lr in 1.0 0.3 0.1 0.03
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name aircraft-lr${lr}-B128-fixBackbone --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset aircraft --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--train_batch_size 128 --eval_batch_size 128 --fix_backbone \
--num_steps 6000 --eval_every 1000
done
