###################### food101 fix the backbone ######################
#save_dir="/mnt/models/Ziyu_model/M2M_ViT"
save_dir="."

#devices="8,9,10,11"
#devices="12,13,14,15"
devices="3"
port=4232
n_gpu=1

for lr in 1.0 0.3 0.1 0.03
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name food101-lr${lr}-B128-fixBackbone --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset food101 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--train_batch_size 128 --eval_batch_size 128 --fix_backbone \
--num_steps 30000 --eval_every 1000
done

###################### food101 mesa ######################
save_dir="."

#devices="0"
#port=7296
#n_gpu=1

devices="4,5"
port=6728
n_gpu=2


for lr in 3e-2 1e-2 3e-3
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name food101-lr${lr}-B128E100-Mesa --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset food101 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--mesa \
--train_batch_size 128 --eval_batch_size 128 \
--num_steps 30000 --eval_every 1000
done


###################### food101 fix_mlps ######################
save_dir="."

#devices="0"
#port=7296
#n_gpu=1

devices="4,5"
port=6728
n_gpu=2


for lr in 3e-2 1e-2 3e-3
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name food101-lr${lr}-B128E100-fix_mlps --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset food101 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--fix_mlps \
--train_batch_size 128 --eval_batch_size 128 \
--num_steps 30000 --eval_every 1000
done


###################### food101 bitfit ######################
save_dir="."

#devices="0"
#port=7296
#n_gpu=1

devices="4,5"
port=6728
n_gpu=2


for lr in 1.0 0.3 0.1 3e-2 1e-2
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name food101-lr${lr}-B128E100-bitfit --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset food101 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--bitfit \
--train_batch_size 128 --eval_batch_size 128 \
--num_steps 30000 --eval_every 1000
done
