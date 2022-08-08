###################### stanford_car fix the backbone ######################
#save_dir="/mnt/models/Ziyu_model/M2M_ViT"
save_dir="."

#devices="8,9,10,11"
#devices="12,13,14,15"
devices="4"
port=4233
n_gpu=1

for lr in 1.0 0.3 0.1 0.03
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name stanford_car-lr${lr}-B128-fixBackbone --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset stanford_car --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--train_batch_size 128 --eval_batch_size 128 --fix_backbone \
--num_steps 16000 --eval_every 1000
done


###################### stanford_car mesa ######################
save_dir="."

#devices="0"
#port=7296
#n_gpu=1

devices="6,7"
port=5488
n_gpu=2


for lr in 0.1 3e-2 1e-2
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name stanford_car-lr${lr}-B128E100-Mesa --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset stanford_car --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--mesa \
--train_batch_size 128 --eval_batch_size 128 \
--num_steps 16000 --eval_every 1000
done

###################### stanford_car fix_mlps ######################
save_dir="."

#devices="0"
#port=7296
#n_gpu=1

devices="6,7"
port=5488
n_gpu=2


for lr in 0.1 3e-2 1e-2
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name stanford_car-lr${lr}-B128E100-fix_mlps --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset stanford_car --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--fix_mlps \
--train_batch_size 128 --eval_batch_size 128 \
--num_steps 16000 --eval_every 1000
done

###################### stanford_car bitfit ######################
save_dir="."

#devices="0"
#port=7296
#n_gpu=1

devices="6,7"
port=5488
n_gpu=2


for lr in 0.1 3e-2 1e-2
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name stanford_car-lr${lr}-B128E100-bitfit --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset stanford_car --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--bitfit \
--train_batch_size 128 --eval_batch_size 128 \
--num_steps 16000 --eval_every 1000
done
