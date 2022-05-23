# cifar100
#save_dir="/mnt/models/Ziyu_model/M2M_ViT"
save_dir="."

devices="0,1,2,3"
#devices="4,5,6,7"
#devices="8,9,10,11"
#devices="12,13,14,15"
port=6764
n_gpu=4

lr=1e-2
# with layernorm
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name resnet-cifar100-lr${lr}-newNorm-B256-Step40k --learning_rate ${lr} --num_workers 5 --output_dir ${save_dir} \
--dataset cifar100 --model_type resnet50 \
--train_batch_size 256 --eval_batch_size 256 --num_steps 20000 --eval_every 200

# cifar100, memory measuring
#save_dir="/mnt/models/Ziyu_model/M2M_ViT"
save_dir="."

devices="0"
#devices="4,5,6,7"
#devices="8,9,10,11"
#devices="12,13,14,15"
port=6764
n_gpu=1

lr=1e-2
# with layernorm
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name resnet-cifar100-lr${lr}-newNorm-B256-Step40k --learning_rate ${lr} --num_workers 5 --output_dir ${save_dir} \
--dataset cifar100 --model_type resnet50 --memory_cost_profile \
--train_batch_size 16 --eval_batch_size 16 --num_steps 20000 --eval_every 200


# aircraft
#save_dir="/mnt/models/Ziyu_model/M2M_ViT"
save_dir="."

#devices="0,1"
#devices="4,5,6,7"
#devices="8,9,10,11"
#devices="12,13,14,15"

devices="3"
port=6729
n_gpu=1

for lr in 1e-3
do
# with layernorm
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name resnet-aircraft-lr${lr}-newNorm-B256-Step40k-HeadLr10times --learning_rate ${lr} --num_workers 5 --output_dir ${save_dir} \
--dataset aircraft --model_type resnet50 \
--train_batch_size 12 --eval_batch_size 12 --num_steps 20000 --eval_every 200 --HeadLr10times
done


save_dir="."

devices="3"
port=6857
n_gpu=1

lr=1e-2

CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port} \
train.py --name resnet-aircraft-lr${lr}-newNorm-B256-Step40k-HeadLr10times-augResizeFirst --learning_rate ${lr} --num_workers 5 --output_dir ${save_dir} \
--dataset aircraft --model_type resnet50 \
--train_batch_size 48 --eval_batch_size 100 --num_steps 10000 --eval_every 1000 --HeadLr10times --train_resize_first


save_dir="."

devices="4,5,6,7"
port=6839
n_gpu=4

for lr in 1e-2 3e-2 3e-3 # with layernorm
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port} \
train.py --name resnet-aircraft-lr${lr}B256-newNorm-Step40k-HeadLr10times-coTuneTrans --learning_rate ${lr} --num_workers 5 --output_dir ${save_dir} \
--dataset aircraft --model_type resnet50 \
--train_batch_size 256 --eval_batch_size 256 --num_steps 20000 \
--eval_every 1000 --HeadLr10times --cotuning_trans
done

# pet
save_dir="."

devices="0,1,2,3"
port=6239
n_gpu=4

for lr in 1e-3 # with layernorm
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port} \
train.py --name resnet-Pet37-lr${lr}B256-newNorm-Step40k-HeadLr10times-coTuneTrans --learning_rate ${lr} --num_workers 5 --output_dir ${save_dir} \
--dataset Pet37 --model_type resnet50 \
--train_batch_size 256 --eval_batch_size 256 --num_steps 20000 \
--eval_every 200 --HeadLr10times --cotuning_trans
done


# cifar10 train from scratch
save_dir="/mnt/models/Ziyu_model/M2M_ViT"

devices="12"
port=6238
n_gpu=1

for lr in 0.1 # with layernorm
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port} \
train.py --name resnet-scratch-cifar10-lr${lr}B256-Step40k-coTuneTrans --learning_rate ${lr} --num_workers 5 --output_dir ${save_dir} \
--dataset cifar10 --model_type resnet50 \
--train_batch_size 256 --eval_batch_size 256 --num_steps 40000 --weight_decay 1e-4 \
--eval_every 200 --train_from_scratch --cotuning_trans --img_size 32
done

# cifar10 train from scratch with backRazor
save_dir="/mnt/models/Ziyu_model/M2M_ViT"

devices="10"
port=6236
n_gpu=1
backPruneRatio=0.9

for lr in 0.1
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port} \
train.py --name resnet-scratch-cifar10-lr${lr}B256-Step20k-coTuneTrans-backRazorR${backPruneRatio} --learning_rate ${lr} --num_workers 5 --output_dir ${save_dir} \
--dataset cifar10 --model_type resnet50 \
--train_batch_size 256 --eval_batch_size 256 --num_steps 20000 --weight_decay 1e-4 \
--new_backrazor --back_prune_ratio ${backPruneRatio} --backrazor_with_layernorm \
--eval_every 200 --train_from_scratch --cotuning_trans --img_size 32
done


# cifar100 train from scratch
save_dir="/mnt/models/Ziyu_model/M2M_ViT"

devices="11"
port=6237
n_gpu=1

for lr in 0.1 0.3 0.03 # with layernorm
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port} \
train.py --name resnet-scratch-cifar100-lr${lr}B256-Step20k-coTuneTrans --learning_rate ${lr} --num_workers 5 --output_dir ${save_dir} \
--dataset cifar100 --model_type resnet50 \
--train_batch_size 256 --eval_batch_size 256 --num_steps 20000 --weight_decay 1e-4 \
--eval_every 200 --train_from_scratch --cotuning_trans --img_size 32
done

# cifar100 train from scratch with backRazor
save_dir="/mnt/models/Ziyu_model/M2M_ViT"

devices="10"
port=6236
n_gpu=1
backPruneRatio=0.9

for lr in 0.1
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port} \
train.py --name resnet-scratch-cifar100-lr${lr}B256-Step20k-coTuneTrans-backRazorR${backPruneRatio} --learning_rate ${lr} --num_workers 5 --output_dir ${save_dir} \
--dataset cifar100 --model_type resnet50 \
--train_batch_size 256 --eval_batch_size 256 --num_steps 20000 --weight_decay 1e-4 \
--new_backrazor --back_prune_ratio ${backPruneRatio} --backrazor_with_layernorm \
--eval_every 200 --train_from_scratch --cotuning_trans --img_size 32
done
