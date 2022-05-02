# baseline
#save_dir="/mnt/models/Ziyu_model/M2M_ViT"
save_dir="."

#devices="0,1,2,3"
devices="4,5,6,7"
#devices="8,9,10,11"
#devices="12,13,14,15"
port=4999
#n_gpu=4
n_gpu=1

lr=1e-2

# with layernorm
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name resnet-cifar100-lr${lr} --learning_rate ${lr} --num_workers 5 --output_dir ${save_dir} \
--dataset cifar100 --model_type resnet50 \
--train_batch_size 8 --eval_batch_size 8


# prune the backward new
#save_dir="/mnt/models/Ziyu_model/M2M_ViT"
save_dir="."

#devices="0,1,2,3"
devices="4,5,6,7"
#devices="8,9,10,11"
#devices="12,13,14,15"
port=4999
#n_gpu=4
n_gpu=1

lr=1e-2

backPruneRatio=0.8

# with layernorm
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name resnet-cifar100-lr${lr}-backPruneR${backPruneRatio} --learning_rate ${lr} --num_workers 5 --output_dir ${save_dir} \
--dataset cifar100 --model_type resnet50 \
--new_backrazor --back_prune_ratio ${backPruneRatio} \
--train_batch_size 8 --eval_batch_size 8