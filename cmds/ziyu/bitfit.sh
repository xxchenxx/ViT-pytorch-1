save_dir="/mnt/models/Ziyu_model/M2M_ViT"
#save_dir="."

#devices="0,1,2,3"
#devices="4,5,6,7"
#devices="8,9,10,11"
devices="12,13,14,15"
port=4573
n_gpu=4

lr=1e-1

CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar100-lr${lr}-bitfit --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset cifar100 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz --bitfit


###################### cifar10 B128 bitfit ######################
save_dir="/mnt/models/Ziyu_model/M2M_ViT"
#save_dir="."

#devices="8,9,10,11"
#devices="12,13,14,15"
devices="9"
port=5577
n_gpu=1

for lr in 0.3 0.1 0.03
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar10-lr${lr}-B128-bitfit --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset cifar10 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--train_batch_size 128 --eval_batch_size 128 --bitfit \
--num_steps 20000 --eval_every 1000
done

###################### cifar100 B128 bitfit ######################
save_dir="/mnt/models/Ziyu_model/M2M_ViT"
#save_dir="."

#devices="8,9,10,11"
#devices="12,13,14,15"
devices="10"
port=4591
n_gpu=1

for lr in 0.3 0.1 0.03
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar100-lr${lr}-B128-bitfit --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset cifar100 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--train_batch_size 128 --eval_batch_size 128 --bitfit \
--num_steps 20000 --eval_every 1000
done

###################### Pet37 B128 bitfit ######################
save_dir="/mnt/models/Ziyu_model/M2M_ViT"
#save_dir="."

#devices="8,9,10,11"
#devices="12,13,14,15"
devices="1"
port=4591
n_gpu=1

for lr in 0.03 0.01 0.003
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name Pet37-lr${lr}-B128-coTuneTrans-bitfit --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset Pet37 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--train_batch_size 128 --eval_batch_size 128 --cotuning_trans --bitfit \
--num_steps 20000 --eval_every 1000
done


##############################
--train_batch_size 2 --eval_batch_size 2
