###################### cifar10 B128 ######################
save_dir="/mnt/models/Ziyu_model/M2M_ViT"
#save_dir="."

#devices="8,9,10,11"
#devices="12,13,14,15"
devices="5"
port=4577
n_gpu=1

for lr in 0.003
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar10-lr${lr}-B128 --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset cifar10 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--train_batch_size 128 --eval_batch_size 128 \
--num_steps 20000 --eval_every 1000
done


###################### cifar10 B128 test the memory ######################
#save_dir="/mnt/models/Ziyu_model/M2M_ViT"
save_dir="."

#devices="8,9,10,11"
#devices="12,13,14,15"
devices="5"
port=4577
n_gpu=1

for lr in 0.003
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar10-lr${lr}-B128 --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset cifar10 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--train_batch_size 16 --eval_batch_size 16 --memory_cost_profile \
--num_steps 20000 --eval_every 1000 --bitfit
done

###################### cifar10 B128 fix the backbone ######################
save_dir="/mnt/models/Ziyu_model/M2M_ViT"
#save_dir="."

#devices="8,9,10,11"
#devices="12,13,14,15"
devices="0"
port=4271
n_gpu=1

for lr in 0.3
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar10-lr${lr}-B128-fixBackbone --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset cifar10 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--train_batch_size 128 --eval_batch_size 128 --fix_backbone \
--num_steps 20000 --eval_every 1000
done

###################### cifar100 B128 ######################
save_dir="/mnt/models/Ziyu_model/M2M_ViT"
#save_dir="."

#devices="8,9,10,11"
#devices="12,13,14,15"
devices="3"
port=4571
n_gpu=1

for lr in 0.03 0.01 0.003
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar100-lr${lr}-B128 --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset cifar100 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--train_batch_size 128 --eval_batch_size 128 \
--num_steps 20000 --eval_every 1000
done

###################### cifar100 fix the backbone ######################
#save_dir="/mnt/models/Ziyu_model/M2M_ViT"
save_dir="."

#devices="8,9,10,11"
#devices="12,13,14,15"
devices="4,5,6,7"
port=4729
n_gpu=1

for lr in 0.3
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar100-lr${lr}-B128-fixBackbone --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset cifar100 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--train_batch_size 128 --eval_batch_size 128 --fix_backbone \
--num_steps 20000 --eval_every 1000
done

###################### aircraft ######################
#save_dir="/mnt/models/Ziyu_model/M2M_ViT"
save_dir="."

#devices="8,9,10,11"
#devices="12,13,14,15"
#devices="0,1,2,3"
devices="4,5,6,7"
port=4889
n_gpu=4

for lr in 1e-2 3e-2 3e-3
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name aircraft-lr${lr}-B128-coTuneTrans-HeadLr10times --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset aircraft --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--train_batch_size 128 --eval_batch_size 128 --cotuning_trans --HeadLr10times \
--num_steps 20000 --eval_every 1000
done


###################### pet37 ######################
save_dir="/mnt/models/Ziyu_model/M2M_ViT"
#save_dir="."

#devices="8,9,10,11"
#devices="12,13,14,15"
#devices="0,1,2,3"
devices="8"
port=5895
n_gpu=1

for lr in 1e-3 3e-4
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name Pet37-lr${lr}-B128-coTuneTrans --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset Pet37 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--train_batch_size 128 --eval_batch_size 128 --cotuning_trans \
--num_steps 20000 --eval_every 1000
done

###################### pet37 fix backbone ######################
save_dir="/mnt/models/Ziyu_model/M2M_ViT"
#save_dir="."

#devices="8,9,10,11"
#devices="12,13,14,15"
#devices="0,1,2,3"
devices="8"
port=5898
n_gpu=1

for lr in 1e-3 3e-4
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name Pet37-lr${lr}-B128-coTuneTrans-fixBackbone --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset Pet37 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--train_batch_size 128 --eval_batch_size 128 --cotuning_trans --fix_backbone \
--num_steps 20000 --eval_every 1000
done


###################### flower ######################
save_dir="/mnt/models/Ziyu_model/M2M_ViT"
#save_dir="."

#devices="8,9,10,11"
#devices="12,13,14,15"
#devices="0,1,2,3"
devices="10"
port=5895
n_gpu=1

for lr in 1e-2 3e-3 1e-3 3e-4
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name flowers102-lr${lr}-B128 --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset flowers102 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--train_batch_size 128 --eval_batch_size 128  \
--num_steps 2000 --eval_every 1000
done

###################### flowers102 fix backbone ######################
save_dir="/mnt/models/Ziyu_model/M2M_ViT"
#save_dir="."

#devices="8,9,10,11"
#devices="12,13,14,15"
#devices="0,1,2,3"
devices="11"
port=6898
n_gpu=1

for lr in 3 0.3 0.03
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name flowers102-lr${lr}-B128-fixBackbone-2k --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset flowers102 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--train_batch_size 128 --eval_batch_size 128 --fix_backbone \
--num_steps 2000 --eval_every 1000
done


###################### stanford_cars ######################
save_dir="/mnt/models/Ziyu_model/M2M_ViT"
#save_dir="."

#devices="8,9,10,11"
#devices="12,13,14,15"
#devices="0,1,2,3"
devices="9"
port=9395
n_gpu=1

for lr in 1e-1 3e-2 1e-2
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name stanford_car-lr${lr}-B128-4k --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset stanford_car --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--train_batch_size 128 --eval_batch_size 128  \
--num_steps 4000 --eval_every 1000
done

###################### stanford_cars fix backbone ######################
save_dir="/mnt/models/Ziyu_model/M2M_ViT"
#save_dir="."

#devices="8,9,10,11"
#devices="12,13,14,15"
#devices="0,1,2,3"
devices="8"
port=5398
n_gpu=1

for lr in 3 0.3 0.03
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name stanford_car-lr${lr}-B128-real-2k-fixBackbone --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset stanford_car --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--train_batch_size 128 --eval_batch_size 128 --fix_backbone \
--num_steps 2000 --eval_every 1000
done
