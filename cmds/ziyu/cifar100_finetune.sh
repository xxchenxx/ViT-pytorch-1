# baseline
save_dir="/mnt/models/Ziyu_model/M2M_ViT"
#save_dir="."

devices="0,1,2,3"
port=4565
n_gpu=4

lr=3e-2

CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar10-lr${lr} --learning_rate ${lr} \
--dataset cifar100 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz


##############################
--train_batch_size 2 --eval_batch_size 2