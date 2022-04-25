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

##############################
--train_batch_size 2 --eval_batch_size 2
