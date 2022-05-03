###################### aircraft ######################
#save_dir="/mnt/models/Ziyu_model/M2M_ViT"
save_dir="."

#devices="8,9,10,11"
#devices="12,13,14,15"
devices="0"
port=4569
n_gpu=1

lr=1e-2

CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar10-lr${lr} --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset aircraft --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--train_batch_size 16 --eval_batch_size 16

