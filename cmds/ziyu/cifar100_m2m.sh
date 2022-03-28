# calculate the mean of attn on one dataset and replace the attn map with it
save_dir="/mnt/models/Ziyu_model/M2M_ViT"
#save_dir="."

devices="0,1,2,3"
#devices="4,5,6,7"
port=4579
n_gpu=4

lr=1e-2

CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar100-lr${lr}-attnAsParam --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset cifar100 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz --attn_replace "parameter"


# calculate the mean of attn on one dataset and replace the attn map with it
save_dir="/mnt/models/Ziyu_model/M2M_ViT"
#save_dir="."

devices="0,1,2,3"
#devices="4,5,6,7"
port=4579
n_gpu=4

lr=1e-2

CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar100-lr${lr}-attnAsParam-clsTokenStay --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset cifar100 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz --attn_replace "parameter" \
--cls_token_stay


#################################################################################
# TODO: reset the root, gpu number and batch size
--train_batch_size 2 --eval_batch_size 2

bash
conda activate mae
cd /home/t-xiaochen/ViT-pytorch-1