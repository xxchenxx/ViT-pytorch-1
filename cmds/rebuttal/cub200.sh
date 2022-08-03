###################### cub200 fix the backbone ######################
#save_dir="/mnt/models/Ziyu_model/M2M_ViT"
save_dir="."

#devices="8,9,10,11"
#devices="12,13,14,15"
devices="0"
port=4229
n_gpu=1

for lr in 1.0 0.3 0.1 0.03
do
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cub200-lr${lr}-B128-fixBackbone --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset cub200 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--train_batch_size 128 --eval_batch_size 128 --fix_backbone \
--num_steps 6000 --eval_every 1000
done
