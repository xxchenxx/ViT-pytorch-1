# baseline
save_dir="."
devices="0,1,2,3"
lr=3e-2

CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=2  train.py \
--name cifar10-lr${lr} --learning_rate ${lr} \
--dataset cifar100 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz


##############################
--train_batch_size 2 --eval_batch_size 2