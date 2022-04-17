save_dir=$1
port=$2

lr=$3
#n_gpu=4
n_gpu=4

pruneStoreAttn=0.9
pruneStoreAct=0.0

lr=3e-3

cmd="python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar100-lr${lr}-pruneStoreAttn${pruneStoreAttn}Act${pruneStoreAct} --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset cifar100 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--attn_store_prune --prune_ratio_attn_mat_store ${pruneStoreAttn} --prune_ratio_act_store ${pruneStoreAct}"

echo ${cmd}
${cmd}
