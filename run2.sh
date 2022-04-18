#DATA_PATH=${OUTPUT}/imagenet_new
OUTPUT_PATH=${OUTPUT}

echo "DATA_PATH is ${DATA_PATH}"
echo "OUTPUT_PATH is ${OUTPUT_PATH}"

#echo "ls DATASET"
#ls ${DATASET}
#echo "ls OUTPUT"
#ls ${OUTPUT}

pip install ml-collections

############# cmds #############
#save_dir="/mnt/models/Ziyu_model/M2M_ViT"
#save_dir="."
save_dir=${OUTPUT}/MAE_CL/vit

port=4581
#n_gpu=4
n_gpu=8

pruneStoreAttn=0.9

for pruneStoreAct in 0.5 0.7
do

lr=1e-2

python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name cifar100-lr${lr}-pruneStoreAttn${pruneStoreAttn}Act${pruneStoreAct} --learning_rate ${lr} --num_workers 2 --output_dir ${save_dir} \
--dataset cifar100 --model_type ViT-B_16 --pretrained_dir ${save_dir}/pretrain/ViT-B_16.npz \
--attn_store_prune --prune_ratio_attn_mat_store ${pruneStoreAttn} --prune_ratio_act_store ${pruneStoreAct}

done