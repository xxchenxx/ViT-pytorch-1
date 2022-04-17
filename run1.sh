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

port=4573

CUDA_VISIBLE_DEVICES=0,1,2,3 bash cmds/to_run/back_prune_attn_tune_lr.sh ${save_dir} ${port} 3e-2 &
CUDA_VISIBLE_DEVICES=4,5,6,7 bash cmds/to_run/back_prune_attn_tune_lr.sh ${save_dir} ${port} 1e-1

wait
