# baseline
save_dir="/mnt/models/Ziyu_model/M2M_ViT"
#save_dir="."

devices="0,1"
#devices="4,5,6,7"
#devices="8,9,10,11"
#devices="12,13,14,15"
port=5848
n_gpu=2

lr=3e-2
# with layernorm
CUDA_VISIBLE_DEVICES=${devices} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port ${port}  \
train.py --name resnet-cifar100-lr${lr}-newNorm-B256-Step40k --learning_rate ${lr} --num_workers 5 --output_dir ${save_dir} \
--dataset cifar100 --model_type resnet50 \
--train_batch_size 256 --eval_batch_size 256 --num_steps 40000 --eval_every 400


