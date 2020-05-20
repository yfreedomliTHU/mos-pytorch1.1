CUDA_VISIBLE_DEVICES=0 python finetune.py \
                       --dropouti 0.4 --dropoutl 0.29 \
                       --dropouth 0.225 --seed 28 --batch_size 12 --lr 25.0 \
                       --epoch 1000 --nhid 960 --emsize 280 --n_experts 15 \
                       --save PTB-20200510-191019 \
                       --single_gpu