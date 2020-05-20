CUDA_VISIBLE_DEVICES=0 python main.py \
                       --dropouti 0.4 --dropoutl 0.29 --dropouth 0.225 --seed 28 \
                       --batch_size 12 --lr 20.0 --epoch 1000 --nhid 960 --nhidlast 620 \
                       --emsize 280 --n_experts 15 --save PTB --single_gpu