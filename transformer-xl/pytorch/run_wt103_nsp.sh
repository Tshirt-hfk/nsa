#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python -u train.py \
        --cuda \
        --data ../data/wikitext-103/ \
        --dataset wt103 \
        --adaptive \
	--noise_sparse_attention \
	--n_layer 16 \
        --d_model 410 \
        --n_head 10 \
        --d_head 41 \
        --d_inner 2100 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00025 \
        --eta_min 0.00001 \
        --warmup_step 0 \
        --max_step 300000 \
        --tgt_len 150 \
        --mem_len 150 \
        --eval_tgt_len 150 \
        --batch_size 60 \
        --multi_gpu \
        --gpu0_bsz 12 \
        --work_dir ./LM-TFM-wt103/nsa_d2_300000_etm1/ \
        ${@:2}
    echo 'Run fine-tuning...'
    python -u train.py \
        --cuda \
        --data ../data/wikitext-103/ \
        --dataset wt103 \
        --adaptive \
	--noise_sparse_attention \
	--n_layer 16 \
        --d_model 410 \
        --n_head 10 \
        --d_head 41 \
        --d_inner 2100 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00001 \
        --warmup_step 0 \
        --max_step 200000 \
        --tgt_len 150 \
        --mem_len 150 \
        --eval_tgt_len 150 \
        --batch_size 60 \
        --multi_gpu \
        --gpu0_bsz 12 \
	--restart \
	--restart_dir ./LM-TFM-wt103/nsa_d2_300000_etm1/ \
	--work_dir ./LM-TFM-wt103/nsa_d2_300000_etm1_ft/ \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --cuda \
        --data ../data/wikitext-103/ \
        --dataset wt103 \
        --tgt_len 64 \
        --mem_len 640 \
        --clamp_len 400 \
        --same_length \
        --split test \
        --work_dir ./LM-TFM-wt103/nsa_d2_300000_etm1_ft/ \
        ${@:2}
else
    echo 'unknown argment 1'
fi
