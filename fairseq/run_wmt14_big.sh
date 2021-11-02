export CUDA_VISIBLE_DEVICES=0,1
warmup=4000
data_dir=data-bin/wmt16_en_de_bpe32k
model_dir=checkpoints/wmt16_big_nsa-d4
python -u train.py $data_dir --arch transformer_vaswani_wmt_en_de_big \
    --share-all-embeddings --sparse-attention 'NSA' \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.001 --lr-scheduler inverse_sqrt \
    --warmup-updates $warmup --warmup-init-lr 1e-07 \
    --dropout 0.3 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --update-freq 48 --max-tokens 4778 \
    --no-progress-bar --save-interval-updates 50 --keep-interval-updates 20 \
    --log-format json --log-interval 50 \
    --save-dir $model_dir \
    --fp16
