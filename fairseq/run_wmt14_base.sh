export CUDA_VISIBLE_DEVICES=0,1
warmup=4000
data_dir=data-bin/wmt16_en_de_bpe32k
model_dir=checkpoints/wmt16_base_nsa-d4
python -u train.py $data_dir --arch transformer_wmt_en_de \
    --share-all-embeddings --sparse-attention 'NSA' \
    --dropout 0.1 --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --lr 0.0007 --min-lr 1e-09 \
    --warmup-updates $warmup --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 --weight-decay 0.0 --max-tokens 4096 \
    --save-dir $model_dir --update-freq 4 --no-progress-bar \
    --log-format json --log-interval 1000 \
    --save-interval-updates 1000 --keep-interval-updates 20 \
    --fp16
