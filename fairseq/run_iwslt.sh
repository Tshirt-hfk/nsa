export CUDA_VISIBLE_DEVICES=0
data_dir=data-bin/iwslt14.tokenized.de-en
model_dir=checkpoints/iwslt_d4
python -u train.py $data_dir --arch transformer_iwslt_de_en \
    --share-decoder-input-output-embed --sparse-attention NSA \
    --optimizer adam --lr 0.0005 -s de -t en \
    --dropout 0.3 --max-tokens 5000 \
    --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-update 60000 \
    --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --adam-betas '(0.9, 0.98)' --save-dir $model_dir \
    --no-progress-bar --log-format json
