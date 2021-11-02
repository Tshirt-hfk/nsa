export CUDA_VISIBLE_DEVICES=0
model_dir=checkpoints/iwslt_d4
model_path=$model_dir/checkpoint_best.pt
python generate.py data-bin/iwslt14.tokenized.de-en/ --path $model_path  --beam 5 --batch-size 128 --remove-bpe
