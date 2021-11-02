model_dir=checkpoints/wmt16_big_d4
model_path=$model_dir/average_model.pt
python scripts/average_checkpoints.py --inputs $model_dir --num-update-checkpoints 10 --output $model_path
