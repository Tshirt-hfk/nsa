model_dir=checkpoints/wmt16_big_d_4
model_path=$model_dir/average_model_103.pt
python generate.py data-bin/wmt16_en_de_bpe32k/ --path $model_path --gen-subset test --beam 4 --batch-size 128 --remove-bpe --lenpen 0.6 | tee /tmp/gen.out
