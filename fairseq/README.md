# Pre-processing, Training, and Evaluation on IWSLT'14 DE-En  dataset

## Dataset download and pre-processing
From the main directory, run the following command:
```
# Dowload the dataset
$ cd examples/translation/
$ bash prepare-iwslt14.sh
$ cd ../..

# Binarize the dataset:
$ TEXT=examples/translation/iwslt14.tokenized.de-en
$ fairseq-preprocess --source-lang de --target-lang en \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/iwslt14.tokenized.de-en
```

## Training
To train a model with a single GPU setup, you can use the following command:
```
$ export CUDA_VISIBLE_DEVICES=0
$ data_dir=data-bin/iwslt14.tokenized.de-en
$ model_dir=checkpoints/iwslt_d4
$ python -u train.py $data_dir --arch transformer_iwslt_de_en \
    --share-decoder-input-output-embed --noise-sparse-attention \
    --optimizer adam --lr 0.0005 -s de -t en \
    --dropout 0.3 --max-tokens 5000 \
    --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-update 60000 \
    --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --adam-betas '(0.9, 0.98)' --save-dir $model_dir \
    --no-progress-bar --log-format json
```

## Evaluation
To evaluate a model, you can use the following command:
```
$ model_dir=checkpoints/iwslt_d4
$ model_path=$model_dir/checkpoint_best.pt
$ python generate.py data-bin/iwslt14.tokenized.de-en/ --path $model_path  --beam 5 --batch-size 128 --remove-bpe
```

# Pre-processing, Training, and Evaluation on WMT'14 En-De dataset

## Dataset download and pre-processing
Please first download the [preprocessed WMT'16 En-De data provided by Google](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8).
Then run the following command from the main directory:
```
# Extract the WMT'16 En-De data:
$ TEXT=wmt16_en_de_bpe32k
$ mkdir $TEXT
$ tar -xzvf wmt16_en_de.tar.gz -C $TEXT

# Preprocess the dataset with a joined dictionary:
$ fairseq-preprocess --source-lang en --target-lang de \
  --trainpref $TEXT/train.tok.clean.bpe.32000 \
  --validpref $TEXT/newstest2013.tok.bpe.32000 \
  --testpref $TEXT/newstest2014.tok.bpe.32000 \
  --destdir data-bin/wmt16_en_de_bpe32k \
  --nwordssrc 32768 --nwordstgt 32768 \
  --joined-dictionary
```

## Training

1. To train a base model:
```
$ export CUDA_VISIBLE_DEVICES=0,1
$ warmup=4000
$ data_dir=data-bin/wmt16_en_de_bpe32k
$ model_dir=checkpoints/wmt16_base_d4
$ python -u train.py $data_dir --arch transformer_wmt_en_de \
    --share-all-embeddings --noise-sparse-attention \
    --dropout 0.1 --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --lr 0.0007 --min-lr 1e-09 \
    --warmup-updates $warmup --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 --weight-decay 0.0 --max-tokens 4096 \
    --save-dir $model_dir --update-freq 4 --no-progress-bar \
    --log-format json --log-interval 1000 \
    --save-interval-updates 1000 --keep-interval-updates 20 \
    --fp16
```
1. To train a big model:
```
$ export CUDA_VISIBLE_DEVICES=0,1
$ warmup=4000
$ data_dir=data-bin/wmt16_en_de_bpe32k
$ model_dir=checkpoints/wmt16_big_d4
$ python -u train.py $data_dir --arch transformer_vaswani_wmt_en_de_big \
    --share-all-embeddings --noise-sparse-attention \
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
```

## Evaluation
To evaluate a model, you can use the following command:
```
# average last 10 checkpoints:
model_dir=checkpoints/wmt16_big_d4
model_path=$model_dir/average_model.pt
python scripts/average_checkpoints.py --inputs $model_dir --num-update-checkpoints 10 --output $model_path

# generate the translation:
$ model_dir=checkpoints/wmt16_big_d_4
model_path=$model_dir/average_model.pt
python generate.py data-bin/wmt16_en_de_bpe32k/ \
    --path $model_path --gen-subset test --beam 4 \
    --batch-size 128 --remove-bpe --lenpen 0.6 | tee /tmp/gen.out
    
# split compound words:
$ bash get_bleu.sh
```