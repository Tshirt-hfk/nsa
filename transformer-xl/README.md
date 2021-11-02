# Pre-processing, Training, and Evaluation on enwik8 dataset

## Dataset download and pre-processing
From the main directory, run the following command:
```
$ mkdir -p data
$ cd data
$ mkdir -p enwik8
$ cd enwik8
$ wget --continue http://mattmahoney.net/dc/enwik8.zip
$ wget https://raw.githubusercontent.com/salesforce/awd-lstm-lm/master/data/enwik8/prep_enwik8.py
$ python prep_enwik8.py
$ cd ../pytorch
```

## Training
To train a model with four single GPUs setup, run the following command:
```
bash run_enwik8_nsp.sh train
```

## Evaluation
To evaluate a model, you can use the following command:
```
bash run_enwik8_nsp.sh eval
```

# Pre-processing, Training, and Evaluation on wikitext-103 dataset

## Dataset download and pre-processing
From the main directory, run the following command:
```
$ mkdir -p data
$ cd data
$ wget --continue https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
$ unzip -q wikitext-103-v1.zip
$ cd wikitext-103
$ mv wiki.train.tokens train.txt
$ mv wiki.valid.tokens valid.txt
$ mv wiki.test.tokens test.txt
$ cd ../pytorch
```

## Training
To train a model with four single GPUs setup, run the following command:
```
bash run_wt103_nsp.sh train
```

## Evaluation
To evaluate a model, you can use the following command:
```
bash run_wt103_nsp.sh eval
```