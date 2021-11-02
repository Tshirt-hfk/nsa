# Noise Sparse Attention： a sparse attention through noise selection

## Requirements and Installation
* [PyTorch](http://pytorch.org/) version >= 1.0.0
* Python version >= 3.6
* **To use Noise Sparse Attention on machine translation tasks, you need to install fairseq** and develop locally:
```bash
cd nsp/fairseq
pip install --editable ./
```

## Training, Evaluation

For training, evaluation, and results, see below links. To ease reproduction of our results

* [Machine Translation](fairseq/README.md)
* [Language Modeling](transformer-xl/README.md)

## Results

For machine translation

| Method | IWSLT'14 De-En | WMT'14 En-De | 
| ---- | ---- | ---- |
| Transformer | 34.82 | 29.11  | 
| Noise Sparse Transformer  | 35.33 | 29.47  | 

For language modeling

| Method | Enwiki8 | WT-103 | 
| ----   | ----    | ----   |
| Transformer-XL | 1.060 | 23.78  | 
| Noise Sparse Transformer-XL  | 1.047 | 23.28  | 
