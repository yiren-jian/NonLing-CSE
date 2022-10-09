# VisualCSE

This repo covers the implementation of **VisualCSE** in  the following paper:  **Non-Linguistic Supervision for Contrastive Learning of Sentence Embeddings** by [Yiren Jian](https://cs.dartmouth.edu/~yirenjian/), [Chongyang Gao](https://gcyzsl.github.io/) and [Soroush Vosoughi](https://www.cs.dartmouth.edu/~soroush/), accepted to NeurIPS 2022.

<img src="overview.png" width="600">

Our code is  heavily borrowed from [SimCSE](https://github.com/princeton-nlp/SimCSE).

## Requirements

We use `python=3.8`, `pytorch-1.10` with `cuda-11.3`, `torchvision-0.11.3`, `torchaudio-0.10.2`, `transformers-4.5.0`.

We tested our code on Nvidia RTX-6000 (for base and unsupervised models), and RTX-A6000 (for large and supervised models).
```
conda create -n VisualCSE-py38 python=3.8
conda activate VisualCSE-py38
pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

And pip install the folllowing:
```
transformers==4.5.0
scipy==1.5.4
datasets==1.2.1
pandas==1.1.5
scikit-learn==0.24.0
prettytable==2.1.0
gradio
setuptools==49.3.0
```

**Note that using different hardwares and softwares will require re-searching the hyper-parameters reported in the paper.**

## Download data
Following [SimCSE](https://github.com/princeton-nlp/SimCSE), downloading the **text ** training data:
```bash
cd data
bash download_wiki.sh
bash download_nli.sh
```

Also, for evaluating the models, downloading the data for downtream tasks:
```bash
cd SentEval/data/downstream
bash download_dataset.sh
```

## Images for training VisualCSE
To train VisualCSE, you will also need a sub-sized ImageNet dataset.

Assuming the `NonLing-CSE` is at your home directory, i.e. `/home/user/NonLing-CSE`, you will need a sub-sized ImageNet at `/home/user/datasets`

```
...
/datasets/sampled-Imagenet/n01871265/n01871265_7110.JPEG
/datasets/sampled-Imagenet/n01871265/n01871265_3107.JPEG
/datasets/sampled-Imagenet/n01871265/n01871265_8069.JPEG
/datasets/sampled-Imagenet/n01871265/n01871265_4796.JPEG
/datasets/sampled-Imagenet/n01871265/n01871265_6902.JPEG
/datasets/sampled-Imagenet/n01871265/n01871265_389.JPEG
...
```

All images used in our study is provided in `sampled-Imagenet.txt`. Note that theses images (60 classes and 500 images per class) are randomly selected from full ImageNet-1k.

## Training VisualCSE
**To faithfully reproduce the results in our paper, please use the same hardwares and software versions listed above.** We provide an example of training unsupervised VisualCSE with `bert-base-uncased`:
```bash
CUDA_VISIBLE_DEVICES=0 bash run_unsup_example.sh 48 0.07 1e-6 0.05 SupCon 0.0
```

It will train and evaluate the model and produce the result (Using RTX-6000):

|          | STS12 | STS13 | STS14 | STS15 | STS16 | STS-B | SICK-R | Avg.  |
| :------: | :---: | :---: | :---: | :---: | :---: | :---: | :----: | :---: |
| RTX-6000 | 71.16 | 83.29 | 75.13 | 81.59 | 80.05 | 80.03 | 71.23  | 77.50 |

To train VisualCSE with `roberta-base`, you will need to edit `run_unsup_example.sh` by replacing `bert-base-uncased` with `roberta-base`, learning rate to `2e-5` and batch size `128`:
```bash
CUDA_VISIBLE_DEVICES=0 bash run_unsup_example.sh 48 0.07 1e-7 0.2 SupCon 0.0
```

It will produce the result  (Using RTX-6000):
|          | STS12 | STS13 | STS14 | STS15 | STS16 | STS-B | SICK-R | Avg.  |
| :------: | :---: | :---: | :---: | :---: | :---: | :---: | :----: | :---: |
| RTX-6000 | 70.41 | 83.51 | 74.87 | 82.79 | 81.67 | 81.89 | 69.95  | 77.87 |

We provide our example training terminal outputs in `training_logs`.

## Evaluating our pre-trained models
You can directly evaluate our *unsupervised* models uploaded to Huggingface:
```bash
python evaluation.py --model_name_or_path gcyzsl/unsup-VisualCSE-bert-base-uncased --pooler cls_before_pooler --task_set sts --mode test
```

You can replace `model_name_or_path` with:
```
gcyzsl/unsup-VisualCSE-bert-base-uncased
gcyzsl/unsup-VisualCSE-roberta-base
gcyzsl/unsup-VisualCSE-roberta-large
```

If you are evaluating *supervised* models, remember to reset `--pooler`:
```bash
python evaluation.py --model_name_or_path gcyzsl/sup-VisualCSE-bert-base-uncased --pooler cls --task_set sts --mode test
```

## Re-searching hyper-parameters
In case of using different hardwares and softwares, you may find useful to re-search the hyper-parameters. We provide example scripts in `search_hyperparams`.

## Acknowlegements
Thanks to [SimCSE](https://github.com/princeton-nlp/SimCSE) for the preliminary implementations.
