# AudioCSE

This repo covers the implementation of **AudioCSE** in  the following paper:  **Non-Linguistic Supervision for Contrastive Learning of Sentence Embeddings** by [Yiren Jian](https://cs.dartmouth.edu/~yirenjian/), [Chongyang Gao](https://gcyzsl.github.io/) and [Soroush Vosoughi](https://www.cs.dartmouth.edu/~soroush/), accepted to NeurIPS 2022.

Our code is  heavily borrowed from [SimCSE](https://github.com/princeton-nlp/SimCSE).

The code is directly adapted from `VisualCSE`. We apologize for the extremely messy code in this repo. You will find that we use `image` and `audio` interchangeably in our `src/models.py`, `src/trainer` and `args` in `train.py`. Please note that *they all refer to audio* in this repo.

## Requirements

We use `python=3.8`, `pytorch-1.10` with `cuda-11.3`, `torchvision-0.11.3`, `torchaudio-0.10.2`, `transformers-4.5.0`.

We tested our code on Nvidia RTX-6000 (for base and unsupervised models), and RTX-A6000 (for large and supervised models).
```
conda create -n AudioCSE-py38 python=3.8
conda activate AudioCSE-py38
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

**Additionally, you will need `pip install timm==0.4.5` for AuidoCSE.**

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

## Additional data for training AudioCSE
Besides the wiki text (downloaded from the previous step), you will need audio data to train AudioCSE. Please see detail in `data_preparation` folder.

## Training AudioCSE
**To faithfully reproduce the results in our paper, please use the same hardwares and software versions listed above.** We provide an example of training unsupervised AudioCSE with `bert-base-uncased`:
```bash
CUDA_VISIBLE_DEVICES=0 bash run_unsup_example.sh 48 0.07 2e-7 0.01 SupCon 0.0
```

It will train and evaluate the model and produce the result (Using RTX-6000):

|          | STS12 | STS13 | STS14 | STS15 | STS16 | STS-B | SICK-R | Avg.  |
| :------: | :---: | :---: | :---: | :---: | :---: | :---: | :----: | :---: |
| RTX-6000 | 71.65 | 84.27 | 76.69 | 83.22 | 78.69 | 79.94 | 70.49  | 77.85 |

To train VisualCSE with `roberta-base`, you will need to edit `run_unsup_example.sh` by replacing `bert-base-uncased` with `roberta-base`, learning rate to `2e-5` and batch size `128`:
```bash
CUDA_VISIBLE_DEVICES=0 bash run_unsup_example.sh 48 0.07 2e-7 0.02 SupCon 0.0  ### 48 0.07 3e-6 0.2 SupCon 0.0
```

It will produce the result  (Using RTX-6000):
|          | STS12 | STS13 | STS14 | STS15 | STS16 | STS-B | SICK-R | Avg.  |
| :------: | :---: | :---: | :---: | :---: | :---: | :---: | :----: | :---: |
| RTX-6000 | 68.44 | 83.93 | 75.77 | 82.38 | 82.07 | 81.63 | 70.56  | 77.83 |

We provide our example training terminal outputs in `training_logs`.

## Evaluating our pre-trained models
You can directly evaluate our *unsupervised* models uploaded to Huggingface:
```bash
python evaluation.py --model_name_or_path gcyzsl/unsup-AudioCSE-bert-base-uncased --pooler cls_before_pooler --task_set sts --mode test
```

You can replace `model_name_or_path` with:
```
gcyzsl/unsup-AudioCSE-bert-base-uncased
gcyzsl/unsup-AudioCSE-roberta-base
gcyzsl/unsup-AudioCSE-roberta-large
```

If you are evaluating *supervised* models, remember to reset `--pooler`:
```bash
python evaluation.py --model_name_or_path gcyzsl/sup-AudioCSE-bert-base-uncased --pooler cls --task_set sts --mode test
```

## Re-searching hyper-parameters
In case of using different hardwares and softwares, you may find useful to re-search the hyper-parameters. We provide example scripts in `search_hyperparams`.

## Acknowlegements
Thanks to [SimCSE](https://github.com/princeton-nlp/SimCSE) for the preliminary implementations.
