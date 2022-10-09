# Data for AudioCSE
Prepare audio data and the dataloader for `AudioCSE`. The code of this repo is heavily borrowed from [SSAST](https://github.com/YuanGongND/ssast).

## Overview
1. Data download at https://www.openslr.org/12. We use `train-clean-100` for `AudioCSE`. <br>
train-clean-100.tar.gz [6.3G]   (training set of 100 hours "clean" speech )   Mirrors: [US]   [EU]   [CN]  
train-clean-360.tar.gz [23G]   (training set of 360 hours "clean" speech )   Mirrors: [US]   [EU]   [CN]  
train-other-500.tar.gz [30G]   (training set of 500 hours "other" speech )   Mirrors: [US]   [EU]   [CN]

2. Run prep_librispeech.py for data preparation.<br>
   a. the path of the data should be changed to your local directory in line 82: librispeech100_path.<br>
   b. the output file name is in line 83: 'librispeech_tr100_cut'.

3. Run run_dataloader to test. <br>
   a. input_json in line 46 should be changed to the directory of the output file generated in step 2.


## Example
In this section, we provide an example of our setup. Hope this will help.

1. We first `git clone https://github.com/YuanGongND/ssast` in our home directory `/home/yiren/ssast`.

2. Then, we download `train-clean-100.tar.gz` and put in `/home/yiren/ssast`. Unzip the `.tar.gz` file to `/home/yiren/ssast/data/LibriSpeech/train-clean-100`, under which are subfolders like:

```
...
/home/yiren/ssast/data/LibriSpeech/train-clean-100/87
/home/yiren/ssast/data/LibriSpeech/train-clean-100/8747
/home/yiren/ssast/data/LibriSpeech/train-clean-100/8770
/home/yiren/ssast/data/LibriSpeech/train-clean-100/8797
/home/yiren/ssast/data/LibriSpeech/train-clean-100/887
/home/yiren/ssast/data/LibriSpeech/train-clean-100/89
/home/yiren/ssast/data/LibriSpeech/train-clean-100/8975
/home/yiren/ssast/data/LibriSpeech/train-clean-100/909
/home/yiren/ssast/data/LibriSpeech/train-clean-100/911
...
```

3. Put the files `dataloader.py`, `prep_librispeech.py`, `run_dataloader.py` of **this** repo to (replace) `/home/yiren/ssast/src/prep_data/librispeech`.

4. At `/home/yiren/ssast/src/prep_data/librispeech`, run `python prep_librispeech.py`. Now you will have `librispeech_tr100_cut.json` in `/home/yiren/ssast/src/prep_data/librispeech`.

DONE!
