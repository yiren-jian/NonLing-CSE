### By Chongyang Gao and Yiren Jian
### Adapted from https://github.com/YuanGongND/ssast

# -*- coding: utf-8 -*-
# @Time    : 7/11/21 6:55 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : prep_librispeech.py

# prepare librispeech data for ssl pretraining

import os,torchaudio,pickle,json,time
import re
import pandas as pd
def walk(path, name):
    sample_cnt = 0
    pathdata = os.walk(path)
    wav_list = []
    begin_time = time.time()
    mid = []
    mid_number = {}
    for root, dirs, files in pathdata:
        # print(root)
        # print(dirs)
        # print(files)
        for file in files:
            if file.endswith('.flac'):
                sample_cnt += 1
                # print(file)
                c_name = re.findall(r"(.*?)-.*", file)[0]
                # print(file, c_name)
                # class_name = re.findall(".*\\(.*)-.*",file)
                # print(file, class_name)
                if c_name not in mid:
                    mid.append(c_name)
                if c_name not in mid_number:
                    mid_number[c_name] = 1
                else:
                    mid_number[c_name] += 1
                cur_path = root + os.sep + file
                # give a dummy label of 'speech' ('/m/09x0r' in AudioSet label ontology) to all librispeech samples
                # the label is not used in the pretraining, it is just to make the dataloader.py satisfy.
                cur_dict = {"wav": cur_path, "labels": c_name}
                wav_list.append(cur_dict)

                if sample_cnt % 1000 == 0:
                    end_time = time.time()
                    print('find {:d}k .wav files, time eclipse: {:.1f} seconds.'.format(int(sample_cnt/1000), end_time-begin_time))
                    begin_time = end_time
                if sample_cnt % 1e4 == 0:
                    with open(name + '.json', 'w') as f:
                        json.dump({'data': wav_list}, f, indent=1)
                    print('file saved.')
    print(sample_cnt)    ### total number of samples in datasets
    display_name = mid
    index = [i for i in range(len(mid))]
    a = sorted(mid_number.items(), key=lambda x: x[1], reverse=True)

    # dataframe = pd.DataFrame({'index':index,'mid':mid,'display_name':display_name})
    # dataframe.to_csv(name+'_names2ids.csv', index=None)


    with open(name + '.json', 'w') as f:
        json.dump({'data': wav_list, 'top_k': a, 'csv': {'index':index,'mid':mid,'display_name':display_name}}, f, indent=1)


    # with open(name + '_counts.json', 'w') as outfile:
    #   json.dump(a, outfile)

# combine json files
def combine_json(file_list, name='librispeech_tr960'):
    wav_list = []
    for file in file_list:
        with open(file + '.json', 'r') as f:
            cur_json = json.load(f)
        wav_list = wav_list + cur_json['data']
    with open(name + '.json', 'w') as f:
        json.dump({'data': wav_list}, f, indent=1)

if __name__ == '__main__':
    librispeech100_path = '/home/yiren/ssast/data/LibriSpeech'
    walk(librispeech100_path, 'librispeech_tr100_cut')
