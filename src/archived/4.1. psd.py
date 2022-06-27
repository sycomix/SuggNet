# %%
# power analysis
# setup
from types import new_class
from typing import overload
import mne
import matplotlib.pyplot as plt
from mne.time_frequency.psd import psd_welch
import pandas as pd
import os.path as op
from mne.time_frequency import psd_multitaper
import numpy as np
import pickle

#%%

# def psd(data_dir, subjects, task, freq_bands = dict(
#   delta=(2, 4), theta=(5, 7),
#   alpha=(8, 12), beta=(13, 29),
#   gamma=(30, 45), broadband=(2,45)) ):
# '''
# the directory of your bids derivatives
# subject: a list of subjects id
# task: the name of task
# freq_bands: a dictionaty of frequency bands

# '''
def rename_key(key):
    return '-'.join(key.split('-')[2:])

ba_list = {'FL': ['Fp1', 'F3', 'F7', 'AF3', 'F1', 'F5', 'FT7'],
 'CL': ['C3', 'FC1', 'FC3', 'FC5', 'C1', 'C5'],
 'PL': ['P3', 'P7', 'CP1', 'CP3', 'CP5', 'TP7', 'P1', 'P5'],
 'OL': ['O1', 'EOG1', 'PO3'],
 'FR': ['Fp2', 'F4', 'F8', 'AF4', 'F2', 'F6', 'FT8'],
 'CR': ['C4', 'FC2', 'FC4', 'FC6', 'C2', 'C6'],
 'PR': ['P4', 'P8', 'CP2', 'CP4', 'CP6', 'TP8', 'P2', 'P6'],
 'OR': ['O2', 'EOG2', 'PO4'],
 'FZ': ['Fpz', 'Fz'],
 'CZ': ['Cz', 'ECG'],
 'PZ': ['CPz', 'Pz'],
 'OZ': ['POz', 'Oz'],
 'all': 'all'}

freq_bands = dict(
  delta=(1, 4), theta=(4, 8),
  alpha=(8, 12), beta=(12, 30),
  gamma=(30, 40), broadband=(1,40))
psd_dict = {}
for i in ba_list:
  for j in freq_bands:
    psd_dict[i+'-'+j] = []

psd_total = pd.DataFrame(index=['empty'], columns=psd_dict.keys())
psd_total_unaggregated = {}
freq_total_unaggregated = {}
# open epochs
subjects = ['04','05','06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '23', '24', '25']

# tasks = ['baseline1', 'baseline2']
data_dir = '/Users/yeganeh/Codes/otka-preprocessing/data/Main-study/derivatives/mne-bids-pipeline'
tasks = ['baseline1', 'baseline2',
         'induction1', 'induction2', 'induction3', 'induction4',
         'experience1', 'experience2', 'experience3', 'experience4']
          
n_ses = '01'
n_jobs = 8
for n_sub in subjects:
  for task in tasks:
    epoch_name = f'sub-{n_sub}_ses-{n_ses}_task-{task}_proc-clean_epo.fif'
    dir = op.join(data_dir, f'sub-{n_sub}/ses-{n_ses}/eeg/{epoch_name}')
    epoch = mne.read_epochs(dir)
    
    ## creat empty lists of brain areas
    ba_list = {}
    for j in ['L','R','Z']:
      for i in ['F','C','P','O']:
        ba_list[i+j] = []

    ### chategorize EEG channels according to their name
    for i in range(len(epoch.ch_names)):
      ch_nam = epoch.ch_names[i]
      temp = ch_nam.lower()
      if temp[-1].isnumeric():
        if int(temp[-1]) % 2 == 0:
          if 'f' in temp and 'c' not in temp:
            ba_list['FR'].append(ch_nam)
          elif 'c' in temp and 'p' not in temp:
            ba_list['CR'].append(ch_nam)
          elif 'p' in temp and 'o' not in temp:
            ba_list['PR'].append(ch_nam)
          elif 'o' in temp:
            ba_list['OR'].append(ch_nam)
        else:
          if 'f' in temp and 'c' not in temp:
            ba_list['FL'].append(ch_nam)
          elif 'c' in temp and 'p' not in temp:
            ba_list['CL'].append(ch_nam)
          elif 'p' in temp and 'o' not in temp:
            ba_list['PL'].append(ch_nam)
          elif 'o' in temp:
            ba_list['OL'].append(ch_nam)
      else:
          if 'f' in temp and 'c' not in temp:
            ba_list['FZ'].append(ch_nam)
          elif 'c' in temp and 'p' not in temp:
            ba_list['CZ'].append(ch_nam)
          elif 'p' in temp and 'o' not in temp:
            ba_list['PZ'].append(ch_nam)
          elif 'o' in temp:
            ba_list['OZ'].append(ch_nam)

    ba_list['all'] = 'all'

    psd_dict = {}
    for i in ba_list:
      for j in freq_bands:
        psd_dict[n_sub+'-'+task+'-'+i+'-'+j] = []

    freq_dict = {}
    for i in ba_list:
      for j in freq_bands:
        freq_dict[n_sub+'-'+task+'-'+i+'-'+j] = []

    print(f'Processing task {task} for subject-{n_sub}...')
    # calculate power for each frequency band within each brain region
    for i, key1 in enumerate(ba_list):
        for key2 in freq_bands:
            kwargs = dict(picks=ba_list[key1],
                        fmin=freq_bands[key2][0],
                        fmax=freq_bands[key2][1], n_jobs=n_jobs)
            psds, freq = psd_multitaper(epoch, **kwargs)
            psd_dict[n_sub+'-'+task+'-'+key1+'-'+key2].append(psds)
            freq_dict[n_sub+'-'+task+'-'+key1+'-'+key2].append(freq)

    print('FINISH!')

    psd_total_unaggregated = psd_total_unaggregated | psd_dict
    freq_total_unaggregated = freq_total_unaggregated | freq_dict

    
    psd_dict = {rename_key(k):v for k,v in psd_dict.items()}

    psd_dict_mean = {}
    for key in psd_dict.keys():
        psd_transformed = 10. * np.log10(psd_dict[key][0])
        psd_dict_mean[key] = psd_transformed.mean(0).mean(0).mean(0)

    psd_df = pd.DataFrame(psd_dict_mean, index=[f'sub-{n_sub}_{task}',])
    psd_total = psd_total.append(psd_df)
    # psd_df.reset_index(inplace=True)



# save aggregated data
dir = '/Users/yeganeh/Codes/otka-preprocessing/doc'
fname = op.join(dir, 'psd_total.csv')
psd_total.to_csv(fname)

# save unaggregated data
a_file = open("psd_total_unaggregated.pkl", "wb")
pickle.dump(psd_total_unaggregated, a_file)
a_file.close()

a_file = open("freq_total_unaggregated.pkl", "wb")
pickle.dump(freq_total_unaggregated, a_file)
a_file.close()

  # columns = []
  # col = psd_df.columns.tolist()
  # for i in range(len(col)):
  #     if col[i][-3] == '1':
  #         temp = col[i].replace('-1.0','')
  #         columns.append(temp)

  # index = [1,2,3,4]

  # df_ = pd.DataFrame(index=index, columns=columns)
  # df_ = df_.fillna(0)

  # col_baseline = [col for col in columns if 'baseline' in col]
  # [columns.remove(col) for col in columns if 'baseline' in col]

  # temp = []
  # # sub_ids = df_.index.tolist()

  # for j in range(len(columns)):
  #     for z in range(1,5):
  #         temp.append(psd_df.loc[0,f'{columns[j]}-{z}.0'])
  #     df_.loc[:,columns[j]] = temp
  #     temp = []


  # temp = []
  # for j in range(len(col_baseline)):
  #     temp.append(psd_df.loc[0,f'{col_baseline[j]}-1.0'])
  #     temp = [x for pair in zip(temp,temp,temp,temp) for x in pair]
  #     df_.loc[:,col_baseline[j]] = temp
  #     temp = []

  # base_post = [col+'-post' for col in col_baseline]
  # df_2 = pd.DataFrame(index=index, columns=base_post)
  # temp = []
  # for j in range(len(col_baseline)):
  #     temp.append(psd_df.loc[0,f'{col_baseline[j]}-2.0'])
  #     temp = [x for pair in zip(temp,temp,temp,temp) for x in pair]
  #     df_2.loc[:,base_post[j]] = temp
  #     temp = []

  # df = pd.concat([df_,df_2], axis=1)
  # fname = op.join(data_dir, f'{n_sub}-{task}-psd_features.csv')
  # # df.to_csv(fname)

  # %%

# df_psd, pad_dict = psd(data_dir, ['04'], 'baseline1')

  # deriv_dir = '/Users/yeganeh/Codes/otka-preprocessing/data/PSD/'
  # fname = op.join(data_dir, f'sub-{n_sub}_ses-{n_ses}_task-{task}-psd.csv')
  # epoch = mne.read_epochs(fname)

# %%
