# %%
# # import necessary modules
import numpy as np
import mne
import os.path as op
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftshift
from mne_bids import write_raw_bids, BIDSPath, print_dir_tree, read_raw_bids, mark_bad_channels
from mne_bids.stats import count_events


elig_par = pd.read_csv('elig_par')
elig_par = elig_par['0'].tolist()
elig_par = [i for i in elig_par if i.isdigit()]


#create BIDS dataset 
study_name = 'Main-study'
data_dir = '/Users/yeganeh/Documents/Raw Files/PLB_HYP_OTKA/Live Sessions'
bids_root = f'/Users/yeganeh/Codes/otka-preprocessing/data/{study_name}'


imp_trg = [2152714, 2160311, 215711, 2152014, 2160314, 21052511, 2131811, 2131814] # sessions that trigger has not worked normally (the last one does not have only Stimulus 2, the forth only contains the triggers for the first trial)
[elig_par.remove(str(i)) for i in imp_trg]


#%%
subject_ids = elig_par

ids_map = dict(zip(subject_ids,range(4, len(elig_par)+4)))
iterators = range(10)
runs = [0, 3, 4, 7, 8, 11, 12, 15, 16, 18]
run_map = dict(zip(iterators, runs))

raw_list = list()
bids_list = list()
bads = list()
new_annot = dict(onset = [], duration = [], description  = [])

for subject_id in subject_ids:
    fname = op.join(data_dir,'{}.vhdr'.format(subject_id))
    raw = mne.io.read_raw_brainvision(fname, eog=('EOG1', 'EOG2'), misc= ['ECG'])
    raw.set_channel_types({'ECG':'ecg'})
    raw.info['line_freq'] = 50  # specify power line frequency
    tmin_index = raw.annotations.description.tolist().index('Stimulus/S  2')

    for iterator in iterators:

        #crop
        tmin = raw.annotations.onset[tmin_index + run_map[iterator]]
        tmax = raw.annotations.onset[tmin_index + run_map[iterator] + 1]
        if run_map[iterator] in [0,18]:
            description = f'baseline{int(iterator/9+1)}'
        elif run_map[iterator] % 2:
            description = f'induction{int((run_map[iterator] - 3)/4 + 1)}'
        else:
            description = f'experience{int(run_map[iterator]/4)}'


        raw_segment = raw.copy().crop(tmin=tmin, tmax=tmax)

        #define a more meaningfull annotations
        onset = raw_segment.annotations.onset
        new_annot = mne.Annotations(onset=[0],
                                duration=[onset[1]-onset[0]],
                                description=[description])
            
        raw_segment.set_annotations(new_annot)
        
        #save 
        dir = '/Users/yeganeh/Documents/Raw Files/PLB_HYP_OTKA'
        fname_new = op.join(dir, f'fif_files/{subject_id}-{description}-raw.fif')
        raw_segment.save(fname_new)
        raw_segment = mne.io.read_raw_fif(fname_new)

        #creat lists
        raw_list.append(raw_segment)
        bids_path = BIDSPath(subject=f'{ids_map[subject_id]:02}',
                            session='01',
                            task= description,
                            # run=f'{iterator+1}',
                            root=bids_root)
        bids_list.append(bids_path)
        

#create bids from lists
for raw, bids_path in zip(raw_list, bids_list):
    write_raw_bids(raw, bids_path, overwrite=True)


#%%
# marking bad channels in bids files
# read pickled file of bad_channels
import pickle
a_file = open("bad_channel.pkl", "rb")
bad_channels_dict = pickle.load(a_file)
a_file.close()

# remove the participants we don't have bids file for
bad_channels_dict.pop('2131814')

# get the bids path corresponding to each participants and mark bad channels for them based on their original ids
for key in bad_channels_dict:
    value = ids_map[key] - 4
    temp_list = bids_list[value*10 : (value+1)*10]
    bads = bad_channels_dict[key]

    for bids_path in temp_list:
        raw = read_raw_bids(bids_path=bids_path, verbose=False)
        mark_bad_channels(ch_names=bads, bids_path=bids_path, verbose=False, overwrite=True)


