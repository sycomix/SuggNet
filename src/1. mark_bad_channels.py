# This code visualize the eeg time series and create a dictionary of bad channels
# (the quality of channels will be examined later in the preprocessing as well) 
#%%
import numpy as np
import mne
import os.path as op
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftshift
from mne_bids import write_raw_bids, BIDSPath, print_dir_tree, inspect_dataset, mark_bad_channels
from mne_bids.stats import count_events
import pickle

data_dir = '/Users/yeganeh/Documents/Raw Files/PLB_HYP_OTKA/Live Sessions'
docs_dir = '/Users/yeganeh/Codes/otka-preprocessing/docs'
elig_par = pd.read_excel(f'{docs_dir}/ids_map.xlsx', header=1, index_col='eeg_id')
chunck2 = [i for i in elig_par.index if elig_par.loc[i,'bids_id']== '[]']

#%%
subject_ids = ['2133014_poathynosis']
bads = {}

# n_sub = '51'
# raw_name = f'sub-{n_sub}_ses-01_task-baseline1_eeg.vhdr'
# data_dir = f'/Users/yeganeh/Codes/otka-preprocessing/data/Main-study/sub-{n_sub}/ses-01/eeg/{raw_name}'

for subject_id in subject_ids:
    fname = op.join(data_dir,'{}.vhdr'.format(subject_id))
    raw = mne.io.read_raw_brainvision(fname, eog=('EOG1', 'EOG2'), misc= ['ECG'], preload=True)
    raw.set_channel_types({'ECG':'ecg'})
    # raw.set_eeg_reference(ref_channels=['M1', 'M2'])
    raw.plot(show=False, n_channels=61, remove_dc=True, highpass=1, lowpass=40)
    plt.show()

    bad = raw.info['bads']
    print(bad)
    bads[subject_id] = bad

# a_file = open("bad_channels_chunck2-3.pkl", "wb")
# pickle.dump(bads, a_file)
# a_file.close()

# %%
# read pickled file
# a_file = open("bad_channels.pkl", "rb")
# output = pickle.load(a_file)
# a_file.close()


