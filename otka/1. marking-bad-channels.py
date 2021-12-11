
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
elig_par = pd.read_excel(f'{docs_dir}/eeg_behavioral_bids_idsmap.xlsx', header=1)
elig_par.set_index('eeg_id', inplace=True)
chunck2 = [i for i in elig_par.index if elig_par.loc[i,'bids_id']== '[]']

#%%
subject_ids = ['214911confusion']
bads = {}

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


