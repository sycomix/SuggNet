#%%
from mne.connectivity import spectral_connectivity
import mne
import pandas as pd
import numpy as np
import os.path as op
from mne.time_frequency import AverageTFR
import matplotlib.pyplot as plt
from mne.viz import circular_layout, plot_connectivity_circle

#%%
data_dir = '/Users/yeganeh/Codes/otka-preprocessing/data/Main-study/derivatives/mne-bids-pipeline'
n_sub = '09'
n_ses = '01'
task = 'baseline1'

# open epoch
epoch_name = f'sub-{n_sub}_ses-{n_ses}_task-{task}_proc-clean_epo.fif'
fname = op.join(data_dir, f'sub-{n_sub}/ses-{n_ses}/eeg/{epoch_name}')
epoch = mne.read_epochs(fname)
epoch.drop_channels(['EOG1','EOG2','ECG','M1','M2'])


# calculate functional connectivity
fmin = (8., 12., 20.)
fmax = (12., 20., 25.)
sfreq = epoch.info['sfreq']
con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
    epoch, method='wpli2_debiased', mode='multitaper', sfreq=sfreq, fmin=fmin,
    fmax=fmax, faverage=True, mt_adaptive=True, n_jobs=7)

# %%
conmat = con[:, :, 0]
fig = plt.figure(num=None, figsize=(16, 16), facecolor='black')
plot_connectivity_circle(conmat, epoch.ch_names, n_lines=300,
                        #  node_angles=node_angles, node_colors=node_colors,
                         title='All-to-All Connectivity left-Auditory '
                         'Condition (PLI)', fig=fig)

# %%
