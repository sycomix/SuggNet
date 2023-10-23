'''
=======================
Functional Connectivity
=======================
'''

import numpy as np
import mne
import os.path as op
from mne.preprocessing import compute_current_source_density
from mne_connectivity import spectral_connectivity


def calculate_connectivity(path,
                           subjects,
                           tasks,
                           fmin,
                           fmax,
                           method,
                           laplacian=True,
                           n_jobs=-1
                           ):
    
    con_dict = {}
    for n_sub in subjects:
        for task in tasks:
            epo_name = f'sub-{n_sub}_ses-01_task-{task}_proc-clean_epo.fif'
            dir = op.join(path, epo_name)
            # open clean epochs
            epochs = mne.read_epochs(dir)

            # pick eeg channels
            epochs = epochs.pick_types(eeg=True)

            # insert FCz position
            pos = insert_fcz_pos(epochs)
            epochs.set_montage(pos)

            # apply surface Laplacian 
            if laplacian:
                epochs_csd = compute_current_source_density(epochs)
                epochs = epochs_csd
                del epochs_csd

            # calculate connectivity
            sfreq = epochs.info['sfreq']
            con = spectral_connectivity(
                epochs, method=method, sfreq=sfreq, fmin=fmin,
                fmax=fmax, faverage=True, mt_adaptive=True, n_jobs=n_jobs, verbose=0)

            con_dict[f'{n_sub}_{task}'] = con


def insert_fcz_pos(epochs):
    '''
    add FCz position based on channels CPz
    (Their positions are the same, only y-axis value is different)
    '''
    import copy
    ch_names = copy.deepcopy(epochs.info['ch_names'])
    [ch_names.remove(i) for i in ['ECG', 'EOG1', 'EOG2']]
    pos_array = epochs._get_channel_positions()

    pos_fcz = pos_array[ch_names.index('CPz')] * np.array([1, -1, 1])
    pos_array = np.insert(pos_array, 58, pos_fcz, axis=0)
    pos_array = np.delete(pos_array, -1, axis=0)

    pos_dict = dict(zip(ch_names, pos_array))
    return mne.channels.make_dig_montage(pos_dict)
