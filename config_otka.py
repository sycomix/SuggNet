study_name = 'otka1-bids-sample'
bids_root = f'~/Codes/otka-bids-preprocessing/data/{study_name}'
deriv_root = f'{bids_root}/derivatives/mne-bids-pipeline/{study_name}'

interactive = True
ch_types = ['eeg']
reject = dict(eeg=100e-6)
baseline = (None, 0)

l_freq = 0.1

subjects = ['01']

interpolate_bads_grand_average = False