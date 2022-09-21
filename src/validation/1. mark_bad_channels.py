# Visualize the eeg time series and create a dictionary of bad channels #

# imports
import mne
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

# initiate an empty dictionary to collect the name of bad channels in.
bads = {}

for subject_path in Path('data/raw/PLB_HYP_Hypnotist_data').glob('*.vhdr'):
    subject_id = subject_path.stem

    raw = mne.io.read_raw_brainvision(subject_path, eog=('EOG1', 'EOG2'), misc=['ECG'])
    raw.set_channel_types({'ECG': 'ecg'})

    raw.plot(n_channels=65, remove_dc=True, highpass=1, lowpass=40, filtorder=0)
    plt.show()

    bad = raw.info['bads']
    print(bad)
    bads[subject_id] = bad

# pickle bad channels
# with open("bad_channels.pkl", "wb") as f:
#     pickle.dump(bads, f)

# # visualize hypnotist's EEG recordings
# path = 'data/raw/PLB_HYP_Hypnotist_data/Relaxation.vhdr'
# raw = mne.io.read_raw_brainvision(path, eog=('EOG1', 'EOG2'), misc=['ECG'])
# raw.set_channel_types({'ECG': 'ecg'})

# raw.plot(n_channels=61, remove_dc=True, highpass=1, lowpass=40, filtorder=0)
# plt.show()
