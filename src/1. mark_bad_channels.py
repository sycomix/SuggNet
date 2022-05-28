# Visualize the eeg time series and create a dictionary of bad channels

# imports
import mne
import matplotlib.pyplot as plt
from pathlib import Path

# initiate an empty dictionary to collect the name of bad channels in.
bads = {}

for subject_path in Path('data/raw_test').glob('*.vhdr'):
    subject_id = subject_path.stem

    raw = mne.io.read_raw_brainvision(subject_path, eog=('EOG1', 'EOG2'), misc=['ECG'])
    raw.set_channel_types({'ECG': 'ecg'})

    # raw.set_eeg_reference(ref_channels=['M1', 'M2'])
    # freqs = np.arange(50, 300, 50)
    # raw.notch_filter(freqs=freqs)

    raw.plot(n_channels=61, remove_dc=True, highpass=1, lowpass=35, filtorder=0)
    plt.show()

    bad = raw.info['bads']
    print(bad)
    bads[subject_id] = bad

# a_file = open("bad_channels_chunck2-3.pkl", "wb")
# pickle.dump(bads, a_file)
# a_file.close()
