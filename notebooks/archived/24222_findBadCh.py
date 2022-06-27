# This script find bad channels using a combination of different criteria including:
# channel impedance
# hfnoise and deviation criteria fro prep pipeline and maybe topomap
# and visual detection at the end

# from preprocessing notebook (bad channels detection section):
### Bad channels detections
# As it was explained in the "topographical map" video, here we want to use topographical maps to validate the detected bad channels. (this is only for deciding on the channels that I was not sure if I should mark them as bad or not)
# Rightfully detecting bad channels is very important, because this will affect re-referencig and autoreject threshold. 
# to-do: 1.we can use Shamlo's algorithm for detecting bad channels. 2.set a colorbad with specific vmin and vmax.

############# pyprep part:
subject='03'
session = '01'
# tasks = [ 
#     'induction1', 'experience1', 
#     'induction2', 'experience2',
#     'induction3', 'experience3',
#     'induction4', 'experience4',
#     'baseline2'
# ]
tasks = ['baseline1']

bids_root = Path('data/Main-study')

# open baseline1
bids_path = mne_bids.BIDSPath(subject=subject, session=session, task='baseline1', root=bids_root)
raw = mne_bids.read_raw_bids(bids_path, extra_params={'preload':True}, verbose=False)

# for task in tasks:
#     bids_path = mne_bids.BIDSPath(subject=subject, session=session, task=task, root=bids_root)
#     raw_temp = mne_bids.read_raw_bids(bids_path, verbose=False)
#     raw.append(raw_temp)


# set montage
pos = make_montage()
raw.set_montage(pos)
pos = raw.get_montage()

prep_params = {
    "ref_chs":"eeg",
    "reref_chs":"eeg",
    "line_freqs":np.arange(50, 1000 / 2, 50),
}

prep = PrepPipeline(raw, prep_params, pos)
prep.fit()
prep.noisy_channels_original

#from pyprep.find_noisy_channels import NoisyChannels
# noisy = NoisyChannels(raw)
# bads = noisy.find_all_bads()
# print(bads)

