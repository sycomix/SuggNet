#%%
# this script creates bids files for the data with impaired first baseline annotations
# # import necessary modules
import numpy as np
import mne
import os.path as op
from numpy.core.numeric import ones
import pandas as pd
from mne_bids import write_raw_bids, BIDSPath, print_dir_tree, read_raw_bids, mark_bad_channels
import dateutil.parser


#create BIDS dataset 
study_name = 'Main-study'
data_dir = '/Users/yeganeh/Documents/Raw Files/PLB_HYP_OTKA/Live Sessions'
bids_root = f'/Users/yeganeh/Codes/otka-preprocessing/data/{study_name}'
docs_dir = '/Users/yeganeh/Codes/otka-preprocessing/docs'
bh_dir = '/Users/yeganeh/Codes/otka-preprocessing/behavioral_data'

# map between eeg and bids ids
ids = pd.read_excel(f'{docs_dir}/eeg_behavioral_bids_idsmap.xlsx', header=1)
chunck3  = ids[40:46] #data with impaired first baseline annotations

# map between eeg, behavioral and and bids ids
bh_id = chunck3['behavioral_id']
eeg_id = chunck3['eeg_id']
sub_ids_map = dict(zip(eeg_id, bh_id))
diff = {}  

chunck3 = chunck3[['eeg_id','bids_id']]
subject_ids = chunck3['eeg_id'] #eeg ids
chunck3.set_index('eeg_id', drop=True, inplace=True)
ids = chunck3.to_dict()
ids_map = ids['bids_id']

iterators = range(10)
runs = [0, 2, 3, 6, 7, 10, 11, 14, 15, 18]
run_map = dict(zip(iterators, runs))

raw_list = list()
bids_list = list()

#%%
for subject_id in subject_ids:
    fname = op.join(data_dir,'{}.vhdr'.format(subject_id))
    raw = mne.io.read_raw_brainvision(fname, eog=('EOG1', 'EOG2'), misc= ['ECG'])
    raw.set_channel_types({'ECG':'ecg'})
    raw.info['line_freq'] = 50  # specify power line frequency
    onset = raw.annotations.onset
    desc = raw.annotations.description
    duration = raw.annotations.duration

# estimation of the start of the first baseline from behavioral data
    # open behavioral data
    bhdata = pd.read_csv(f'{bh_dir}/subject-{sub_ids_map[subject_id]}.csv')
    bh_trg1 = dateutil.parser.parse(bhdata.loc[0,'timestamp_trigger_1'])
    bh_baseline_d = dateutil.parser.parse(bhdata.loc[2,'timestamp_baseline_start']) - bh_trg1 #duration between trigger 1 and start of the baseline.
    bh_baseline_d = bh_baseline_d.total_seconds() 
    # this duration should be added to the onset of trigger 1 in the eeg trigger onsets
    index = list(desc).index('Stimulus/S  1')
    eeg_trg1 = onset[index]
    baseline_start = bh_baseline_d + eeg_trg1

# new annotation
    new_onset = np.insert(onset, -19, baseline_start)
    new_duration = np.insert(duration, -19, 0.001)
    new_desc = np.insert(desc, -19, 'Stimulus/S  2')
    new_annot = mne.Annotations(onset = new_onset, duration = new_duration, description  = new_desc)
    raw.set_annotations(new_annot)

    tmax_index = raw.annotations.description.tolist().index('Stimulus/S  9')
    onset_orig = raw.annotations.onset


    for iterator in iterators:

        #crop
        tmax = onset_orig[tmax_index - run_map[iterator]]
        tmin = onset_orig[tmax_index - run_map[iterator] - 1]
        if run_map[iterator] in [0,18]:
            description = f'baseline{int(1/(iterator+1)+1)}'
        elif run_map[iterator] % 2:
            description = f'induction{int((20 - run_map[iterator] - 1) /4)}'
        else:
            description = f'experience{int((20 - run_map[iterator] - 2) /4)}'


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