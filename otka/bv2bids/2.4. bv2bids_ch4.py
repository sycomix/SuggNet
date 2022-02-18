# this is the code for creating bids file from data with bathroom beak or software crash!
# we will create the bids file for each of these participants manually.
#%%
# import necessary modules
import numpy as np
import mne
import os.path as op
from numpy.core.numeric import ones
import pandas as pd
from mne_bids import write_raw_bids, BIDSPath, print_dir_tree, read_raw_bids, mark_bad_channels

# directories
study_name = 'Main-study'
data_dir = '/Users/yeganeh/Documents/Raw Files/PLB_HYP_OTKA/Live Sessions'
bids_root = f'/Users/yeganeh/Codes/otka-preprocessing/data/{study_name}'
docs_dir = '/Users/yeganeh/Codes/otka-preprocessing/docs'
bh_dir = '/Users/yeganeh/Codes/otka-preprocessing/behavioral_data'
fif_dir = '/Users/yeganeh/Documents/Raw Files/PLB_HYP_OTKA'

raw_list = list()
bids_list = list()
#%%
#################################### bids 47 ######################################
subject_ids = ['plb-hyp-live213914', 'plb-hyp-live2139142']
bids_id = '47'
raw_list = list()
bids_list = list()

# the first file contains only the recording from the first baseline
fname = op.join(data_dir,f'{subject_ids[0]}.vhdr')
raw_baseline1 = mne.io.read_raw_brainvision(fname, eog=('EOG1', 'EOG2'), misc= ['ECG'])
raw_baseline1.set_channel_types({'ECG':'ecg'})
raw_baseline1.info['line_freq'] = 50  # specify power line frequency
onset = raw_baseline1.annotations.onset

tmin_index = raw_baseline1.annotations.description.tolist().index('Stimulus/S  2')
tmin = onset[tmin_index]
tmax = onset[tmin_index+1]
description = 'baseline1'

raw_segment = raw_baseline1.copy().crop(tmin=tmin, tmax=tmax)
onset = raw_segment.annotations.onset
new_annot = mne.Annotations(onset=[0],
                        duration=[onset[1]-onset[0]],
                        description=[description])
    
raw_segment.set_annotations(new_annot)

## save 
fname_new = op.join(fif_dir, f'fif_files/{subject_ids[0]}-{description}-raw.fif')
raw_segment.save(fname_new)
raw_segment = mne.io.read_raw_fif(fname_new)

## creat lists
raw_list.append(raw_segment)
bids_path = BIDSPath(subject=bids_id,
                    session='01',
                    task= description,
                    # run=f'{iterator+1}',
                    root=bids_root)
bids_list.append(bids_path)

# open the second segment which contains the rest of recordings of the session
fname = op.join(data_dir,f'{subject_ids[1]}.vhdr')
raw = mne.io.read_raw_brainvision(fname, eog=('EOG1', 'EOG2'), misc= ['ECG'])
raw.set_channel_types({'ECG':'ecg'})
raw.info['line_freq'] = 50  # specify power line frequency
onset_orig = raw.annotations.onset

tmax_index = -2

iterators = range(9)
runs = [0, 2, 3, 6, 7, 10, 11, 14, 15]
run_map = dict(zip(iterators, runs))

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
    fname_new = op.join(fif_dir, f'fif_files/{subject_ids[1]}-{description}-raw.fif')
    raw_segment.save(fname_new)
    raw_segment = mne.io.read_raw_fif(fname_new)

    #creat lists
    raw_list.append(raw_segment)
    bids_path = BIDSPath(subject=bids_id,
                        session='01',
                        task= description,
                        # run=f'{iterator+1}',
                        root=bids_root)
    bids_list.append(bids_path)

#################################### bids 48 ######################################
subject_ids = ['2131611','2131611_baseline_posthyp']
bids_id = '48'

# the first segment contains all the recordings except the the posthypnotic baseline
fname = op.join(data_dir,f'{subject_ids[0]}.vhdr')
raw = mne.io.read_raw_brainvision(fname, eog=('EOG1', 'EOG2'), misc= ['ECG'])
raw.set_channel_types({'ECG':'ecg'})
raw.info['line_freq'] = 50  # specify power line frequency
onset_orig = raw.annotations.onset

iterators = range(9)
runs = [0, 3, 4, 7, 8, 11, 12, 15, 16]
run_map = dict(zip(iterators, runs))

tmin_index = raw.annotations.description.tolist().index('Stimulus/S  2')

for iterator in iterators:

    #crop
    tmin = onset_orig[tmin_index + run_map[iterator]]
    tmax = onset_orig[tmin_index + run_map[iterator] + 1]
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
    fname_new = op.join(fif_dir, f'fif_files/{subject_ids[0]}-{description}-raw.fif')
    raw_segment.save(fname_new)
    raw_segment = mne.io.read_raw_fif(fname_new)

    #creat lists
    raw_list.append(raw_segment)
    bids_path = BIDSPath(subject=bids_id,
                        session='01',
                        task= description,
                        # run=f'{iterator+1}',
                        root=bids_root)
    bids_list.append(bids_path)

# open files of the posthypnotic baseline recording
fname = op.join(data_dir,f'{subject_ids[1]}.vhdr')
raw = mne.io.read_raw_brainvision(fname, eog=('EOG1', 'EOG2'), misc= ['ECG'])
raw.set_channel_types({'ECG':'ecg'})
raw.info['line_freq'] = 50  # specify power line frequency

# the begining and the end of the baseline is not marked with triggers and inspected manually instead
tmin = 39
tmax = 342
description = 'baseline2'

raw_segment = raw.copy().crop(tmin=tmin, tmax=tmax)
new_annot = mne.Annotations(onset=[0],
                        duration=[342-39],
                        description=[description])
    
raw_segment.set_annotations(new_annot)

## save 
fname_new = op.join(fif_dir, f'fif_files/{subject_ids[0]}-{description}-raw.fif')
raw_segment.save(fname_new)
raw_segment = mne.io.read_raw_fif(fname_new)

## creat lists
raw_list.append(raw_segment)
bids_path = BIDSPath(subject=bids_id,
                    session='01',
                    task= description,
                    # run=f'{iterator+1}',
                    root=bids_root)
bids_list.append(bids_path)


#################################### bids 49 ######################################
subject_ids = ['2152014','2152014-2']
bids_id = '49'

# the first file includes the first baseline and the first trial
fname = op.join(data_dir,f'{subject_ids[0]}.vhdr')
raw = mne.io.read_raw_brainvision(fname, eog=('EOG1', 'EOG2'), misc= ['ECG'])
raw.set_channel_types({'ECG':'ecg'})
raw.info['line_freq'] = 50  # specify power line frequency
onset_orig = raw.annotations.onset

iterators = range(3)
runs = [0, 3, 4]
# , 7, 8, 11, 12, 15, 16
run_map = dict(zip(iterators, runs))

tmin_index = raw.annotations.description.tolist().index('Stimulus/S  2')

for iterator in iterators:

    #crop
    tmin = onset_orig[tmin_index + run_map[iterator]]
    tmax = onset_orig[tmin_index + run_map[iterator] + 1]
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
    fname_new = op.join(fif_dir, f'fif_files/{subject_ids[0]}-{description}-raw.fif')
    raw_segment.save(fname_new)
    raw_segment = mne.io.read_raw_fif(fname_new)

    #creat lists
    raw_list.append(raw_segment)
    bids_path = BIDSPath(subject=bids_id,
                        session='01',
                        task= description,
                        # run=f'{iterator+1}',
                        root=bids_root)
    bids_list.append(bids_path)

# the second file includes recordings from the rest of experimental trials and the post hypnotic baseline.
fname = op.join(data_dir,f'{subject_ids[1]}.vhdr')
raw = mne.io.read_raw_brainvision(fname, eog=('EOG1', 'EOG2'), misc= ['ECG'])
raw.set_channel_types({'ECG':'ecg'})
raw.info['line_freq'] = 50  # specify power line frequency
onset_orig = raw.annotations.onset

iterators = range(7)
runs = [0, 2, 3, 6, 7, 10, 11]
run_map = dict(zip(iterators, runs))

tmax_index = raw.annotations.description.tolist().index('Stimulus/S  9')
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
    fname_new = op.join(dir, f'fif_files/{subject_ids[0]}-{description}-raw.fif')
    raw_segment.save(fname_new)
    raw_segment = mne.io.read_raw_fif(fname_new)

    #creat lists
    raw_list.append(raw_segment)
    bids_path = BIDSPath(subject=bids_id,
                        session='01',
                        task= description,
                        # run=f'{iterator+1}',
                        root=bids_root)
    bids_list.append(bids_path)

#################################### bids 50 ######################################
# there was a software crash in this session; participants listen to one of the experimental trials twice and here we remove her first listening.
subject_id = '2132614'
bids_id = '50'

# open data
fname = op.join(data_dir,f'{subject_id}.vhdr')
raw = mne.io.read_raw_brainvision(fname, eog=('EOG1', 'EOG2'), misc= ['ECG'])
raw.set_channel_types({'ECG':'ecg'})
raw.info['line_freq'] = 50  # specify power line frequency

new_onset = np.delete(raw.annotations.onset, [4,5,6,7,8])
new_desc = np.delete(raw.annotations.description, [4,5,6,7,8])
new_duration = np.delete(raw.annotations.duration, [4,5,6,7,8])

new_annot = mne.Annotations(onset=new_onset, duration=new_duration, description=new_desc)
raw.set_annotations(new_annot)

iterators = range(10)
runs = [0, 3, 4, 7, 8, 11, 12, 15, 16, 18]
run_map = dict(zip(iterators, runs))

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
    bids_path = BIDSPath(subject=bids_id,
                        session='01',
                        task= description,
                        # run=f'{iterator+1}',
                        root=bids_root)
    bids_list.append(bids_path)

#create bids from lists
for raw, bids_path in zip(raw_list, bids_list):
    write_raw_bids(raw, bids_path, overwrite=True)

#################################### bids 51 ######################################
# there was a software crash in this session, each experimental trial is recorded separately without any triggers