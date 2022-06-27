#%%
from pathlib import Path
import mne
import os.path as op
import datetime
from numpy import column_stack
import pandas as pd
import pickle

# ids of eeg files
eeg_dir  = '/Users/yeganeh/Documents/Raw Files/PLB_HYP_OTKA/Live Sessions'
sessions_dir = Path(eeg_dir)
session_files = list(sessions_dir.glob('*.vhdr'))
eeg_ids = [str(session_files[i]).split('/')[7][:-5] for i in range(len(session_files))]
## remove unwanted ids
eeg_ids.remove('1')

## sort eeg ids based on their recording dates
dates = []
for sub_id in eeg_ids:
    raw = mne.io.read_raw_brainvision(op.join(eeg_dir, f'{sub_id}.vhdr'))
    dates.append(raw.info.get('meas_date'))
ids = pd.DataFrame({'ids':eeg_ids, 'dates':dates})
ids.sort_values('dates', inplace=True)
ids.reset_index(inplace=True)


#%%
# open the eeg_ids with edited index
docs_dir = '/Users/yeganeh/Codes/otka-preprocessing/docs'
fname = op.join(docs_dir, 'eeg_ids.xlsx')
eeg_ids_edited = pd.read_excel(fname, header=1)
eeg_ids_edited.drop(columns=['index'], inplace=True)
eeg_ids_edited.rename(columns={'Unnamed: 0':'index'}, inplace=True)
eeg_ids_edited.set_index('index', drop=True, inplace=True)

# open and get ids of behavioral files
bh_data = pd.read_excel(op.join(docs_dir,'data_with_hypnotizability.xlsx'), header=1)
bh_ind_dates = bh_data[['index','datetime']]

# concat these two
ids_map = pd.concat([eeg_ids_edited,bh_ind_dates], axis=1)

# sanity check: compare datetime in two dataset
# all([ids_map.loc[i,'datetime'].split()[0] == ids_map.loc[i,'date'].split()[0] for i in range(len(ids_map.index))])

#%%
# relate them to bids' ids
# open bids id_map
fname = op.join(docs_dir, 'bids_eeg_ids_map')
a_file = open(fname, "rb")
bids_id_map = pickle.load(a_file)
a_file.close()

# create a series of bids_ids_map and concat that to the main dataframe
bids_ids = {}
keys = ids_map.ids.tolist() #ids of all the eeg files
for key in keys:
    bids_ids[key] = []

keys = [int(i) for i in bids_id_map.keys()]
for key in keys:
    bids_ids[key] = bids_id_map[str(key)]

bids_id_sr = pd.Series(bids_ids, name='bids_ids')
temp = ids_map.set_index('ids')
eeg_bh_bids_map = pd.concat([temp, bids_id_sr], axis=1)

# # beautiflize and save
eeg_bh_bids_map.rename(columns={'index':'behavioral_ind'})
eeg_bh_bids_map.drop(columns=['dates', 'datetime'], inplace=True)
eeg_bh_bids_map.to_csv('eeg_bh_bids_ids_map.csv')

