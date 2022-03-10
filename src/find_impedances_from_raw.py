""" find impendances above threshold (10kohm)"""

import pandas as pd
from mne.io import read_raw_brainvision

ids_map = pd.read_excel('docs/ids_map.xlsx', header=1, index_col='behavioral_id')
ids_map = ids_map.loc[:2132614].drop_duplicates('bids_id')
subject_ids = ids_map.reset_index().eeg_id
bids = ids_map.reset_index().bids_id

high_imp = {}
for sub, bid in zip(subject_ids, bids):
    raw = read_raw_brainvision(f'data/Live Sessions/{sub}.vhdr', misc=['ECG'])
    ch_names = raw.ch_names
    high_imp[bid] = [dict({ch_nam: raw.impedances[ch_nam]['imp']})
                            for ch_nam in ch_names
                            if raw.impedances[ch_nam]['imp'] >= 11]


df = pd.DataFrame.from_dict(high_imp, orient='index')
df.to_csv('docs/high_imp.csv')
