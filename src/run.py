"""run the whole analysis at once"""

# imports
import pandas as pd
from xarray_creator import xarray_creator

# uploads
ids_map = pd.read_excel('docs/ids_map.xlsx', header=1, index_col='behavioral_id')
ids_map = ids_map.loc[:2132614].drop_duplicates('bids_id')
ids_map = ids_map[['bad_channels','bids_id', 'language', 'true_hyp_ind']]
ids_map.set_index('bids_id', inplace=True)
ids_map = ids_map.astype({'true_hyp_ind':int})  # change type of true hypnosis column from float to intger

# create xarray and save it
xarray_creator('data/BIDS_data', tasks='all', subjects=['03', '04'], dir='data/dataset/ds1.nc',
               ids_map=ids_map)
