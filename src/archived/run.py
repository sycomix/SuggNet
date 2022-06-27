"""run the whole analysis at once"""

# imports
import pandas as pd  # noqa
from preprocessing import preprocessing # noqa
from spectral_analysis import calculatePSD, extract_psds_freatures

# uploads
ids_map = pd.read_excel('docs/ids_map.xlsx', header=1, index_col='behavioral_id')
ids_map = ids_map.loc[:2132614].drop_duplicates('bids_id')
ids_map = ids_map[['bad_channels', 'bids_id', 'language', 'true_hyp_ind']]
ids_map.set_index('bids_id', inplace=True)
ids_map = ids_map.astype({'true_hyp_ind': int})  # change type of true hypnosis column from float to intger

subjects = [f"{num:02d}" for num in range(1, 52)]
tasks = [
    'baseline1', 'induction1', 'induction2', 'induction3', 'induction4',
    'experience1', 'experience2', 'experience3', 'experience4',
    'baseline2'
    ]
freqs = dict(
  delta=(1, 4), theta=(4, 8),
  alpha=(8, 13), beta=(13, 30),
  lower_gamma=(30, 40), broadband=(1,40))

subjects = ['52']
tasks = ['baseline1', 'induction1',
         'induction4','experience1',
         'experience4','baseline2']

# preprocessing
# subjects.remove('02')  # will analyze sub '02' seperately (becuase of the ICA template)
# preprocessing(path='data/BIDS_data',
#               subjects=subjects,
#               tasks=tasks,
#               ref_chs='average',
#               ids_map=ids_map,
#               filter_bounds=(1, 42),
#               ica_n_components=30,
#               ica_report=False)

# calculate psd
# calculatePSD(path='data/clean_data',
#              subjects=subjects,
#              tasks=tasks,
#              freqs=freqs,
#              n_overlap=0.5,
#              n_job=-1
#              )

# calculate extra psd features
extract_psds_freatures()
