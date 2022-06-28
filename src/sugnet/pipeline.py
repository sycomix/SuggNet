
from pathlib import Path
import pandas as pd
import numpy as np


def _query_csv(path: Path,
               subject_condition: np.ndarray):
    """Query the given csv file for the given subject conditions.
    """

    data = pd.read_csv(path, index_col=0)
    data['bids_id'] = data['bids_id'].apply(int).apply(lambda x: str(x).rjust(2, '0'))
    data = data.query('condition.str.contains("experience")')
    all_idx = data[['bids_id', 'procedure']].agg(''.join, axis=1)
    valid_idx = all_idx.to_frame('idx').query('idx in @subject_condition').index
    data.drop(columns=['hypnosis_depth', 'procedure', 'description', 'session', 'condition',
                       'bids_id'], errors='ignore', inplace=True)
    return data.loc[valid_idx]


def extract_features(subjects: np.ndarray,
                     kind: str,
                     frequency_band=None,
                     data_dir=Path('data/classification_datasets'),
                     power_types='periodic',
                     **kwargs) -> np.ndarray:
    """Extract features from the given subjects.

    Args:
    power_types: in ['periodic', 'nonperiodic' 'iaf', 'all'] effective only when kind is 'power source'.
    """

    subject_condition = pd.DataFrame(subjects).agg(''.join, axis=1).to_list()

    if kind.lower() == 'chance':
        n_features = kwargs.get('n_features', 4)
        X = np.random.rand((len(subjects), n_features))
        return X

    elif kind.lower() == 'power source':
        path = data_dir / 'power_source.csv'
        data = _query_csv(path, subject_condition)
        col_names = data.columns
        # TODO: add statement when frequency_band is 'all' and power_types is 'periodic'
        if frequency_band != 'all' and power_types == 'periodic':
            col_names = [col for col in data.columns if frequency_band in col]
        elif power_types == 'nonperiodic':
            col_names = [col for col in data.columns if 'exponent' in col or 'offset' in col]
        elif power_types == 'iaf':
            col_names = [col for col in data.columns if 'IAF' in col]
        return data.fillna(0)[col_names]

    elif kind.lower() == 'power sensor':
        path = data_dir / 'power_sensor.csv'
        data = _query_csv(path, subject_condition)
        col_names = data.columns
        if frequency_band != 'all':
            col_names = [col for col in data.columns if frequency_band in col]
        return data[col_names]

    elif kind.lower() == 'correlation source':
        path = data_dir / 'correlation_source.csv'
        data = _query_csv(path, subject_condition)
        col_names = data.columns
        if frequency_band != 'all':
            col_names = [col for col in data.columns if frequency_band in col]
        return data[col_names]

    elif kind.lower() == 'correlation sensor':
        path = data_dir / 'correlation_sensor.csv'
        return _query_csv(path, subject_condition)

    elif kind.lower() == 'plv source':
        path = data_dir / 'plv_source.csv'
        data = _query_csv(path, subject_condition)
        col_names = data.columns
        if frequency_band != 'all':
            col_names = [col for col in data.columns if frequency_band in col]
        return data[col_names]

    raise ValueError(f'Unknown feature kind: {kind}')


if __name__ == '__main__':
    # test
    extract_features(np.array([['01', 'whitenoise'],
                               ['01', 'confusion'],
                               ['02', 'confusion'],
                               ['02', 'embedded']]),
                     kind='correlation sensor')
