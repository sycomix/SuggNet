"""Preprocessing"""

# imports
import mne
import pandas as pd  # noqa
import numpy as np  # noqa
import matplotlib.pyplot as plt  # noqa
from autoreject import AutoReject
from autoreject import get_rejection_threshold
import xarray as xr
import ast
from calculateDispersion import _amplitude_vector


def preprocessing(path, subjects, task, resampling_frq=None, ref_chs=None,
                  filter_bounds=None, autoreject=None, qc=True, n_job=1):

    """
    Parameters
    ----------
    path : str
        path to xarray dataset

    subjects : list of str or int

    task : str

    resampling_frq : int
        resampling frequency

    ref_chs : list of str

    filter_bounds : tuple of float
        first item in tuple is the lower pass-band edge and second item is the upper
        pass-band edge

    autoreject : str
        can be 'global' or 'local'. if none, it won't calculate rejection threshold to
        remove bad epochs.

    Returns
    -------
    xarray dataset of preprocessed data

    """

# open dataset and read it into a mne raw object.
    # open dataset
    ds = xr.open_dataset(path)
    # create mne object from dataset
    # info object
    ch_names = list(ds.coords['channels'].values)
    sfreq = ds.attrs['sfreq']
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

    # initialize empty lists for storing amplitude vectors, rejection threshold before for loop
    ampVectors = []
    ampVectors_after_autoreject = []
    rejects = {}

    for i, sub in enumerate(subjects):
        # mne object
        data = ds[task][i].to_numpy()
        bads = ast.literal_eval(ds.attrs['bad_channels'][i])
        info['bads'] = bads
        raw = mne.io.RawArray(data, info)
        raw.set_channel_types({
            'ECG': 'ecg',
            'EOG1': 'eog',
            'EOG2': 'eog'
            })

        # resampling
        if resampling_frq is not None:
            raw.resample(resampling_frq)

        # rereferencing
        if ref_chs is not None:
            # raw.add_reference_channels(ref_channels = 'FCz') # for adding the reference channel to the data
            raw.set_eeg_reference(ref_channels=ref_chs)

        # filtering
        if filter_bounds is not None:
            raw.filter(
                l_freq=filter_bounds[0],
                h_freq=filter_bounds[1],
                n_jobs=n_job
                )

        # create amplitude vector before epoching and running autoreject
        if qc:
            ampVector = _amplitude_vector(raw)
            ampVectors.append(ampVector)

        # epoching
        new_annot = mne.Annotations(
            onset=[0],
            duration=[int(len(raw)/sfreq)],
            description=[task])
        raw.set_annotations(new_annot)
        events, event_dict = mne.events_from_annotations(raw, chunk_duration=1)
        epoch = mne.Epochs(raw, events, event_dict, baseline=None, preload=True)

        # autoreject
        if autoreject == 'global':
            reject = get_rejection_threshold(epoch, decim=2)
            rejects[sub] = reject
            epoch.drop_bad(reject=reject)

        if autoreject == 'local':
            ar = AutoReject()
            epochs_clean = ar.fit_transform(epoch)

        # epoch to continous data and amplitude vector after epoching
        continuous_data = _epoch_to_continuous(epoch)
        ampVector = _amplitude_vector(continuous_data)
        ampVectors_after_autoreject.append(ampVector)

        # TODO ICA

        # TODO amplitude vector after ica

    # TODO calculate dispersion vector from each stage and save the plot

    # TODO save clean dataset


# datasets = []
# for example in examples:
#     ds = create_an_xarray_dataset(example)
#     datasets.append(ds)
# combined = xarray.concat(datasets, dim='example')

def _epoch_to_continuous(epoch):
    array = epoch.get_data()

    return array
