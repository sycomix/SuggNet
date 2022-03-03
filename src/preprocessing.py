"""
=============
Preprocessing
=============
"""

# imports
import mne
import numpy as np
from autoreject import AutoReject
from autoreject import get_rejection_threshold
import xarray as xr
import ast
from calculateDispersion import amplitude_vector, dispersion_with_mean
import copy


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

    # initialize empty containers for storing amplitude vectors, rejection threshold, etc.
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
            ampVector = amplitude_vector(raw)
            ampVectors.append(ampVector)

        # epoching
        new_annot = mne.Annotations(
            onset=[0],
            duration=[int(len(raw)/sfreq)],
            description=[task])
        raw.set_annotations(new_annot)
        events, event_dict = mne.events_from_annotations(raw, chunk_duration=1)
        epoch = mne.Epochs(raw, events, event_dict, baseline=None, preload=True)

        # autoreject (TODO in mne-bids-pipeline they have this step after ICA)
        if autoreject == 'global':
            reject = get_rejection_threshold(epoch, decim=2)
            rejects[sub] = reject
            epoch.drop_bad(reject=reject)

        if autoreject == 'local':
            ar = AutoReject()
            epochs_clean = ar.fit_transform(epoch)  # noqa

        # epoch to continous data and calculate amplitude vector after epoching
        continuous_data = _epoch_to_continuous(epoch)
        ampVector = amplitude_vector(continuous_data)
        ampVectors_after_autoreject.append(ampVector)

        # TODO ICA

        # TODO amplitude vector after ica

        # save clean epochs
        epoch.save(f'data/clean_data/sub-{sub}_ses-01_task-{task}_epo.fif')

    # calculate dispersion vector from each stage and save the plot
    pos = _make_montage()
    dispersion_with_mean(ampVector, pos)
    del ampVectors
    dispersion_with_mean(ampVectors_after_autoreject, pos)
    del ampVectors_after_autoreject
    
    # TODO save reject threshold in csv format

    # clean dataset
    # read epoch from file
    mne.read_epochs()

    # epoch to continuous

    # use utility function to_xarray to create xarray dataset


def _epoch_to_continuous(epoch):
    array = epoch.get_data()
    shape = array.shape
    init = np.zeros(shape[1:])
    for i in range(shape[0]):
        init = np.hstack((init, array[i]))

    init = np.delete(init, np.s_[:shape[2]], 1)

    return init


def _make_montage(path='data/raw/plb-hyp-live2131111.vhdr'):
    """
    Create a montage from barin vision raw data

    Parameters
    ----------
    path : str
        path to barinvision data header file

    """
    raw = mne.io.read_raw_brainvision(path, verbose=False, misc=['ECG'])
    raw.set_channel_types({'ECG': 'ecg'})
    ch_names = copy.deepcopy(raw.info['ch_names'])  # make a deep copy of the lists of the
    # channel names otherwise ecg channel will be removed in the raw object!!

    ch_names.remove('ECG')
    pos_array = raw._get_channel_positions()
    pos_dict = dict(zip(ch_names, pos_array))
    pos = mne.channels.make_dig_montage(pos_dict)

    return pos
