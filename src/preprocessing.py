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
    ch_names = list(ds.coords['channel'].values)
    # TODO sfreq = ds.attrs['sfreq']
    sfreq = 1000
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

    # initialize empty containers for storing amplitude vectors, rejection threshold, etc.
    ampVectors = []
    ampVectors_after_autoreject = []
    rejects = {}

    for i, sub in enumerate(subjects):
        # mne object
        data = ds[task][i].to_numpy()
        # TODO bads = ast.literal_eval(ds.attrs['bad_channels'][i])
        # info['bads'] = bads
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
        epochs = mne.make_fixed_length_epochs(raw, duration=1, preload=True)
        # note: for creating epochs with mne.epochs, tmin and tmax should be determined!

        # TODO ICA (This section finds blink components automatically and removes them, it is possible that there is no
        # blink component for some participants) (can make a copy of raw, highpass filter it in 1 hz, calculate ICA,
        # and then remove the compoments from the original raw (the same for autoreject!))
        # TODO log everything! the ccomponent it has removed, plots and all the reports, we should become aware of
        # decision it made automaticly.

        # TODO amplitude vector after ica

        # autoreject
        if autoreject == 'global':
            reject = get_rejection_threshold(epochs, decim=2)
            rejects[sub] = reject
            epochs_clean = epochs.copy().drop_bad(reject=reject)
            del epochs

        if autoreject == 'local':
            ar = AutoReject()
            epochs_clean = ar.fit_transform(epochs)

        # epochs to continous data and calculate amplitude vector after epoching
        continuous_data = _epochs_to_continuous(epochs)
        ampVector = amplitude_vector(continuous_data)
        ampVectors_after_autoreject.append(ampVector)

        # save clean epochs
        epochs_clean.save(f'data/clean_data/sub-{sub}_ses-01_task-{task}_epo.fif')

    # calculate dispersion vector from each stage and save the plot
    pos = _make_montage()
    dispersion_with_mean(ampVectors, pos)
    del ampVectors

    if autoreject:
        dispersion_with_mean(ampVectors_after_autoreject, pos)
        del ampVectors_after_autoreject

    # TODO save reject threshold in csv format

    # clean dataset
    # read epochss from file
    # mne.read_epochs()

    # epoch to continuous

    # use to_xarray to create xarray dataset


def _epochs_to_continuous(epochs):
    array = epochs.get_data()
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
