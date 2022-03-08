"""
=============
Preprocessing
=============
"""

# imports
import numpy as np
import copy
import ast  # noqa
import xarray as xr
import mne
from autoreject import AutoReject
from autoreject import get_rejection_threshold
from calculateDispersion import amplitude_vector, dispersion_with_mean
from run_ica import run_ica


def preprocessing(path, subjects, task, resampling_frq=None, ref_chs=None,
                  filter_bounds=None, ica=None, autoreject=None, qc=True, n_job=1):

    """
    Parameters
    ----------
    path : str
        path to xarray dataset

    subjects : list of str or int

    task : str

    template : string
        path to a instance of mne.preprocessing.ica. this ica and its eog and ecg components
        will be used as a template for biophysiological ICs detection

    resampling_frq : int
        resampling frequency

    ref_chs : list of str

    filter_bounds : tuple of float
        first item in tuple is the lower pass-band edge and second item is the upper
        pass-band edge
    
    ica : boolean

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
    pos = _make_montage()

    # initialize empty containers for storing amplitude vectors, rejection threshold, etc.
    ampVectors = []
    ampVectors_after_ica = []
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
        # set channels positions
        raw.set_montage(pos)

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

        # apply ICA, remove eog components based on the template, and save the report
        if ica:
            raw_ica = run_ica(raw)
            del raw

        # calculate amplidute vector after ica
        if qc:
            ampVector = amplitude_vector(raw_ica)
            ampVectors_after_ica.append(ampVector)

        # epoching (note: for creating epochs with mne.epochs, tmin and tmax should be specified!)
        epochs = mne.make_fixed_length_epochs(raw, duration=1, preload=True)

        # autoreject
        if autoreject == 'global':
            reject = get_rejection_threshold(epochs, decim=2)
            rejects[sub] = reject
            epochs.drop_bad(reject=reject)
            reject_log = np.array([
                i for i in range(len(epochs.drop_log)) if epochs.drop_log[i] != ()
                ])
            np.save(f'data/rejectlog/{sub}-{task}.npy', reject_log)

        if autoreject == 'local':
            ar = AutoReject()
            epochs = ar.fit_transform(epochs)  # TODO check if this is ok to save the
            # fit_transformed data to the save object

        # epochs to continous data and calculate amplitude vector after epoching
        if qc:
            continuous_data = _epochs_to_continuous(epochs)
            ampVector = amplitude_vector(continuous_data)
            ampVectors_after_autoreject.append(ampVector)

        # save clean epochs
        epochs.save(f'data/clean_data/sub-{sub}_ses-01_task-{task}_epo.fif')

    # calculate dispersion vector from each stage and save the plot
    dispersion_with_mean(ampVectors, pos, fname='docs/dv.npy', save=True)
    del ampVectors

    if ica and qc:
        dispersion_with_mean(ampVectors_after_ica, pos, fname='docs/dv_after_ica.npy', save=True)
        del ampVectors_after_ica

    if autoreject and qc:
        dispersion_with_mean(ampVectors_after_autoreject, pos, fname='docs/dv_after_autoreject.npy', save=True)
        del ampVectors_after_autoreject

    # clean dataset
    # read epochss from file
    # mne.read_epochs()

    # epoch to continuous (maybe I won't do this, depending on how I calculate psd, bacuase near the rejected epochs
    # I would see edge effects and if I retuen this to continuos data I cannot track them!)

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
