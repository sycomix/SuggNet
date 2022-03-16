"""
=============
Preprocessing
=============
"""

# imports
import numpy as np
import pandas as pd
import copy
import ast  # noqa
import xarray as xr
import mne
from autoreject import AutoReject
from autoreject import get_rejection_threshold
from calculateDispersion import amplitude_vector, dispersion_report
from run_ica import run_ica


def preprocessing(path,
                  subjects,
                  tasks,
                  resampling_frq=None,
                  ref_chs=None,
                  filter_bounds=None,
                  ica_n_components=None,
                  autoreject='global',
                  ica_report=True,
                  n_job=-1):

    """
    Parameters
    ----------
    path : str
        path to xarray dataset

    subjects : list of str or int

    task : str

    fname_ica_template : string
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
    
    ica_report : boolean
        whether output a report on ica components removal or not

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
    sfreq = ds.attrs['sfreq']
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    pos = _make_montage()

    # initialize empty containers for storing amplitude vectors, rejection threshold, etc.
    ampVectors = []
    ampVectors_after_ica = []
    ampVectors_after_autoreject = []
    rejects = {}

    for task in tasks:
        for i, sub in enumerate(subjects):
            # mne object
            i = int(sub) - 1  # relate sub and i then retrieve array form ds
            data = ds[task][i].to_numpy()
            bads = ast.literal_eval(ds.attrs['bad_channels'][i])
            info['bads'] = bads
            if sub == '02':
                info['bads'] = []  # TODO this is only a quick solution, I should mark this in the dataset
            raw = mne.io.RawArray(data, info)
            raw.set_channel_types({
                'ECG': 'ecg',
                'EOG1': 'eog',
                'EOG2': 'eog'
                })
            print(f'############sub-{sub}-{task}####################')
            # set channels positions
            raw.set_montage(pos)

            # interpolate bad channels
            if raw.info['bads'] != []:
                raw.interpolate_bads()

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
            ampVector = amplitude_vector(raw, ch_names=ch_names)
            ampVectors.append(ampVector)

            # apply ICA, remove eog and ecg ICs using templates, and save the report
            raw = run_ica(raw, sub, task, n_components=ica_n_components, report=ica_report)

            # calculate amplidute vector after ica
            ampVector = amplitude_vector(raw, ch_names=ch_names)
            ampVectors_after_ica.append(ampVector)

            # epoching (note: for creating epochs with mne.epochs, tmin and tmax should be specified!)
            epochs = mne.make_fixed_length_epochs(raw, duration=1, preload=True)
            del raw

            # autoreject
            if autoreject == 'global':
                # pick only eeg channels for getting rejection threshold:
                reject = get_rejection_threshold(epochs.copy().pick_types(eeg=True))
                rejects[sub] = reject
                epochs.drop_bad(reject=reject)
                # reject_log = np.array([
                #         i for i in range(len(epochs.drop_log)) if epochs.drop_log[i] != ()
                #         ])
                # np.save(f'data/rejectlog/{sub}-{task}.npy', reject_log)

            if autoreject == 'local':
                ar = AutoReject()  # TODO consider setting random state
                epochs = ar.fit_transform(epochs)  # TODO check if this is ok to save the
                # fit_transformed data to the save object

            # epochs to continous data and calculate amplitude vector after epoching
            continuous_data = _epochs_to_continuous(epochs)
            ampVector = amplitude_vector(continuous_data, ch_names=ch_names, thisIsNotNumpy=False)
            ampVectors_after_autoreject.append(ampVector)

            # save clean epochs
            epochs.save(f'data/clean_data/sub-{sub}_ses-01_task-{task}_epo.fif', overwrite=True)
            # just for safety!! should be deleted later!!
            ampVectors_collection = [ampVectors, ampVectors_after_ica, ampVectors_after_autoreject]
            np.save('docs/dispersionVectors.npy', np.array(ampVectors_collection))

    # calculate dispersion vector from each stage and save a report
    # first concatenate DVs from each stage and then input it into the custom function
    ampVectors_collection = [ampVectors, ampVectors_after_ica, ampVectors_after_autoreject]
    # calculate unnormalized and normalized DVs with mean for each DV in the collection,
    # make and save a report
    dispersion_report(ampVectors_collection,
                      pos,
                      fname='docs/dispersionVectors.npy',
                      fname_report='DV_report.html',
                      save=True,
                      normalization=False)
    dispersion_report(ampVectors_collection,
                      pos,
                      fname='docs/dispersionVectors_normalized.npy',
                      fname_report='DV-normalized_report.html',
                      save=True,
                      normalization=True)

    # cleans-up and savings
    df = pd.DataFrame.from_dict(rejects)
    df.to_csv('data/rejectlog/rejection_thresholds.csv')
    del ampVectors_collection, ampVectors, ampVectors_after_ica, ampVectors_after_autoreject

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
    raw.cut(1, 10)  # we cut a small segment of the data to speed up when loading data in next lines
    raw.set_channel_types({'ECG': 'ecg'})
    raw.add_reference_channels('FCz')  # FCz was used as online reference
    ch_names = copy.deepcopy(raw.info['ch_names'])  # make a deep copy of the lists of the
    # channel names otherwise ecg channel will be removed in the raw object!!
    ch_names.remove('ECG')

    pos_array = raw._get_channel_positions()

    # add FCz position based on channels CPz (Their positions are the same, only y-axis value is different)
    pos_fcz = pos_array[ch_names.index('CPz')] * np.array([1, -1, 1])
    pos_array = np.insert(pos_array, 60, pos_fcz, axis=0)
    pos_array = np.delete(pos_array, -1, axis=0)

    pos_dict = dict(zip(ch_names, pos_array))
    pos = mne.channels.make_dig_montage(pos_dict)

    return pos
