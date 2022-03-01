"""create xarray dataset"""

# imports
import xarray as xr
import pandas as pd
import numpy as np
import mne_bids


def xarray_creator(path, subjects, tasks, dir, realHyp=True, sfreq=1000):
    """
    path : str
        path to the folder that contains eeg data from different subjects in different tasks

    dir : str
        the path in which save the dataset

    sfreq : int
        sampling frequency rate

    """
    allSubjectsTasks, channels, time = _dict_maker(path, subjects, tasks, realHyp)
    # create xarray from dictionary
    # data variables
    baseline = allSubjectsTasks['baseline']  # TODO find a way to create these without knowing the variable names
    experience = allSubjectsTasks['experience']

    # coordinates
    subjects = subjects
    channels = channels
    time = time

    # attributes
    bad_channels = pd.read_csv('docs/bad_channels_bids.csv', index_col='bids_id')
    bads = [__bad_channel_matcher(bad_channels, sub) for sub in subjects]
    sfreq = sfreq
    # montage = make_montage()

    dataset = xr.Dataset(
        {
            "baseline": (["subject", "channel", "time"], baseline),
            "experience": (["subject", "channel", "time"], experience)
        },
        coords={
            "subject": (["subject"], subjects),
            "channel": (["channel"], channels),
            "time": (["time"], time)
        },
        attrs={
            "bad_channels": bads,
            "sfreq": sfreq
        }

    )

    # save dataset in the given directory
    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in dataset.data_vars}
    dataset.to_netcdf(dir, engine="h5netcdf", encoding=encoding)


def _dict_maker(path, subjects, tasks, realHyp=True):
    """

    open files and create a dictionary from them

    Parameters
    ----------
    path :  str
        path to bids data
        
    subjects : list of str
        subjects' bids id. It can also take up 'all'.

    tasks : list of str
        tasks names. It can also take up 'all'.

    bids_root : str
        path to bids data

    realHyp : boolean
        If True, it read a csv file and will only include the expereince from those trials
        where hypnosis was introduced as hypnosis.

    """
    if subjects == 'all':
        subjects = [f"{num:02d}" for num in range(1, 51)]  # TODO retrieve 51 from number of files in bids_root

    if tasks == 'all':
        tasks = [
            'baseline1',
            'induction1', 'experience1',
            'induction2', 'experience2',
            'induction3', 'experience3',
            'induction4', 'experience4',
            'baseline2'
        ]

    if realHyp:
        realHypInd = pd.read_csv('docs/realHyp.csv', index_col='bids_id')
        tasks = ['baseline1', 'experience']
        allSubjectsTasks = {}

        for task in tasks:

            # timpoints and initialize
            timepoints = 300001  # change this if resampling data
            allSubjects = np.empty((1, 61, timepoints))

            # Open data of a specific task for all subjects one by one, reshape and append them
            for sub in subjects:
                # find hypnosis-as-hypnosis trial index (only for experience)
                if task == 'experience':
                    ind = __task_ind_finder(realHypInd, sub)
                    task = task + str(ind)
                bids_path = mne_bids.BIDSPath(subject=sub, session='01', task=task, root=path)
                raw = mne_bids.read_raw_bids(bids_path, verbose=False)
                # Cut noisy parts only for experience and baseline task
                if task[:-1] in ['baseline', 'experience']:
                    raw = _cut_noisy(raw, task, 'hun')

                # Get eeg data as np.array and add subject dimension
                oneSubject = np.expand_dims(raw.get_data(), 0)

                allSubjects = np.append(allSubjects, oneSubject, axis=0)

            # remove empty array
            allSubjects = np.delete(allSubjects, 0, axis=0)

            # unify task indexes across participants
            task = task[:-1]

            # mega data
            allSubjectsTasks[task] = allSubjects

    subjects = subjects
    channels = raw.ch_names
    _, time = raw.get_data(return_times=True)

    # montage = make_montage()
    return allSubjectsTasks, channels, time


# cut noisy parts
def _cut_noisy(raw, task, language):
    """
    First step of preprocessing in my pipeline!
    it's important to get rid of noisy parts in baseline and experinece segments: informed from the audio files.

    Parameters
    ----------
    raw : mne.io.Raw
        eeg raw data
    task : str
        can be experience1, etc or baseline1 or baseline2
    language : str
        language of the experiment ['eng' or 'hun']
    """
    # validate task and language name
    import re
    from collections import namedtuple

    if not re.fullmatch('(baseline[12])|(experience[1-4])', task):
        raise Exception('Invalid task!')

    tasklang = task + language

    # helper named tuple to make the code more readable
    Interval = namedtuple('Interval', ['tmin', 'tmax'])

    cut_intervals = {
        '(experience[1-4](eng|hun))|baseline1hun': Interval(tmin=20, tmax=320),
        'baseline1eng': Interval(tmin=25, tmax=325),
        'baseline2(eng|hun)': Interval(tmin=30, tmax=330),
        }
    interval = [value for key, value in cut_intervals.items() if re.fullmatch(key, tasklang)][0]

    raw.crop(tmin=interval.tmin, tmax=interval.tmax)

    return raw


# real hypnosis index finder
def __task_ind_finder(realHypInd, sub):
    exp_ind = realHypInd.loc[int(sub), 'true_hyp_ind']
    return exp_ind


# bad channel finder
def __bad_channel_matcher(bad_channels, sub):
    """
    find the name of channels marked as bad for each
    """
    bads = bad_channels.loc[int(sub), 'bad_channels']
    return bads
