"""calculate and visualize dispersion vector (using mean for normalization)"""

# Import necessary modules
import mne
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path # noqa
from scipy.stats import median_abs_deviation


# transalted from Vislab's MATLAB code to python
def dispersion_with_mean(ampVectors, pos, fname, visualize=True, save=True):
    """
    calculate dispersion vector [1]
    this function use mean for normalization

    Parameters
    ----------
    ampVectors : list of np.array
    visualize : boolean
    matrix : array-like
        amplitude matrix. rows are channels and columns are recordings

    Reference
    ---------
    [1] Bigdely-Shamlo, N., Touryan, J., Ojeda, A., Kothe, C., Mullen, T., & Robbins, K. (2020).
    Automated EEG mega-analysis I: Spectral and amplitude characteristics across studies. NeuroImage, 207, 116361.

    """

    amplitude_matrix = np.array(ampVectors).T
    mean_recording = amplitude_matrix.mean(0)
    matrix_normalized = amplitude_matrix / mean_recording
    stdFromMad = 1.4826 * median_abs_deviation(matrix_normalized, 1)
    dispersion = stdFromMad / np.median(matrix_normalized)

    if visualize:
        _visualize_dispersion(dispersion, pos)
    
    if save:
        np.save(fname, dispersion)

    return dispersion


def amplitude_vector(raw):
    # Calculate robust std from MAD
    array = raw.get_data()
    ampVector = 1.4826 * median_abs_deviation(array, 1)
    # delete ecg and eog channels as they are not of interest but will impact dispersion vector calculation
    ampVector = np.delete(ampVector, [raw.ch_names.index(i) for i in ['EOG1', 'EOG2', 'ECG']])

    return ampVector


def _visualize_dispersion(dispVector, pos):
    ch_pos = np.array(list(pos.get_positions()['ch_pos'].values()))
    ch_pos = ch_pos[:, 0:2]

    # remove position of EOG1 and EOG2 channels that are not included in the dispersion vector compitation
    ch_pos = np.delete(ch_pos, [pos.ch_names.index(i) for i in ['EOG1', 'EOG2']], 0)

    mne.viz.plot_topomap(dispVector, ch_pos)
    plt.save()