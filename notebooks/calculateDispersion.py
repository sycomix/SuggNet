"""calculate and visualize dispersion vector (using mean for normalization)"""

# Import necessary modules
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne_bids
from pathlib import Path
from pyprep.prep_pipeline import PrepPipeline
from scipy.stats import median_abs_deviation


# transalted from Vislab's MATLAB code to python
def dispersion_with_mean(matrix):
    """
    calculate dispersion vector [1]
    this function use mean for normalization

    Parameters
    ----------
    matrix : array-like
        amplitude matrix. rows are channels and columns are recordings

    Reference
    ---------
    [1] Bigdely-Shamlo, N., Touryan, J., Ojeda, A., Kothe, C., Mullen, T., & Robbins, K. (2020).
    Automated EEG mega-analysis I: Spectral and amplitude characteristics across studies. NeuroImage, 207, 116361.

    """
    mean_amp = matrix.mean(1)
    matrix_normalized = matrix / mean_amp
    stdFromMad = 1.4826 * median_abs_deviation(matrix_normalized, 1)
    dispersion = stdFromMad / np.median(matrix_normalized)

    return dispersion


def _make_amplitude_matrix(ampVectors):
    """
    ampVectors : list of np.array

    """
    # inital zeros array
    amplitude_matrix = [
        np.append(np.zeros(ampVectors[0].shape), ampVectors[i])
        for i in range(len(ampVectors))
    ]
    # delete the intial array
    amplitude_matrix = np.delete(amplitude_matrix, 0, axis=0)
    return amplitude_matrix


def _amplitude_vector(raw):
    # Calculate robust std from MAD
    # if type(raw) !=
    ampVector = 1.4826 * median_abs_deviation(raw.get_data(), 1)
    return ampVector
