"""
run ica and detect biophysiological components automatically using template
"""

import numpy as np
import mne
from mne.preprocessing import ICA, read_ica, corrmap
from mne import Report


def run_ica(raw,
            sub,
            n_components=None,
            random_state=42):

    # open templates
    eog_ica, eog_3rd_ica, ecg_ica, eog_inds, ecg_inds = _open_templates()

    # filter before ica
    # raw_filt = raw.copy().filter(1, None)  # if the main filtering highpass is less than 1 hz uncomment this line
    # and change the ica.fit(raw) to ica.fit(raw_filter)

    # apply ica
    ica = ICA(n_components, random_state=random_state)
    ica.fit(raw)

    # create list of icas
    icas_eog = [eog_ica]
    icas_eog.append(ica)
    icas_ecg = [ecg_ica]
    icas_ecg.append(ica)

    # detect eog components
    eog_epochs = mne.preprocessing.create_eog_epochs(raw=raw)
    _, eog_scores = ica.find_bads_eog(raw, ch_name=['EOG1', 'EOG2', 'Fpz'])
    # we know that there is two components in the eog template so:
    [corrmap(icas_eog, template=(0, eog_inds[i]), threshold=0.9, label='blink', plot=False)
     for i in range(2)]
    # detecting oculomotor activity using third compenents
    icas_eog.append(eog_3rd_ica)
    eog_inds.append(7)  # use components 7 from 3rd_ica as a template:
    corrmap(icas_eog, template=(2, eog_inds[2]), threshold=0.85, label='oculomotor', plot=False)

    if 'blink' in ica.labels_.keys():
        eog_comps = ica.labels_['blink']
    else:
        eog_comps = []

    # detect ecg components
    ecg_epochs = mne.preprocessing.create_ecg_epochs(raw=raw)
    _, ecg_scores = ica.find_bads_ecg(raw, ch_name='ECG')
    corrmap(icas_ecg, template=(0, ecg_inds[0]), threshold=0.85, label='heartbeat', plot=False)
    if 'heartbeat' in ica.labels_.keys():
        ecg_comps = ica.labels_['heartbeat']
    else:
        ecg_comps = []

    ica.exclude = eog_comps + ecg_comps
    print(f'EOG AND ECG COMPONENTS: {ica.labels_}')

    # save report and save ica
    report = Report(title=f'{sub}-ICA Report')
    report.add_ica(
        ica=ica,
        title='ICA cleaning',
        inst=raw,
        ecg_evoked=ecg_epochs.average(),
        eog_evoked=eog_epochs.average(),
        ecg_scores=ecg_scores,
        eog_scores=eog_scores,
        n_jobs=-1
    )
    # save reports and icas
    report.save(f'data/ica/reports/{sub}-report_ica.html', open_browser=False, overwrite=True)
    ica.save(f'data/ica/fitted_icas/{sub}_ica.fif', overwrite=True)

    # apply ica
    print(f'Applying ICA for sub-{sub}...')
    ica.apply(raw, exclude=ica.exclude)

    return raw


def _open_templates(fname_eog_ica='data/ica/templates/eog-ica.fif',
                    fname_eog_3rd_ica='data/ica/templates/eog_3rd-ica.fif',
                    fname_ecg_ica='data/ica/templates/ecg-ica.fif',
                    fname_eog_template='data/ica/templates/eog_template.npy',
                    fname_ecg_template='data/ica/templates/ecg_template.npy'
                    ):
    # read ica and template
    eog_ica = read_ica(fname_eog_ica)
    eog_3rd_ica = read_ica(fname_eog_3rd_ica)
    ecg_ica = read_ica(fname_ecg_ica)
    eog_inds = np.load(fname_eog_template, allow_pickle=True)
    eog_inds = eog_inds[0]
    ecg_inds = np.load(fname_ecg_template, allow_pickle=True)
    ecg_inds = ecg_inds[0]

    return eog_ica, eog_3rd_ica, ecg_ica, eog_inds, ecg_inds
