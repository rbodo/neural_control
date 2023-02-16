"""
Data contains 39 sessions from 10 mice. Time bins for all measurements are
10 ms, starting 500 ms before stimulus onset. The mouse had to determine which
side has the highest contrast. Each mouse dataset contains the fields below.
For extra variables, check out the extra notebook and extra data files
(lfp, waveforms and exact spike times, non-binned).

* `dat['mouse_name']`: mouse name
* `dat['date_exp']`: when a session was performed
* `dat['spks']`: neurons by trials by time bins.
* `dat['brain_area']`: brain area for each neuron recorded.
* `dat['ccf']`: Allen Institute brain atlas coordinates for each neuron.
* `dat['ccf_axes']`: axes names for the Allen CCF.
* `dat['contrast_right']`: contrast level for the right stimulus, which is
    always contralateral to the recorded brain areas.
* `dat['contrast_left']`: contrast level for left stimulus.
* `dat['gocue']`: when the go cue sound was played.
* `dat['response_time']`: when the response was registered, which has to be
    after the go cue. The mouse can turn the wheel before the go cue (and
    nearly always does!), but the stimulus on the screen won't move before the
    go cue.
* `dat['response']`: which side the response was (`-1`, `0`, `1`). When the
    right-side stimulus had higher contrast, the correct choice was `-1`. `0`
    is a no go response.
* `dat['feedback_time']`: when feedback was provided.
* `dat['feedback_type']`: if the feedback was positive (`+1`, reward) or
    negative (`-1`, white noise burst).
* `dat['wheel']`: turning speed of the wheel that the mice uses to make a
    response, sampled at `10 ms`.
* `dat['pupil']`: pupil area  (noisy, because pupil is very small) + pupil
    horizontal and vertical position.
* `dat['face']`: average face motion energy from a video camera.
* `dat['licks']`: lick detections, 0 or 1.
* `dat['trough_to_peak']`: measures the width of the action potential waveform
    for each neuron. Widths `<=10` samples are "putative fast spiking neurons".
* `dat['%X%_passive']`: same as above for `X` = {`spks`, `pupil`, `wheel`,
    `contrast_left`, `contrast_right`} but for  passive trials at the end of
    the recording when the mouse was no longer engaged and stopped making
    responses.
* `dat['prev_reward']`: time of the feedback (reward/white noise) on the
    previous trial in relation to the current stimulus time.
* `dat['reaction_time']`: ntrials by 2. First column: reaction time computed
    from the wheel movement as the first sample above `5` ticks / 10 ms bin.
    Second column: direction of the wheel movement (`0` = no move detected).

The original dataset is here:
https://figshare.com/articles/dataset/Dataset_from_Steinmetz_et_al_2019/9598406
"""
import sys
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

AREAS = {
    'visual cortex': ['VISa', 'VISam', 'VISl', 'VISp', 'VISpm', 'VISrl'],
    'thalamus': ['CL', 'LD', 'LGd', 'LH', 'LP', 'MD', 'MG', 'PO', 'POL', 'PT',
                 'RT', 'SPF', 'TH', 'VAL', 'VPL', 'VPM'],
    'hippocampus': ['CA', 'CA1', 'CA2', 'CA3', 'DG', 'SUB', 'POST'],
    'other cortex': ['ACA', 'AUD', 'COA', 'DP', 'ILA', 'MOp', 'MOs', 'OLF',
                     'ORB', 'ORBm', 'PIR', 'PL', 'SSp', 'SSs', 'RSP', 'TT'],
    'midbrain': ['APN', 'IC', 'MB', 'MRN', 'NB', 'PAG', 'RN', 'SCs', 'SCm',
                 'SCig', 'SCsg', 'ZI'],
    'basal ganglia': ['ACB', 'CP', 'GPe', 'LS', 'LSc', 'LSr', 'MS', 'OT',
                      'SNr', 'SI'],
    'cortical subplate': ['BLA', 'BMA', 'EP', 'EPd', 'MEA']
}


class SteinmetzData:
    def __init__(self, path_data: str):
        self.data_all = np.load(path_data, allow_pickle=True)['arr_0']
        self.data_subject = None
        self.dt = None
        self.num_timesteps = None

    def filter_subject(self, index: int) -> np.ndarray:
        self.data_subject = self.data_all[index]
        self.dt = self.data_subject['bin_size']
        self.num_timesteps = self.spikes.shape[-1]
        return self.data_subject

    def filter_spikes(self, area_condition: Optional[np.ndarray] = None,
                      trial_condition: Optional[np.ndarray] = None
                      ) -> np.ndarray:
        if area_condition is None:
            area_condition = slice(None)
        if trial_condition is None:
            trial_condition = slice(None)
        return self.spikes[area_condition][:, trial_condition]

    @property
    def spikes(self):
        """
        Binary spike array with shape [num_neurons, num_trials, num_timesteps].
        """
        return self.data_subject['spks']

    @property
    def response(self):
        return self.data_subject['response']

    @property
    def contrast_right(self):
        return self.data_subject['contrast_right']

    @property
    def contrast_left(self):
        return self.data_subject['contrast_left']

    def is_in_area(self, areas: list) -> np.ndarray:
        return np.array(np.isin(self.data_subject['brain_area'], areas))

    def is_trial_correct(self) -> np.ndarray:
        """The following are the correct responses:
        if contrast_left > contrast_right : response == 1
        if contrast_left < contrast_right : response == -1
        if contrast_left == contrast_right : response == 0
        """
        return np.array(self.response == np.sign(self.contrast_left -
                                                 self.contrast_right))

    def get_times(self):
        return self.dt * np.arange(self.num_timesteps)


def get_mean_firing_rate(spikes: np.ndarray, dt: float,
                         neuron_average: Optional[bool] = True,
                         trial_average: Optional[bool] = True,
                         time_average: Optional[bool] = False
                         ) -> np.ndarray:
    axis = tuple(np.flatnonzero([neuron_average, trial_average, time_average]))
    return np.mean(spikes, axis) / dt


def as_dataframe(rates: np.ndarray):
    import pandas as pd
    num_neurons, num_trials, num_steps = np.shape(rates)
    out = np.column_stack(
        [np.repeat(np.arange(num_neurons), num_trials * num_steps),
         np.tile(np.repeat(np.arange(num_trials), num_steps), num_neurons),
         np.ravel(rates)])
    return pd.DataFrame(out, columns=['neurons', 'trials', 'spikes'])


def main():
    path_data = '/data/datasets/steinmetz/steinmetz.npz'
    subject_index = 9
    areas = ['VISa', 'VISam', 'VISl', 'VISp', 'VISpm', 'VISrl']
    data = SteinmetzData(path_data)
    data.filter_subject(subject_index)
    trial_condition = data.is_trial_correct()
    area_conditions = {area: data.is_in_area([area]) for area in areas}
    # area_condition = data.is_in_area(areas)
    # trial_conditions = {
    #     'correct response': data.is_trial_correct(),
    #     'incorrect response': np.logical_not(data.is_trial_correct())
    #     'left responses': data.response >= 0,
    #     'right responses': data.response < 0,
    #     'stimulus on the right': data.contrast_right > 0,
    #     'no stimulus on the right': data.contrast_right == 0
    # }
    times = data.get_times()
    for label, area_condition in area_conditions.items():
        spikes = data.filter_spikes(area_condition, trial_condition)
        plt.plot(times, get_mean_firing_rate(spikes, data.dt), label=label)
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Firing rate [Hz]')
    plt.show()
    print()


if __name__ == '__main__':
    main()
    sys.exit()
