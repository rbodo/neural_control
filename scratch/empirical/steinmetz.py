import sys

import numpy as np
import matplotlib.pyplot as plt
import pysensors as ps
from empirical.steinmetz import SteinmetzData, get_mean_firing_rate


def main():
    path_data = '/data/datasets/steinmetz'
    subject_index = 9
    areas = ['VISl', 'VISp', 'VISrl']
    data = SteinmetzData(path_data)
    data.filter_subject(subject_index)
    trial_condition = data.is_trial_correct()
    # Use only trials with stimulus on the right and correct repsonse.
    # trial_condition = np.logical_and(data.response < 0,
    #                                  data.contrast_right > 0)
    area_conditions = {area: data.is_in_area([area]) for area in areas}
    times = data.get_times()
    rates = []
    for label, area_condition in area_conditions.items():
        spikes = data.filter_spikes(area_condition, trial_condition)
        r = get_mean_firing_rate(spikes, data.dt)
        rates.append(r)
        plt.plot(times, r, label=label)
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Firing rate [Hz]')
    plt.show()

    X = np.array(rates).T
    n_sensors = 1
    model = ps.SSPOR(n_sensors=n_sensors)
    model.fit(X)
    sensors = model.get_selected_sensors()
    X_reconstructed = model.predict(X[:, sensors])
    plt.plot(X)
    plt.show()
    plt.plot(X_reconstructed)
    plt.show()
    print()


if __name__ == '__main__':
    main()
    sys.exit()
