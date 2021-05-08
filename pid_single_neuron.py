from pid import PID
import matplotlib.pyplot as plt
import numpy as np

num_timesteps = 400
setpoints = 0.1 * np.ones(num_timesteps)
setpoints[100:] *= 2

pid = PID(setpoint=1, k_p=1, k_i=1, k_d=1)

process_values = []
threshold = 1
membrane_potential = 0
spikecount = 0
for t, setpoint in enumerate(setpoints):

    membrane_potential += 0.1
    if membrane_potential > threshold:
        membrane_potential -= threshold
        spikecount += 1

    firing_rate = spikecount / (t + 1)
    control_variable = pid.update(firing_rate, setpoint=setpoint, t=t)
    threshold -= 0.1 * control_variable

    process_values.append(firing_rate)

times = np.arange(num_timesteps)
plt.plot(times, process_values, label='measured')
plt.plot(times, setpoints, label='desired')
plt.xlim(0, num_timesteps)
plt.xlabel('Time')
plt.ylabel('Firing rate')
plt.legend()
plt.show()

print()
