from src.pid import PID
import matplotlib.pyplot as plt
import numpy as np

num_timesteps = 400
setpoints = 0.2 * np.ones(num_timesteps)
n = 50
setpoints[:n] = np.arange(n) / n / 2
setpoints[200:] *= 2

pid = PID(k_p=2, k_i=0.1, k_d=0.1)

process_values = []
threshold = 1
membrane_potential = 0
spikecount = 0
for t, setpoint in enumerate(setpoints):

    inp = setpoint  # Neuron is a simple source follower.
    firing_rate = spikecount / (t + 1)
    control_variable = pid.update(firing_rate, setpoint=setpoint, t=t)
    inp += control_variable

    membrane_potential += inp
    if membrane_potential > threshold:
        # membrane_potential = 0
        membrane_potential -= threshold
        spikecount += 1

    firing_rate = spikecount / (t + 1)
    # control_variable = pid.update(firing_rate, setpoint=setpoint, t=t)
    # threshold -= 0.1 * control_variable

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
