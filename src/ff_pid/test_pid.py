from src.pid import PID
import matplotlib.pyplot as plt
import numpy as np

num_timesteps = 200
setpoints = np.ones(num_timesteps)
setpoints[num_timesteps//2:] *= 2

pid = PID(setpoint=1, k_p=1, k_i=1, k_d=1)

process_values = []
process_value = 0
for t, setpoint in enumerate(setpoints):
    pid.setpoint = setpoint
    control_variable = pid.update(process_value, t=t)
    process_value += 0.1 * control_variable

    process_values.append(process_value)

times = np.arange(num_timesteps)
plt.plot(times, process_values, label='measured')
plt.plot(times, setpoints, label='desired')
plt.xlim(0, num_timesteps)
plt.xlabel('Time')
plt.ylabel('Process Value')
plt.legend()
plt.show()

print()
