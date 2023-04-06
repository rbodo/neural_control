import numpy as np
from skimage.filters import gabor_kernel
from matplotlib import pyplot as plt


def norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


frequency = 0.1
sigma = 9
theta = -np.pi/4
contrast = 0.1
gk = gabor_kernel(frequency, theta, sigma_x=sigma, sigma_y=sigma).real
gk = np.clip(gk, 0, None)
plt.gray()
g = norm(gk)
plt.figure()
plt.imshow(contrast * g)
plt.show()

print()
