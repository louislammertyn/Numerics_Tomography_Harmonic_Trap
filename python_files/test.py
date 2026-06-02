from create_waveform import example_synth_dim
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

a, drive = example_synth_dim()

xs = np.linspace(-10, 10, 5000)
ys = np.linspace(-10,10, 10)

a = drive[11](*np.meshgrid(xs,ys))[5,:]

frames = np.array([pattern(*np.meshgrid(xs,ys))[5,:] for pattern in drive])
np.save("patterns.npy", frames)

pattern = frames[33]


plt.figure()
plt.plot(xs, a)
plt.savefig("pattern.png")
plt.show()