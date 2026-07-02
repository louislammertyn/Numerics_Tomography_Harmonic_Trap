from create_waveform import example_synth_dim, example_synth_dim_2drive
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

a, drive = example_synth_dim()

xs = np.linspace(-10, 10, 5000)
ys = np.linspace(-10,10, 10)

a = drive[11](*np.meshgrid(xs,ys))[5,:]

frames = np.array([pattern(*np.meshgrid(xs,ys))[5,:] for pattern in drive])
np.save("../data/pattern_single_drive.npy", frames)

pattern = frames[33]


plt.figure()
plt.plot(xs, a)
plt.savefig("pattern_single_drive.png")
plt.show()

a2, drive2 = example_synth_dim_2drive(detuning=0.02)

xs = np.linspace(-10, 10, 5000)
ys = np.linspace(-10,10, 10)

a2 = drive2[11](*np.meshgrid(xs,ys))[5,:]

frames = np.array([pattern(*np.meshgrid(xs,ys))[5,:] for pattern in drive2])
np.save("../data/pattern_2drive.npy", frames)

pattern = frames[33]


plt.figure()
plt.plot(xs, a2)
plt.savefig("pattern_2drive.png")
plt.show()