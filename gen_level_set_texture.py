import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from icecream import ic


sdf = np.load("tmp/align.npy")
# sdf = np.load("tmp/regularize.npy")
dim = int(np.sqrt(len(sdf)))
sdf = sdf.reshape(dim, dim)

min_level_set = -0.03
max_level_set = 0.15
levels = np.concatenate([np.linspace(-0.04, 0, 3)[:-1], np.linspace(0, 0.15, 9)[0:]])
division = np.abs(min_level_set) / (max_level_set - min_level_set)

sdf_cm = mpl.colors.LinearSegmentedColormap.from_list(
    "SDF", [(0, "#DBD7C6"), (division, "#743E66"), (1, "#DBD7C6")], N=256
)


plt.figure()
plt.contourf(sdf, levels=levels, cmap=sdf_cm)
plt.contour(sdf, levels=levels, colors="black", linewidths=0.1)
plt.contour(sdf, levels=[0.0], colors="#370544")
plt.axis("equal")
plt.axis("off")
plt.show()


exit()

"tmp/regularize.npy"
