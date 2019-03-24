import scipy as sp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm

############## Options to generate nice figures
fig_width_pt = 500.0  # Get this from LaTeX using \showthe\column-width
inches_per_pt = 1.0 / 72.27  # Convert pt to inch
golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = fig_width * golden_mean / 3  # height in inches
fig_size = [fig_width, fig_height]

############## Colors I like to use
my_yellow = [235. / 255, 164. / 255, 17. / 255]
my_blue = [58. / 255, 93. / 255, 163. / 255]
dark_gray = [68./255, 84. /255, 106./255]
my_red = [163. / 255, 93. / 255, 58. / 255]

my_color = dark_gray # pick color for theme

params_keynote = {
    'axes.labelsize': 16,
    'font.size': 16,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    # 'text.usetex': True,
    # 'text.latex.preamble': '\\usepackage{sfmath}',
    'font.family': 'sans-serif',
    'figure.figsize': fig_size
}
############## Parameters I use for IEEE papers
params_ieee = {
    'figure.autolayout' : True,
    'axes.labelsize': 8,
    'font.size': 8,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    # 'text.usetex': True,
    # 'text.latex.preamble': '\\usepackage{sfmath}',
    'font.family': 'sans-serif',
    'figure.figsize': fig_size
}

############## Choose parameters you like
matplotlib.rcParams.update(params_ieee)
# plt.rcParams.update({'legend.fontsize': 5.6})


import numpy as np

#GA
d1 = "20190319-190144-MAGIC-TSNE-.npy"
val1 = np.load("npData/" + d1)

d1 = "20190319-190144-MAGIC-y-.npy"
val2 = np.load("npData/" + d1)

d1 = "20190319-205132-MAGIC-KMeans-2k-labels.npy"
val3 = np.load("npData/" + d1)

d1 = "20190319-205148-MAGIC-KMeans-3k-labels.npy"
val4 = np.load("npData/" + d1)


d1 = "20190319-210920-MAGIC-EM-2k-labels.npy"
val5 = np.load("npData/" + d1)

d1 = "20190319-210910-MAGIC-EM-3k-labels.npy"
val6 = np.load("npData/" + d1)



# vals = range(len(test_acc))

# d1 = "20190319-174545-COVER-EM-2k-labels.npy"
# val2 = np.load("npData/" + d1)

# d1 = "20190319-174553-COVER-EM-7k-labels.npy"
# val4 = np.load("npData/" + d1)
# # vals = range(len(test_acc))

from sklearn.utils import resample

print(val1)

ax = plt.subplot(161)
ax.set_title("Unlabeled")
# plt.xlabel("K")
# plt.ylabel("Percentage")
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

plt.tick_params(        # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,
    left=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft=False) # labels along the bottom edge are off

a1, a2 = resample(val1[:, 0], val1[:, 1], n_samples=1500)
plt.scatter(a1, a2, s=2)
ax = plt.subplot(162)
ax.set_title("Dataset Labels")
plt.tick_params(        # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,
    left=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft=False)
for i in range(0, 2):
    where = np.where(val2 ==i)
    where = resample(where[0], n_samples=min(1500 // 2, len(where[0])))
    plt.scatter(val1[where, 0], val1[where, 1], s=2, alpha=0.5)



ax = plt.subplot(163)
ax.set_title("K-Means\nk = 2")
# plt.xlabel("K")
# plt.ylabel("Percentage")
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

plt.tick_params(        # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,
    left=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft=False) # labels along the bottom edge are off

for i in range(0, 2):
    where = np.where(val3 ==i)
    if (len(where[0]) > 0):
        where = resample(where[0], n_samples=min(1500 // 2, len(where[0])))
        plt.scatter(val1[where, 0], val1[where, 1], s=2, alpha=0.5)

ax = plt.subplot(164)
ax.set_title("K-Means\nk = 3")
plt.tick_params(        # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,
    left=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft=False)
for i in range(0, 4):
    where = np.where(val4 ==i)
    if (len(where[0]) > 0):
        where = resample(where[0], n_samples=min(1500 // 3, len(where[0])))
        plt.scatter(val1[where, 0], val1[where, 1], s=2, alpha=0.5)


ax = plt.subplot(165)
ax.set_title("EM\nk = 2")
plt.tick_params(        # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,
    left=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft=False) # labels along the bottom edge are off

for i in range(0, 4):
    where = np.where(val5 ==i)
    if (len(where[0]) > 0):
        where = resample(where[0], n_samples=min(1500 // 2, len(where[0])))
        plt.scatter(val1[where, 0], val1[where, 1], s=2, alpha=0.5)

ax = plt.subplot(166)
ax.set_title("EM\nk = 3")
plt.tick_params(        # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,
    left=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft=False)
for i in range(0, 4):
    where = np.where(val6 ==i)
    if (len(where[0]) > 0):
        where = resample(where[0], n_samples=min(1500 // 3, len(where[0])))
        plt.scatter(val1[where, 0], val1[where, 1], s=2, alpha=0.5)

plt.savefig("../tex/figures/MAGIC-cluster.pdf")
plt.show()
