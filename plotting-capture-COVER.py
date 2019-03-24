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
fig_height = fig_width * golden_mean * 1/3  # height in inches
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
    'axes.labelsize': 4,
    'font.size': 4,
    'legend.fontsize': 4,
    'xtick.labelsize': 4,
    'ytick.labelsize': 4,
    # 'text.usetex': True,
    # 'text.latex.preamble': '\\usepackage{sfmath}',
    'font.family': 'sans-serif',
    'figure.figsize': fig_size
}

############## Choose parameters you like
matplotlib.rcParams.update(params_ieee)
# plt.rcParams.update({'legend.fontsize': 5.6})


import numpy as np

# #GA
# d1 = "20190319-163607-COVER-KMeans-capture.npy"
# val1 = np.load("npData/" + d1)

# d1 = "20190319-165427-COVER-KMeans-purity.npy"
# val3 = np.load("npData/" + d1)

# d1 = "20190319-164200-COVER-EM-capture.npy"
# val2 = np.load("npData/" + d1)

# d1 = "20190319-165602-COVER-EM-purity.npy"
# val4 = np.load("npData/" + d1)
# # vals = range(len(test_acc))


# plt.tight_layout()
#GA
d1 = "20190319-144931-COVER-KMeans-k-test.npy"
val1 = np.load("npData/" + d1)

d1 = "20190319-153318-COVER-EM-k-test.npy"
val2 = np.load("npData/" + d1)
# vals = range(len(test_acc))

ax = plt.subplot(141)
ax.set_title("K Means")
plt.xlabel("K")
plt.ylabel("Sum squared distances")
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

plt.plot(val1[0], val1[1], 'bx-',  markersize=4, color=my_yellow)

ax = plt.subplot(142)
ax.set_title("Expectation Maximization")
plt.xlabel("K")
plt.ylabel("Lower bound log likelihood")
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.plot(val2[0], val2[1], 'bx-', color=my_yellow,  markersize=4)
# plt.plot(vals2, valid_acc2, color=my_blue, label='Validation')






#GA
d1 = "20190319-163607-COVER-KMeans-capture.npy"
val1 = np.load("npData/" + d1)

d1 = "20190319-165440-COVER-KMeans-purity.npy"
val3 = np.load("npData/" + d1)

d1 = "20190319-164200-COVER-EM-capture.npy"
val2 = np.load("npData/" + d1)

d1 = "20190319-165602-COVER-EM-purity.npy"
val4 = np.load("npData/" + d1)
# vals = range(len(test_acc))


ax = plt.subplot(143)
ax.set_title("K Means")
plt.xlabel("K")
plt.ylabel("Percentage")
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

plt.plot(val1[0], val1[1], 'bx-', color=my_yellow, label='Capture',  markersize=4)
plt.plot(val3[0], val3[1], 'bx-', color=my_blue, label='Purity',  markersize=4)
plt.legend()
ax = plt.subplot(144)
ax.set_title("Expectation Maximization")
plt.xlabel("K")
plt.ylabel("Percentage")
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.plot(val2[0], val2[1], 'bx-', color=my_yellow, label='Capture',  markersize=4)
plt.plot(val4[0], val4[1], 'bx-', color=my_blue, label='Purity',  markersize=4)
plt.legend()
# plt.plot(vals2, valid_acc2, color=my_blue, label='Validation')

# plt.savefig("../tex/figures/COVER-capture.pdf")
plt.savefig("../tex/figures/COVER-capture.pdf")
plt.show()
