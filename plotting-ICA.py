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
fig_height = fig_width * golden_mean * 2/3  # height in inches
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
d1 = "20190320-123941-COVER-ICA-sorted-kurtosis.npy"
val1 = np.load("npData/" + d1)

d1 = "20190320-123955-COVER-PLAIN-sorted-kurtosis.npy"
val2 = np.load("npData/" + d1)
# vals = range(len(test_acc))

d1 = "20190320-125051-COVER-PLAIN-kurtosis.npy"
val3 = np.load("npData/" + d1)

ax = plt.subplot(121)
ax.set_title("Kurtosis Values")
plt.xlabel("Axis sorted by Kurtosis")
plt.ylabel("Kurtosis")
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

plt.bar(range(len(val1)), val1, color=my_blue, label='ICA')
plt.bar(range(len(val2)), val2, color=my_yellow, label='Untransformed')
plt.legend()

ax = plt.subplot(122)
ax.set_title("Kurtosis of First 20 Attributes")
plt.xlabel("Attribute #")
plt.ylabel("Kurtosis")
plt.bar(range(len(val3[0:20])), val3[0:20], color=my_blue)

# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# plt.plot(val2[0], val2[1], 'bx-', color=my_yellow)
# plt.plot(vals2, valid_acc2, color=my_blue, label='Validation')

# plt.savefig("../tex/figures/COVER-Ktest.pdf")
plt.savefig("../tex/figures/COVER-ICA.pdf")
plt.show()