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
    'font.size': 6,
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
plt.rcParams.update({'legend.fontsize': 4})


import numpy as np

#GA
d1 = "20190320-185853-COVER-PCA-clusterTest.npy"
val1 = np.load("npData/" + d1)

d1 = "20190320-184607-COVER-ICA-clusterTest.npy"
val2 = np.load("npData/" + d1)

d1 = "20190320-190015-COVER-RP-clusterTest.npy"
val3 = np.load("npData/" + d1)

d1 = "20190320-190424-COVER-ANOVA-clusterTest.npy"
val4 = np.load("npData/" + d1)


#GA
d1 = "20190320-195403-COVER-PCA-EM-clusterTest.npy"
val5 = np.load("npData/" + d1)

d1 = "20190320-195110-COVER-ICA-EM-clusterTest.npy"
val6 = np.load("npData/" + d1)

d1 = "20190320-195705-COVER-RP-EM-clusterTest.npy"
val7 = np.load("npData/" + d1)

d1 = "20190320-195821-COVER-ANOVA-EM-clusterTest.npy"
val8 = np.load("npData/" + d1)


#GA
d1 = "20190320-204700-MAGIC-PCA-clusterTest.npy"
val9 = np.load("npData/" + d1)

d1 = "20190320-204808-MAGIC-ICA-clusterTest.npy"
val10 = np.load("npData/" + d1)

d1 = "20190320-204900-MAGIC-RP-clusterTest.npy"
val11 = np.load("npData/" + d1)

d1 = "20190320-204938-MAGIC-ANOVA-clusterTest.npy"
val12 = np.load("npData/" + d1)


#GA
d1 = "20190320-203927-MAGIC-PCA-EM-clusterTest.npy"
val13 = np.load("npData/" + d1)

d1 = "20190320-204021-MAGIC-ICA-EM-clusterTest.npy"
val14 = np.load("npData/" + d1)

d1 = "20190320-204419-MAGIC-RP-EM-clusterTest.npy"
val15 = np.load("npData/" + d1)

d1 = "20190320-204512-MAGIC-ANOVA-EM-clusterTest.npy"
val16 = np.load("npData/" + d1)


# vals = range(len(test_acc))

ax = plt.subplot(141)
ax.set_title("Cover Type K-Means")
# plt.xlabel("Number of Resulting Dimensions")
# plt.ylabel("Harmonic mean of purity and capture")
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

plt.plot(val1[0], val1[1], label='PCA')
plt.plot(val2[0], val2[1], label='ICA')
print(val2[0][np.argmax(val2[1])])
plt.plot(val3[0], val3[1], label='RP')
plt.plot(val4[0], val4[1], label='F Test')

plt.legend()

ax = plt.subplot(142)
ax.set_title("Cover Type EM")
# plt.xlabel("Number of Resulting Dimensions")
# plt.ylabel("Harmonic mean of purity and capture")
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

plt.plot(val5[0], val5[1], label='PCA')
plt.plot(val6[0], val6[1], label='ICA')
plt.plot(val7[0], val7[1], label='RP')
plt.plot(val8[0], val8[1], label='F Test')

plt.legend()


ax = plt.subplot(143)
ax.set_title("MAGIC K-Means")
# plt.xlabel("Number of Resulting Dimensions")
# plt.ylabel("Harmonic mean of purity and capture")
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

plt.plot(val9[0], val9[1], label='PCA')
plt.plot(val10[0], val10[1], label='ICA')
plt.plot(val11[0], val11[1], label='RP')
plt.plot(val12[0], val12[1], label='F Test')

plt.legend()


ax = plt.subplot(144)
ax.set_title("MAGIC EM")
# plt.xlabel("Number of Resulting Dimensions")
# plt.ylabel("Harmonic mean of purity and capture")
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

plt.plot(val13[0], val13[1], label='PCA')
plt.plot(val14[0], val14[1], label='ICA')
plt.plot(val15[0], val15[1], label='RP')
plt.plot(val16[0], val16[1], label='F Test')

plt.legend()



plt.savefig("../tex/figures/COVER-dim-cluster.pdf")
plt.show()
