import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

fn = sys.argv[1]
dtw_threshold = sys.argv[2]

ned_scores, dtw_scores = [], []

with open(fn) as f:
    i = 0
    for line in f:
        i += 1
        if i > 5000:
            break
        ned, dtw = line.strip().split()
        ned_scores.append(float(ned))
        dtw_scores.append(float(dtw))

x = np.array(dtw_scores)
y = np.array(ned_scores)

g = sns.jointplot(x, y, kind='reg', xlim=[0.8,1], ylim=[0,1])
g.set_axis_labels("DTW", "NED")
#sns.regplot(x, y, ax=g.ax_joint, scatter=False)
g.ax_joint.axvline(x=dtw_threshold, lw=1, color='r', linestyle='--')
plt.show()
