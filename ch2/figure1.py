from matplotlib import pyplot as plt
from sklearn.datasets import load_iris

data = load_iris()

features = data['data']

print(features)
feature_names = data['feature_names']
target = data['target']
print(target)
target_names = data['target_names']
labels = target_names[target]

fig,axes = plt.subplots(2,3)
pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

color_marker = [
    ("r", ">"),
    ("g","o"),
    ("b","x")
]

for i, (p0,p1) in enumerate(pairs):
    ax = axes.flat[i]

    for t in range(3):
        c, marker = color_marker[t]
        ax.scatter(features[target == t, p0],
                   features[target == t, p1],
                   marker=marker,c=c)

    ax.set_xlabel(feature_names[p0])
    ax.set_ylabel(feature_names[p1])
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
fig.savefig('figure1.png')