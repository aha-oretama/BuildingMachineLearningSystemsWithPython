from matplotlib import pyplot as plt
from sklearn.datasets import load_iris

data = load_iris()

features = data['data']

print(features)
feature_names = data['feature_names']
print(feature_names)
target = data['target']
print(target)
target_names = data['target_names']
labels = target_names[target]
print(labels)

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

# Distinct setosa
plength = features[:, 2]
is_setosa = (labels == "setosa")

max_setosa = plength[is_setosa].max()
min_non_setosa = plength[~is_setosa].min()
print('Maximum of setosa: {0}.'.format(max_setosa))
print('Minimum of others: {0}.'.format(min_non_setosa))

features = features[~is_setosa]
labels = labels[~is_setosa]
is_virginica = (labels == 'virginica')

## 1D
best_acc = -1.0
best_fi = -1.0
best_t = -1.0

for fi in range(features.shape[1]):
    thresh = features[:,fi]
    for t in thresh:
        feature_i = features[:,fi]
        pred = (feature_i > t)
        acc = (pred == is_virginica).mean()
        if acc > best_acc:
            best_acc = acc
            best_fi = fi
            best_t = t

print("best_acc:{0}".format(best_acc))
print("best_fi:{0}".format(best_fi))
print("best_t:{0}".format(best_t))
