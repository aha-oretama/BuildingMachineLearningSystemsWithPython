import os
import scipy as sp
from sklearn.feature_extraction.text import CountVectorizer
import sys

posts = [open(os.path.join("./data",f)).read() for f in os.listdir("./data")]

vectorizer = CountVectorizer(min_df=1)

X_train = vectorizer.fit_transform(posts)
n_samples, n_features = X_train.shape
print("#samples: %d, #features: %d" % (n_samples,n_features))

print(vectorizer.get_feature_names())

new_post = "imaging databases"
new_post_vec = vectorizer.transform([new_post])

print(new_post_vec)
print(new_post_vec.toarray())


def dist_raw(v1, v2):
    delta = v1 - v2
    return sp.linalg.norm(delta.toarray())

def dist_norm(v1, v2):
    v1_normalized = v1/sp.linalg.norm(v1.toarray())
    v2_normalized = v2/sp.linalg.norm(v2.toarray())
    delta = v1_normalized - v2_normalized
    return sp.linalg.norm(delta.toarray())


best_doc = None
best_dist = sys.maxsize
best_i = None

for i in range(0, n_samples):
    post = posts[i]
    if post == new_post:
        continue
    post_vec = X_train.getrow(i)
    d = dist_norm(post_vec, new_post_vec)
    print("=== Post %i with dist=%.2f: %s" % (i, d, post))
    if(d < best_dist):
        best_dist = d
        best_i = i
        best_doc = post

print("Best post is %i with dist=%.2f" % (best_i,best_dist))
