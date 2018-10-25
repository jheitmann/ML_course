
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

from proj1_helpers import load_csv_data
import numpy as np

ITERATIONS = 50000
SAMPLE_SIZE = 250000 # less or eq than 250000. step=250k/SAMPLE_SIZE

y, X, _, _ = load_csv_data('all/train.csv', step=int(250000./SAMPLE_SIZE))

# Cleans dataset by removing all features that admit undefined values.
undef_features = [i for i, feature in enumerate(X.T) if -999 in feature]
X = np.delete(X, undef_features, axis=1)

clf = LogisticRegression(solver='newton-cg', max_iter=ITERATIONS).fit(X, y)
print(cross_validate(clf, X, y, scoring=['accuracy', 'precision']))
