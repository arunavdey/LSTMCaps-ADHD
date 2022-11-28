import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt


adhd = pd.read_csv("features/adhd_func.csv")
control = pd.read_csv("features/control_func.csv")

data = pd.concat([adhd, control])
data = data.drop(['idx'], axis=1)

X = data.iloc[1:, 0:-1]
y = data.iloc[1:, -1]

model = ExtraTreesClassifier()
model.fit(X, y)

print(model.feature_importances_)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(17).plot(kind='barh')
plt.show()
