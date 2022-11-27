from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from deep_belief_network.dbn.tensorflow import SupervisedDBNClassification
import numpy as np
import pandas as pd
from sklearn.metrics._classification import accuracy_score

adhd = pd.read_csv("../feature-extraction/features/KKI_adhd_func.csv")
control = pd.read_csv("../feature-extraction/features/KKI_control_func.csv")

data = pd.concat([adhd, control])
# data = data.drop(['idx'], axis=1)
# data = data.drop(['reho'], axis=1)
# data = data.drop(['falff'], axis=1)
data = data.drop(['x'], axis=1)
data = data.drop(['y'], axis=1)
data = data.drop(['z'], axis=1)

data = data.sample(frac=1)

x = data.iloc[1:1000, 0:-1]
y = data.iloc[1:1000, -1]

ss = StandardScaler()
x = ss.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

print("creating classifier")
classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=100,
                                         batch_size=32,
                                         activation_function='relu',
                                         dropout_p=0.2)
print("classifier created")

print("fitting data")
classifier.fit(x_train, y_train)
print("data fit")

print("predicting data")
y_pred = classifier.predict(x_test)
print("data predicted")

print('\nAccuracy of Prediction: %f' % accuracy_score(x_test, y_pred))
