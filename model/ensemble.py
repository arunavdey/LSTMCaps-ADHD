import keras
import numpy as np
from capsulenet import get_model as getCapsNet
from lstm import get_model as getLSTMNet

# TODO
# write APIs for both LSTM and CapsNet
# get ensemble working


models = [getCapsNet(), getLSTMNet()]

print(models[0].shape)  # 16, 4
print(models[1].shape)  # aVeryBigNumber, 4


for model in models:
    print('pred:', np.argmax(model, 1))

# CapsNet.summary()
# LSTMNet.summary()

# ensemble_output = keras.layers.Average()(models)
# ensemble_model = keras.Model(inputs=models, outputs=ensemble_output)

# ensemble_model.summary()
