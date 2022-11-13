import keras
import lstm
import capsulenet
from capsulenet import get_model as CapsNet
from lstm import get_model as LSTMNet

# TODO
# write APIs for both LSTM and CapsNet
# get ensemble working

models = [CapsNet, LSTMNet]

model_inputs = [keras.Input(shape=()), keras.Input(shape=())]
model_outputs = [model(model_inputs[0]) for model in models]

ensemble_output = keras.layers.Average()(model_outputs)
ensemble_model = keras.Model(inputs=model_inputs, outputs=ensemble_output)
