from __future__ import division, absolute_import, print_function, unicode_literals
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

#-----------
# Loading dataset
#-----------
df = pd.read_csv('data/no2Hourly.csv')
print(df.head())

# Taking the variables of interest
uni_data = df['no2']
uni_data.index = df['date']

print(uni_data.head())

# Dataset normalization
print('Dataset size: ' + str(len(uni_data)))
data_mean = uni_data[:len(uni_data)].mean()
#print('Mean: ' + str(data_mean))
data_std = uni_data[:len(uni_data)].std()
#print('Std: ' + str(data_std))
uni_data = (uni_data - data_mean)/data_std


# Samples:770 (dataset training/Timesteps)
# Timesteps:48 (Number of hours to check)
# Featues: 1 (Variables to observe, in this case just 1 (No2))
def get_data_shaped(data, start_index, end_index, timesteps):
    samples = list()
    labels = list()

    for i in range(start_index, end_index, timesteps):
        # grab from i to i + timesteps
        _sample = data[i:i+timesteps]
        _label = data[i:i+timesteps]
        samples.append(_sample)
        labels.append(_label)

    print('Size of samples: ' + str(len(samples)))
    print('Size of labels : ' + str(len(labels)))

    data_to_train = np.array(samples)
    data_labels = np.array(labels)

    print('Shape of samples: ' + str(data_to_train.shape))
    print('Size of labels : ' + str(data_labels.shape))

    # Reshaping the data into [samples, timesteps, features]
    data_to_train = data_to_train.reshape(len(samples), timesteps, 1)
    data_labels = data_labels.reshape(len(labels), timesteps, 1)

    print('Reshaped data_to_train: ' + str(data_to_train.shape))

    return data_to_train, data_labels

# With 770 samples, lets define a batch size not too big. 7 is a good number
# because de division of 770/7 gives an integer that is not too small (higher than 50)
# and not too high (over 400)
BATCH = 7
UNITS = 110
INPUTS = BATCH * UNITS

model = Sequential()
model.add(LSTM(24, activation='relu', batch_input_shape=(None, 48, 1), return_sequences=True))
model.add(Dense(1))
model.summary()
model.compile(optimizer='adam', loss='mse')

x_train, y_train = get_data_shaped(uni_data, 0, 36960, 48)
x_test, y_test = get_data_shaped(uni_data, 36961, 52561, 48)

# fit model
model.fit(x_train, y_train, epochs=10,  verbose=0)

yhat = model.predict(x_test, verbose=0)
print(x_test)
print(yhat)
