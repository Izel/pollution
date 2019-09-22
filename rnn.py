# univariate lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import pandas as pd


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

df = pd.read_csv('data/no2Hourly.csv')
print(df.head())

# Taking the variables of interest
uni_data = df['no2']
uni_data.index = df['date']

# Dataset normalization
print('Dataset size: ' + str(len(uni_data)))
data_mean = uni_data[:len(uni_data)].mean()
#print('Mean: ' + str(data_mean))
data_std = uni_data[:len(uni_data)].std()
#print('Std: ' + str(data_std))
uni_data = (uni_data - data_mean)/data_std

# define input sequence
raw_seq = df['no2'].values

# choose a number of time steps
n_steps = 48

# split into samples
X, y = split_sequence(raw_seq, n_steps)

# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(X, y, epochs=20, verbose=0)

# demonstrate prediction
x_input = X([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
