import os
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import numpy as np
from datetime import date
import csv
from folium.plugins import HeatMap
import os
import folium
import time
import webbrowser
import branca.colormap as cmp
from folium.features import DivIcon
from folium.plugins import FloatImage

os.environ ['TF_CPP_MIN_LOG_LEVEL'] = '2'

caminho = r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\Previsao'
cmh = os.path.join(caminho, 'Resultados') 
################################################Potreiro 20 Infestado############################################################# 
# load dataset
dataset1 = read_csv('entrada_Inf.csv', header=0, delimiter=';')
values1 = dataset1.values
# integer encode direction
encoder = LabelEncoder()

dataset2 = read_csv('entclima1.csv', header=0, delimiter=';')
values2 = dataset2.values
# integer encode direction
encoder = LabelEncoder()

#Drop NA Values
#values1=values1[~np.isnan(values1).any(axis=1)]
#values2=values2[~np.isnan(values2).any(axis=1)]

# ensure all data is float
values1 = values1.astype('float32')
values2 = values2.astype('float32')

real1 = values1[:, -1]

# normalize features
scaler = MinMaxScaler()
scaled1 = scaler.fit_transform(values1)
scaled2 = scaler.fit_transform(values2)

scaled1 = DataFrame(scaled1)
scaled2 = DataFrame(scaled2)
#print(scaled)
#print(scaled2) 
# split into train and test sets
#values = scaled.values
#n_train = 24
#n_train = 48
train1 = values1
test1 = values2
#print(train)
#print(test)
# split into input and outputs
train_X1, train_y1 = train1[:, :-1], train1[:, -1]
test_X1, test_y1 = test1[:, :-1], test1[:, -1]

#print(train_X)
#print(train_y)
#print(test_X)
#print(test_y)

# reshape input to be 3D [samples, timesteps, features]
train_X1 = train_X1.reshape((train_X1.shape[0], 1, train_X1.shape[1]))
test_X1 = test_X1.reshape((test_X1.shape[0], 1, test_X1.shape[1]))

#print(train_X)
#print(train_y)
#print(test_X)
#print(test_y)

print(train_X1.shape, train_y1.shape, test_X1.shape, test_y1.shape)
print(train_X1.shape[1])

if not os.path.exists('M_INF_PREVISAO.h5'):
	print('modelo inexistente... criando modelo...')
	# design network
	model = Sequential()
	#model.add(LSTM(30, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh', return_sequences = True))#bom para 1 e 2
	#model.add(LSTM(15, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh', return_sequences = True))#bom para 1 e 2
	#model.add(LSTM(7, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh'))


	model.add(LSTM(30, input_shape=(train_X1.shape[1], train_X1.shape[2]),	kernel_initializer='normal',  return_sequences = True))#bom para 1 e 2
	model.add(LSTM(15, input_shape=(train_X1.shape[1], train_X1.shape[2]),	kernel_initializer='normal'))

	#model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='sigmoid'))
	#model.add(LSTM(10, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='softmax'))
	#model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='softplus'))
	#model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='softsign'))
	#model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='hard_sigmoid'))
	#model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='linear'))
	#model.add(LSTM(10, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='selu'))#bom para 2
	#model.add(LSTM(15, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='relu'))#bom para 2

	#model.add(Dense(500, kernel_initializer='normal', activation='tanh'))#para 2 melhorou

	# Hidden - Layers

	#model.add(Dropout(0.3, noise_shape=None, seed=None))
	#model.add(Dense(50, activation = "tanh"))
	#model.add(Dropout(0.2, noise_shape=None, seed=None))
	#model.add(Dense(10, activation = "tanh"))

	model.add(Dense(1, kernel_initializer='normal'))

	#model.compile(loss='mse', optimizer='SGD')
	model.compile(loss='mean_squared_error', optimizer='rmsprop')#mto bom para 1 e 2
	#model.compile(loss='mse', optimizer='Adagrad')#mto bom para 1
	#model.compile(loss='mse', optimizer='Adadelta')#mto bom para 1 e 2
	#model.compile(loss='mse', optimizer='Adam')#mto bom para 1
	#model.compile(loss='mse', optimizer='Adamax')#mto bom para 1
	#model.compile(loss='mse', optimizer='Nadam')

	# fit network
	history = model.fit(train_X1, train_y1, epochs=5000, batch_size=72, validation_data=(test_X1, test_y1), verbose=0, shuffle=False)
	#------------------
	# save the network's architecture
	print('Salvando modelo...')
	model.save('M_INF_PREVISAO.h5')
	#------------------
else:
	#------------------
	# load the network's architecture
	model = load_model('M_INF_PREVISAO.h5')
	history =  model
	#------------------
	print('Modelo encontrado e carregado')
	
# make a prediction
yhat1 = model.predict(test_X1)
test_X1 = test_X1.reshape((test_X1.shape[0], test_X1.shape[2]))
# invert scaling for forecast
inv_yhat1 = concatenate((test_X1, yhat1), axis=1)
inv_yhat1 = scaler.inverse_transform(inv_yhat1)
inv_yhat1 = inv_yhat1[:,-1]
# invert scaling for actual
test_y1 = test_y1.reshape((len(test_y1), 1))
inv_y1 = concatenate((test_X1, test_y1), axis=1)
#print(inv_y1[:,-1])
#inv_y1 = scaler.inverse_transform(inv_y1)
inv_y1 = inv_y1[:,-1]

print(model.summary())


from keras.utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#from ann_visualizer.visualize import ann_viz

#ann_viz(model, title="My first neural network")

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y1, inv_yhat1))
print('P20 - Infestado')
print('Test RMSE: %.3f' % rmse)
print('Real')
print(inv_y1)
print('Predito')
print(inv_yhat1)

file = open(cmh + "\dados.txt", "w")
file.write('P20 - Infestado' + '\n')
file.write('Test RMSE:' + '%.3f' % rmse + '\n')
file.write('Real' + '\n')
for a in inv_y1:
	file.write(str(a) + ',' + ' ')
file.write('\n' + 'Predito' + '\n')
for b in inv_yhat1:
	file.write(str(b) + ',' + ' ')
file.close()
################################################Potreiro 20 Infestado############################################################# 

################################################Potreiro 20 Mirapasto############################################################# 
# load dataset
dataset3 = read_csv('entrada_Mira.csv', header=0, delimiter=';')
values3 = dataset3.values
# integer encode direction
encoder = LabelEncoder()

dataset4 = read_csv('entclima2.csv', header=0, delimiter=';')
values4 = dataset4.values
# integer encode direction
encoder = LabelEncoder()

#Drop NA Values
#values3=values3[~np.isnan(values3).any(axis=1)]
#values4=values4[~np.isnan(values4).any(axis=1)]

# ensure all data is float
values3 = values3.astype('float32')
values4 = values4.astype('float32')

real2 = values3[:, -1]

# normalize features
scaler = MinMaxScaler()
scaled3 = scaler.fit_transform(values3)
scaled4 = scaler.fit_transform(values4)

scaled3 = DataFrame(scaled3)
scaled4 = DataFrame(scaled4)
#print(scaled)
#print(scaled2) 
# split into train and test sets
#values = scaled.values
#n_train = 24
#n_train = 48
train2 = values3
test2 = values4
#print(train)
#print(test)
# split into input and outputs
train_X2, train_y2 = train2[:, :-1], train2[:, -1]
test_X2, test_y2 = test2[:, :-1], test2[:, -1]

#print(train_X)
#print(train_y)
#print(test_X)
#print(test_y)

# reshape input to be 3D [samples, timesteps, features]
train_X2 = train_X2.reshape((train_X2.shape[0], 1, train_X2.shape[1]))
test_X2 = test_X2.reshape((test_X2.shape[0], 1, test_X2.shape[1]))

#print(train_X)
#print(train_y)
#print(test_X)
#print(test_y)

print(train_X2.shape, train_y2.shape, test_X2.shape, test_y2.shape)
print(train_X2.shape[1])

if not os.path.exists('M_MIRA_PREVISAO.h5'):
	print('modelo inexistente... criando modelo...')
	# design network
	model = Sequential()
	#model.add(LSTM(30, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh', return_sequences = True))#bom para 1 e 2
	#model.add(LSTM(15, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh', return_sequences = True))#bom para 1 e 2
	#model.add(LSTM(7, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh'))


	model.add(LSTM(30, input_shape=(train_X2.shape[1], train_X2.shape[2]),	kernel_initializer='normal',  return_sequences = True))#bom para 1 e 2
	model.add(LSTM(15, input_shape=(train_X2.shape[1], train_X2.shape[2]),	kernel_initializer='normal'))

	#model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='sigmoid'))
	#model.add(LSTM(10, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='softmax'))
	#model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='softplus'))
	#model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='softsign'))
	#model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='hard_sigmoid'))
	#model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='linear'))
	#model.add(LSTM(10, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='selu'))#bom para 2
	#model.add(LSTM(15, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='relu'))#bom para 2

	#model.add(Dense(500, kernel_initializer='normal', activation='tanh'))#para 2 melhorou

	# Hidden - Layers

	#model.add(Dropout(0.3, noise_shape=None, seed=None))
	#model.add(Dense(50, activation = "tanh"))
	#model.add(Dropout(0.2, noise_shape=None, seed=None))
	#model.add(Dense(10, activation = "tanh"))

	model.add(Dense(1, kernel_initializer='normal'))

	#model.compile(loss='mse', optimizer='SGD')
	model.compile(loss='mean_squared_error', optimizer='rmsprop')#mto bom para 1 e 2
	#model.compile(loss='mse', optimizer='Adagrad')#mto bom para 1
	#model.compile(loss='mse', optimizer='Adadelta')#mto bom para 1 e 2
	#model.compile(loss='mse', optimizer='Adam')#mto bom para 1
	#model.compile(loss='mse', optimizer='Adamax')#mto bom para 1
	#model.compile(loss='mse', optimizer='Nadam')

	# fit network
	history = model.fit(train_X2, train_y2, epochs=5000, batch_size=72, validation_data=(test_X2, test_y2), verbose=0, shuffle=False)
	#------------------
	# save the network's architecture
	print('Salvando modelo...')
	model.save('M_MIRA_PREVISAO.h5')
	#------------------
else:
	#------------------
	# load the network's architecture
	model = load_model('M_MIRA_PREVISAO.h5')
	history =  model
	#------------------
	print('Modelo encontrado e carregado')
	
# make a prediction
yhat2 = model.predict(test_X2)
test_X2 = test_X2.reshape((test_X2.shape[0], test_X2.shape[2]))
# invert scaling for forecast
inv_yhat2 = concatenate((test_X2, yhat2), axis=1)
inv_yhat2 = scaler.inverse_transform(inv_yhat2)
inv_yhat2 = inv_yhat2[:,-1]
# invert scaling for actual
test_y2 = test_y2.reshape((len(test_y2), 1))
inv_y2 = concatenate((test_X2, test_y2), axis=1)
#inv_y2 = scaler.inverse_transform(inv_y2)
inv_y2 = inv_y2[:,-1]

print(model.summary())


from keras.utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#from ann_visualizer.visualize import ann_viz

#ann_viz(model, title="My first neural network")

# calculate RMSE
rmse2 = sqrt(mean_squared_error(inv_y2, inv_yhat2))
print('P20 - Mirapasto')
print('Test RMSE: %.3f' % rmse2)
print('Real')
print(inv_y2)
print('Predito')
print(inv_yhat2)

file = open(cmh + "\dados.txt", "a")
file.write('\n\r' + 'P20 - Mirapasto' + '\n')
file.write('Test RMSE:' + '%.3f' % rmse2 + '\n')
file.write('Real' + '\n')
for a in inv_y2:
	file.write(str(a) + ',' + ' ')
file.write('\n' + 'Predito' + '\n')
for b in inv_yhat2:
	file.write(str(b) + ',' + ' ')
file.close()
################################################Potreiro 20 Mirapasto#############################################################

################################################Potreiro 21 Infestado############################################################# 
# load dataset
dataset5 = read_csv('entrada_Inf.csv', header=0, delimiter=';')
values5 = dataset5.values
# integer encode direction
encoder = LabelEncoder()

dataset6 = read_csv('entclima3.csv', header=0, delimiter=';')
values6 = dataset6.values
# integer encode direction
encoder = LabelEncoder()

#Drop NA Values
#values5=values5[~np.isnan(values5).any(axis=1)]
#values6=values6[~np.isnan(values6).any(axis=1)]

# ensure all data is float
values5 = values5.astype('float32')
values6 = values6.astype('float32')

real3 = values5[:, -1]

# normalize features
scaler = MinMaxScaler()
scaled5 = scaler.fit_transform(values5)
scaled6 = scaler.fit_transform(values6)

scaled5 = DataFrame(scaled5)
scaled6 = DataFrame(scaled6)
#print(scaled)
#print(scaled2) 
# split into train and test sets
#values = scaled.values
#n_train = 24
#n_train = 48
train3 = values5
test3 = values6
#print(train)
#print(test)
# split into input and outputs
train_X3, train_y3 = train3[:, :-1], train3[:, -1]
test_X3, test_y3 = test3[:, :-1], test3[:, -1]

#print(train_X)
#print(train_y)
#print(test_X)
#print(test_y)

# reshape input to be 3D [samples, timesteps, features]
train_X3 = train_X3.reshape((train_X3.shape[0], 1, train_X3.shape[1]))
test_X3 = test_X3.reshape((test_X3.shape[0], 1, test_X3.shape[1]))

#print(train_X)
#print(train_y)
#print(test_X)
#print(test_y)

print(train_X3.shape, train_y3.shape, test_X3.shape, test_y3.shape)
print(train_X3.shape[1])

if not os.path.exists('M_INF_PREVISAO.h5'):
	print('modelo inexistente... criando modelo...')
	# design network
	model = Sequential()
	#model.add(LSTM(30, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh', return_sequences = True))#bom para 1 e 2
	#model.add(LSTM(15, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh', return_sequences = True))#bom para 1 e 2
	#model.add(LSTM(7, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh'))


	model.add(LSTM(30, input_shape=(train_X3.shape[1], train_X3.shape[2]),	kernel_initializer='normal',  return_sequences = True))#bom para 1 e 2
	model.add(LSTM(15, input_shape=(train_X3.shape[1], train_X3.shape[2]),	kernel_initializer='normal'))

	#model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='sigmoid'))
	#model.add(LSTM(10, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='softmax'))
	#model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='softplus'))
	#model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='softsign'))
	#model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='hard_sigmoid'))
	#model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='linear'))
	#model.add(LSTM(10, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='selu'))#bom para 2
	#model.add(LSTM(15, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='relu'))#bom para 2

	#model.add(Dense(500, kernel_initializer='normal', activation='tanh'))#para 2 melhorou

	# Hidden - Layers

	#model.add(Dropout(0.3, noise_shape=None, seed=None))
	#model.add(Dense(50, activation = "tanh"))
	#model.add(Dropout(0.2, noise_shape=None, seed=None))
	#model.add(Dense(10, activation = "tanh"))

	model.add(Dense(1, kernel_initializer='normal'))

	#model.compile(loss='mse', optimizer='SGD')
	model.compile(loss='mean_squared_error', optimizer='rmsprop')#mto bom para 1 e 2
	#model.compile(loss='mse', optimizer='Adagrad')#mto bom para 1
	#model.compile(loss='mse', optimizer='Adadelta')#mto bom para 1 e 2
	#model.compile(loss='mse', optimizer='Adam')#mto bom para 1
	#model.compile(loss='mse', optimizer='Adamax')#mto bom para 1
	#model.compile(loss='mse', optimizer='Nadam')

	# fit network
	history = model.fit(train_X3, train_y3, epochs=5000, batch_size=72, validation_data=(test_X3, test_y3), verbose=0, shuffle=False)
	#------------------
	# save the network's architecture
	print('Salvando modelo...')
	model.save('M_INF_PREVISAO.h5')
	#------------------
else:
	#------------------
	# load the network's architecture
	model = load_model('M_INF_PREVISAO.h5')
	history =  model
	#------------------
	print('Modelo encontrado e carregado')
	
# make a prediction
yhat3 = model.predict(test_X3)
test_X3 = test_X3.reshape((test_X3.shape[0], test_X3.shape[2]))
# invert scaling for forecast
inv_yhat3 = concatenate((test_X3, yhat3), axis=1)
inv_yhat3 = scaler.inverse_transform(inv_yhat3)
inv_yhat3 = inv_yhat3[:,-1]
# invert scaling for actual
test_y3 = test_y3.reshape((len(test_y3), 1))
inv_y3 = concatenate((test_X3, test_y3), axis=1)
#inv_y2 = scaler.inverse_transform(inv_y2)
inv_y3 = inv_y3[:,-1]

print(model.summary())


from keras.utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#from ann_visualizer.visualize import ann_viz

#ann_viz(model, title="My first neural network")

# calculate RMSE
rmse3 = sqrt(mean_squared_error(inv_y3, inv_yhat3))
print('P21 - Infestado')
print('Test RMSE: %.3f' % rmse3)
print('Real')
print(inv_y3)
print('Predito')
print(inv_yhat3)

file = open(cmh + "\dados.txt", "a")
file.write('\n\r' + 'P21 - Infestado' + '\n')
file.write('Test RMSE:' + '%.3f' % rmse3 + '\n')
file.write('Real' + '\n')
for a in inv_y3:
	file.write(str(a) + ',' + ' ')
file.write('\n' + 'Predito' + '\n')
for b in inv_yhat3:
	file.write(str(b) + ',' + ' ')
file.close()
################################################Potreiro 21 Infestado#############################################################

################################################Potreiro 21 Mirapasto############################################################# 
# load dataset
dataset7 = read_csv('entrada_Mira.csv', header=0, delimiter=';')
values7 = dataset7.values
# integer encode direction
encoder = LabelEncoder()

dataset8 = read_csv('entclima4.csv', header=0, delimiter=';')
values8 = dataset8.values
# integer encode direction
encoder = LabelEncoder()

#Drop NA Values
#values7=values7[~np.isnan(values7).any(axis=1)]
#values8=values8[~np.isnan(values8).any(axis=1)]

# ensure all data is float
values7 = values7.astype('float32')
values8 = values8.astype('float32')

real4 = values7[:, -1]

# normalize features
scaler = MinMaxScaler()
scaled7 = scaler.fit_transform(values7)
scaled8 = scaler.fit_transform(values8)

scaled7 = DataFrame(scaled7)
scaled8 = DataFrame(scaled8)
#print(scaled)
#print(scaled2) 
# split into train and test sets
#values = scaled.values
#n_train = 24
#n_train = 48
train4 = values7
test4 = values8
#print(train)
#print(test)
# split into input and outputs
train_X4, train_y4 = train4[:, :-1], train4[:, -1]
test_X4, test_y4 = test4[:, :-1], test4[:, -1]

#print(train_X)
#print(train_y)
#print(test_X)
#print(test_y)

# reshape input to be 3D [samples, timesteps, features]
train_X4 = train_X4.reshape((train_X4.shape[0], 1, train_X4.shape[1]))
test_X4 = test_X4.reshape((test_X4.shape[0], 1, test_X4.shape[1]))

#print(train_X)
#print(train_y)
#print(test_X)
#print(test_y)

print(train_X4.shape, train_y4.shape, test_X4.shape, test_y4.shape)
print(train_X4.shape[1])

if not os.path.exists('M_MIRA_PREVISAO.h5'):
	print('modelo inexistente... criando modelo...')
	# design network
	model = Sequential()
	#model.add(LSTM(30, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh', return_sequences = True))#bom para 1 e 2
	#model.add(LSTM(15, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh', return_sequences = True))#bom para 1 e 2
	#model.add(LSTM(7, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh'))


	model.add(LSTM(30, input_shape=(train_X4.shape[1], train_X4.shape[2]),	kernel_initializer='normal',  return_sequences = True))#bom para 1 e 2
	model.add(LSTM(15, input_shape=(train_X4.shape[1], train_X4.shape[2]),	kernel_initializer='normal'))

	#model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='sigmoid'))
	#model.add(LSTM(10, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='softmax'))
	#model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='softplus'))
	#model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='softsign'))
	#model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='hard_sigmoid'))
	#model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='linear'))
	#model.add(LSTM(10, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='selu'))#bom para 2
	#model.add(LSTM(15, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='relu'))#bom para 2

	#model.add(Dense(500, kernel_initializer='normal', activation='tanh'))#para 2 melhorou

	# Hidden - Layers

	#model.add(Dropout(0.3, noise_shape=None, seed=None))
	#model.add(Dense(50, activation = "tanh"))
	#model.add(Dropout(0.2, noise_shape=None, seed=None))
	#model.add(Dense(10, activation = "tanh"))

	model.add(Dense(1, kernel_initializer='normal'))

	#model.compile(loss='mse', optimizer='SGD')
	model.compile(loss='mean_squared_error', optimizer='rmsprop')#mto bom para 1 e 2
	#model.compile(loss='mse', optimizer='Adagrad')#mto bom para 1
	#model.compile(loss='mse', optimizer='Adadelta')#mto bom para 1 e 2
	#model.compile(loss='mse', optimizer='Adam')#mto bom para 1
	#model.compile(loss='mse', optimizer='Adamax')#mto bom para 1
	#model.compile(loss='mse', optimizer='Nadam')

	# fit network
	history = model.fit(train_X4, train_y4, epochs=5000, batch_size=72, validation_data=(test_X4, test_y4), verbose=0, shuffle=False)
	#------------------
	# save the network's architecture
	print('Salvando modelo...')
	model.save('M_MIRA_PREVISAO.h5')
	#------------------
else:
	#------------------
	# load the network's architecture
	model = load_model('M_MIRA_PREVISAO.h5')
	history =  model
	#------------------
	print('Modelo encontrado e carregado')
	
# make a prediction
yhat4 = model.predict(test_X4)
test_X4 = test_X4.reshape((test_X4.shape[0], test_X4.shape[2]))
# invert scaling for forecast
inv_yhat4 = concatenate((test_X4, yhat4), axis=1)
inv_yhat4 = scaler.inverse_transform(inv_yhat4)
inv_yhat4 = inv_yhat4[:,-1]
# invert scaling for actual
test_y4 = test_y4.reshape((len(test_y4), 1))
inv_y4 = concatenate((test_X4, test_y4), axis=1)
#inv_y2 = scaler.inverse_transform(inv_y2)
inv_y4 = inv_y4[:,-1]

print(model.summary())


from keras.utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#from ann_visualizer.visualize import ann_viz

#ann_viz(model, title="My first neural network")

# calculate RMSE
rmse4 = sqrt(mean_squared_error(inv_y4, inv_yhat4))
print('P21 - Mirapasto')
print('Test RMSE: %.3f' % rmse4)
print('Real')
print(inv_y4)
print('Predito')
print(inv_yhat4)

file = open(cmh + "\dados.txt", "a")
file.write('\n\r' + 'P21 - Mirapasto' + '\n')
file.write('Test RMSE:' + '%.3f' % rmse4 + '\n')
file.write('Real' + '\n')
for a in inv_y4:
	file.write(str(a) + ',' + ' ')
file.write('\n' + 'Predito' + '\n')
for b in inv_yhat4:
	file.write(str(b) + ',' + ' ')
file.close()
################################################Potreiro 21 Mirapasto#############################################################

############################Criação arquivo csv das saídas################################################################


with open(r"C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\Previsao\saida_rede.csv", 'w', newline='') as file:
	
	writer = csv.writer(file, delimiter=';')
	
	writer.writerow(["Id", "TX"])
	writer.writerow(["Um", "%.2f" % inv_yhat1[0]])
	writer.writerow(["Dois", "%.2f" % inv_yhat2[0]])
	writer.writerow(["Tres", "%.2f" % inv_yhat3[0]])
	writer.writerow(["Quatro", "%.2f" % inv_yhat4[0]])
############################################################################################################################
###########################Criação do gráfico de linha#####################################################################
arq1 = read_csv(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\Previsao\lotacao1.csv', header=0, delimiter=';')
val1 = arq1.values
tam1 = len(arq1)
peso1 = val1[:, -1]
data1 = val1[:, 0]#É necessário o arquivo csv estar com o formato da data em: Ano-Mês-Dia
dia1 = data1[(tam1-12):,]

arq11 = read_csv(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\Previsao\e1_tx.csv', header=0, delimiter=';')
dados11 = arq11.values
tam11 = len(arq11)
val1 = dados11[(tam11-12):, -1]

arq2 = read_csv(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\Previsao\lotacao2.csv', header=0, delimiter=';')
val2 = arq2.values
tam2 = len(arq2)
peso2 = val2[:, -1]
data2 = val2[:, 0]#É necessário o arquivo csv estar com o formato da data em: Ano-Mês-Dia
dia2 = data2[(tam2-12):,]

arq21 = read_csv(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\Previsao\e2_tx.csv', header=0, delimiter=';')
dados21 = arq21.values
tam21 = len(arq21)
val2 = dados21[(tam21-12):, -1]

arq3 = read_csv(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\Previsao\lotacao3.csv', header=0, delimiter=';')
val3 = arq3.values
tam3 = len(arq3)
peso3 = val3[:, -1]
data3 = val3[:, 0]#É necessário o arquivo csv estar com o formato da data em: Ano-Mês-Dia
dia3 = data3[(tam3-12):,]

arq31 = read_csv(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\Previsao\e3_tx.csv', header=0, delimiter=';')
dados31 = arq31.values
tam31 = len(arq31)
val3 = dados31[(tam31-12):, -1]

arq4 = read_csv(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\Previsao\lotacao4.csv', header=0, delimiter=';')
val4 = arq4.values
tam4 = len(arq4)
peso4 = val4[:, -1]
data4 = val4[:, 0]#É necessário o arquivo csv estar com o formato da data em: Ano-Mês-Dia
dia4 = data4[(tam4-12):,]

arq41 = read_csv(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\Previsao\e4_tx.csv', header=0, delimiter=';')
dados41 = arq41.values
tam41 = len(arq41)
val4 = dados41[(tam41-12):, -1]

p1 = (peso1[(tam1-1)] * 0.12)
p2 = (peso2[(tam2-1)] * 0.12)
p3 = (peso3[(tam3-1)] * 0.12)
p4 = (peso4[(tam4-1)] * 0.12)

a = '2019-11-06'
b = '2020-02-21'

#####Gráfico P20 Infestado#############################################################
fig, ax = pyplot.subplots()
pyplot.ylabel('kg/ha')
fig.autofmt_xdate()
ax.plot(dia1, val1, label='Real P20 Inf', linestyle='--', marker='o')
ax.plot(a, inv_yhat1[0], label='Predito P20 Inf', linestyle='--', marker='o')
#pyplot.plot_date(dia1, peso1[(tam1-12):], label='Taxa de lotação', linestyle='--', marker='o')#[(tam-12):] para pegar o 12 últimos dados da coluna dias
pyplot.legend()
dpi = fig.get_dpi()
pyplot.savefig(cmh + '\P20Inf_prev.png', dpi=dpi*2)
pyplot.close()
#####Gráfico P20 Mirapasto#############################################################
fig, ax = pyplot.subplots()
pyplot.ylabel('kg/ha')
fig.autofmt_xdate()
ax.plot(dia2, val2, label='Real P20 MIRA', linestyle='--', marker='o')
ax.plot(b, inv_yhat2[0], label='Predito P20 MIRA', linestyle='--', marker='o')
#pyplot.plot_date(dia2, peso2[(tam2-12):], label='Taxa de lotação', linestyle='--', marker='o')#[(tam-12):] para pegar o 12 últimos dados da coluna dias
pyplot.legend()
dpi = fig.get_dpi()
pyplot.savefig(cmh + '\P20Mira_prev.png', dpi=dpi*2)
pyplot.close()
#####Gráfico P21 Infestado#############################################################
fig, ax = pyplot.subplots()
pyplot.ylabel('kg/ha')
fig.autofmt_xdate()
ax.plot(dia3, val3, label='Real P21 Inf', linestyle='--', marker='o')
ax.plot(a, inv_yhat3[0], label='Predito P21 Inf', linestyle='--', marker='o')
#pyplot.plot_date(dia3, peso3[(tam3-12):], label='Taxa de lotação', linestyle='--', marker='o')#[(tam-12):] para pegar o 12 últimos dados da coluna dias
pyplot.legend()
dpi = fig.get_dpi()
pyplot.savefig(cmh + '\P21Inf_prev.png', dpi=dpi*2)
pyplot.close()
#####Gráfico P21 Mirapasto#############################################################
fig, ax = pyplot.subplots()
pyplot.ylabel('kg/ha')
fig.autofmt_xdate()
ax.plot(dia4, val4, label='Real P21 MIRA', linestyle='--', marker='o')
ax.plot(b, inv_yhat4[0], label='Predito P21 MIRA', linestyle='--', marker='o')
#pyplot.plot_date(dia4, peso4[(tam4-12):], label='Taxa de lotação', linestyle='--', marker='o')#[(tam-12):] para pegar o 12 últimos dados da coluna dias
pyplot.legend()
dpi = fig.get_dpi()
pyplot.savefig(cmh + '\P21Mira_prev.png', dpi=dpi*2)
pyplot.close()
###########################Criação do mapa de vegetação####################################################################
caminho = r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\Previsao\Resultados'
dados = read_csv(r"C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\Previsao\saida_rede.csv", header=0, delimiter=';')
poligono = r'C:\Users\Ânderson Fischoeder\Desktop\img_sentinel\geojson\areatotal.geojson'

colunas = dados.set_index('Id')['TX']

m = folium.Map([-31.322331, -53.986178], zoom_start=16)

tile = folium.TileLayer(
		tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
		attr = 'Esri',
		name = 'Esri Satellite',
		overlay = False,
		control = True
	   ).add_to(m)

linear = cmp.LinearColormap(
 ['Red', 'Yellow', 'Green'],
 vmin=0, vmax=150,
 caption='Taxa de Acúmulo' 
)

folium.GeoJson(
	poligono,
	style_function=lambda feature: {
		'fillColor': linear(colunas[feature['properties']['Name']]),
		'color': 'black',
		'weight': 1,   
		'fillOpacity': 1,
		'lineOpacity': 0.2,
	}
).add_to(m)
linear.add_to(m)

arq11 = read_csv(r"C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\Previsao\e1.csv", header=0, delimiter=';')
dados11 = arq11.values
tam11 = len(arq11)
arq21 = read_csv(r"C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\Previsao\e2.csv", header=0, delimiter=';')
dados21 = arq21.values
tam21 = len(arq21)
arq31 = read_csv(r"C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\Previsao\e3.csv", header=0, delimiter=';')
dados31 = arq31.values
tam31 = len(arq31)
arq41 = read_csv(r"C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\Previsao\e4.csv", header=0, delimiter=';')
dados41 = arq41.values
tam41 = len(arq41)

print(p1, dados11[(tam11-1):, -1] + inv_yhat1)
print(p2, dados21[(tam21-1):, -1] + inv_yhat2)
print(p3, dados31[(tam31-1):, -1] + inv_yhat3)
print(p4, dados41[(tam41-1):, -1] + inv_yhat4)

if p1 == (dados11[(tam11-1):, -1] + inv_yhat1):
	mark1 = folium.CircleMarker(location=[-31.323459, -53.988220],
							radius = 6,
							popup="Manter animais",
							fill=False, # Set fill to True
							fill_color='yellow',
							color = 'yellow',
							fill_opacity=0.7)
	m.add_child(mark1)
elif p1 < (dados11[(tam11-1):, -1] + inv_yhat1):
	mark1 = folium.CircleMarker(location=[-31.323459, -53.988220],
							radius = 6,
							popup="Inserir animais",
							fill=False, # Set fill to True
							fill_color='lightgreen',
							color = 'lightgreen',
							fill_opacity=0.7)
	m.add_child(mark1)
else:
	mark1 = folium.CircleMarker(location=[-31.323459, -53.988220],
							radius = 6,
							popup="Retirar animais",
							fill=False, # Set fill to True
							fill_color='red',
							color = 'red',
							fill_opacity=0.7)
	m.add_child(mark1)

if p2 == (dados21[(tam21-1):, -1] + inv_yhat2):
	mark2 = folium.CircleMarker(location=[-31.323239, -53.983092],
							radius = 6,
							popup="Manter animais",
							fill=False, # Set fill to True
							fill_color='yellow',
							color = 'yellow',
							fill_opacity=0.7)
	m.add_child(mark2)
elif p2 < (dados21[(tam21-1):, -1] + inv_yhat2):
	mark2 = folium.CircleMarker(location=[-31.323239, -53.983092],
							radius = 6,
							popup="Inserir animais",
							fill=False, # Set fill to True
							fill_color='lightgreen',
							color = 'lightgreen',
							fill_opacity=0.7)
	m.add_child(mark2)
else:
	mark2 = folium.CircleMarker(location=[-31.323239, -53.983092],
							radius = 6,
							popup="Retirar animais",
							fill=False, # Set fill to True
							fill_color='red',
							color = 'red',
							fill_opacity=0.7)
	m.add_child(mark2)

if p3 == (dados31[(tam31-1):, -1] + inv_yhat3):
	mark3 = folium.CircleMarker(location=[-31.319280, -53.988660],
							radius = 6,
							popup="Manter animais",
							fill=False, # Set fill to True
							fill_color='yellow',
							color = 'yellow',
							fill_opacity=0.7)
	m.add_child(mark3)
elif p3 < (dados31[(tam31-1):, -1] + inv_yhat3):
	mark3 = folium.CircleMarker(location=[-31.319280, -53.988660],
							radius = 6,
							popup="Inserir animais",
							fill=False, # Set fill to True
							fill_color='lightgreen',
							color = 'lightgreen',
							fill_opacity=0.7)
	m.add_child(mark3)
else:
	mark3 = folium.CircleMarker(location=[-31.319280, -53.988660],
							radius = 6,
							popup="Retirar animais",
							fill=False, # Set fill to True
							fill_color='red',
							color = 'red',
							fill_opacity=0.7)
	m.add_child(mark3)

if p4 == (dados41[(tam41-1):, -1] + inv_yhat4):
	mark4 = folium.CircleMarker(location=[-31.320114, -53.983263],
							radius = 6,
							popup="Manter animais",
							fill=False, # Set fill to True
							fill_color='yellow',
							color = 'yellow',
							fill_opacity=0.7)
	m.add_child(mark4)
elif p4 < (dados41[(tam41-1):, -1] + inv_yhat4):
	mark4 = folium.CircleMarker(location=[-31.320114, -53.983263],
							radius = 6,
							popup="Inserir animais",
							fill=False, # Set fill to True
							fill_color='lightgreen',
							color = 'lightgreen',
							fill_opacity=0.7)
	m.add_child(mark4)
else:
	mark4 = folium.CircleMarker(location=[-31.320114, -53.983263],
							radius = 6,
							popup="Retirar animais",
							fill=False, # Set fill to True
							fill_color='red',
							color = 'red',
							fill_opacity=0.7)
	m.add_child(mark4)


Text1 = folium.Marker(location=[-31.323510, -53.987188,], 
			icon=DivIcon(
			icon_size=(150,36),
			icon_anchor=(0,0),
			html= "%.2f" % dados.values[0][1] + " kg.ha⁻¹.dia⁻¹"
			))
m.add_child(Text1)

Text2 = folium.Marker(location=[-31.322100, -53.986018], 
			icon=DivIcon(
			icon_size=(150,36),
			icon_anchor=(0,0),
			html= "%.2f" % dados.values[1][1] + " kg.ha⁻¹.dia⁻¹"
			))
m.add_child(Text2)

Text3 = folium.Marker(location=[-31.320385, -53.986242], 
			icon=DivIcon(
			icon_size=(150,36),
			icon_anchor=(0,0),
			html= "%.2f" % dados.values[2][1] + " kg.ha⁻¹.dia⁻¹"
			))
m.add_child(Text3)

Text4 = folium.Marker(location=[-31.317837, -53.986891], 
			icon=DivIcon(
			icon_size=(150,36),
			icon_anchor=(0,0),
			html= "%.2f" % dados.values[3][1] + " kg.ha⁻¹.dia⁻¹"
			))
m.add_child(Text4)

m.save(os.path.join(caminho, 'Resultado.html'))

url = r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\Previsao\Resultados\Resultado.html'

webbrowser.open(url,new=2)	  
