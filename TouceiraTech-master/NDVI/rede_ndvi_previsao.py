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
from scipy import stats
from sklearn.metrics import r2_score
from datetime import datetime

now = datetime.now()

log = open('log.txt', 'a')
log.write("------ Excecução iniciada em: " + str(now) + "\n\n")

caminho = r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\NDVI'
cmh = os.path.join(caminho, 'NDVI_previsao') 
################################################Potreiro 20 Infestado############################################################# 
# load dataset
dataset1 = read_csv('entrada_Inf_ndvi.csv', header=0, delimiter=';')
values1 = dataset1.values
# integer encode direction
encoder = LabelEncoder()

dataset2 = read_csv('entclima_ndvi1.csv', header=0, delimiter=';')
values2 = dataset2.values
# integer encode direction
encoder = LabelEncoder()

#Drop NA Values
values1=values1[~np.isnan(values1).any(axis=1)]
values2=values2[~np.isnan(values2).any(axis=1)]

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
values1 = scaled1.values
values2 = scaled2.values
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
#print(test_X1)
#print(test_y1)

# reshape input to be 3D [samples, timesteps, features]
train_X1 = train_X1.reshape((train_X1.shape[0], 1, train_X1.shape[1]))
test_X1 = test_X1.reshape((test_X1.shape[0], 1, test_X1.shape[1]))

#print(train_X)
#print(train_y)
#print(test_X)
#print(test_y)

print(train_X1.shape, train_y1.shape, test_X1.shape, test_y1.shape)
print(train_X1.shape[1])

if not os.path.exists('M_INF_NDVI_PREVISAO.h5'):
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
	#model.save('M_INF_NDVI_PREVISAO.h5')
	#------------------
else:
	#------------------
	# load the network's architecture
	model = load_model('M_INF_NDVI_PREVISAO.h5')
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
inv_y1 = scaler.inverse_transform(inv_y1)
inv_y1 = inv_y1[:,-1]

print(model.summary())

from keras.utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#from ann_visualizer.visualize import ann_viz

#ann_viz(model, title="My first neural network")

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y1, inv_yhat1))
desvioAmostralpred1 = np.std(inv_yhat1) #desvio padrão populacional
varianciaAmostralpred1 = inv_yhat1.var() #variancia populacional

desvioAmostralreal1 = np.std(inv_y1) #desvio padrão populacional
varianciaAmostralreal1 = inv_y1.var() #variancia populacional

slope, intercept, r_value, p_value, std_err = stats.linregress(inv_y1, inv_yhat1)

'''
coeffs1 = np.polyfit(inv_y1, inv_yhat1, 5)
p1 = np.poly1d(coeffs1)
# fit values, and mean
yhat1 = p1(inv_y1)							  # or [p(z) for z in x]
ybar1 = np.sum(inv_yhat1)/len(inv_yhat1)		   # or sum(y)/len(y)
ssreg1 = np.sum((yhat1-ybar1)**2)			   # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot1 = np.sum((inv_yhat1 - ybar1)**2)			 # or sum([ (yi - ybar)**2 for yi in y])
r1 = ssreg1 / sstot1'''
################################################Potreiro 20 Infestado############################################################# 
now = datetime.now()

log = open('log.txt', 'a')
log.write("------ Excecução P20 Infestado encerrada em: " + str(now) + "\n\n")

now = datetime.now()

log = open('log.txt', 'a')
log.write("------ Excecução P20 Mirapasto iniciada em: " + str(now) + "\n\n")
################################################Potreiro 20 Mirapasto############################################################# 
# load dataset
dataset3 = read_csv('entrada_Mira_ndvi.csv', header=0, delimiter=';')
values3 = dataset3.values
# integer encode direction
encoder = LabelEncoder()

dataset4 = read_csv('entclima_ndvi2.csv', header=0, delimiter=';')
values4 = dataset4.values
# integer encode direction
encoder = LabelEncoder()

#Drop NA Values
values3=values3[~np.isnan(values3).any(axis=1)]
values4=values4[~np.isnan(values4).any(axis=1)]

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
values3 = scaled3.values
values4 = scaled4.values
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
if not os.path.exists('M_MIRA_NDVI_PREVISAO.h5'):
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
	#model.save('M_MIRA_NDVI_PREVISAO.h5')
	#------------------
else:
	#------------------
	# load the network's architecture
	model = load_model('M_MIRA_NDVI_PREVISAO.h5')
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
inv_y2 = scaler.inverse_transform(inv_y2)
inv_y2 = inv_y2[:,-1]

print(model.summary())

from keras.utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#from ann_visualizer.visualize import ann_viz

#ann_viz(model, title="My first neural network")

# calculate RMSE
rmse2 = sqrt(mean_squared_error(inv_y2, inv_yhat2))
desvioAmostralpred2 = np.std(inv_yhat2) #desvio padrão populacional
varianciaAmostralpred2 = inv_yhat2.var() #variancia populacional

desvioAmostralreal2 = np.std(inv_y2) #desvio padrão populacional
varianciaAmostralreal2 = inv_y2.var() #variancia populacional

slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(inv_y2, inv_yhat2)
'''
coeffs2 = np.polyfit(inv_y2, inv_yhat2, 5)
p2 = np.poly1d(coeffs2)
# fit values, and mean
yhat2 = p2(inv_y2)							  # or [p(z) for z in x]
ybar2 = np.sum(inv_yhat2)/len(inv_yhat2)		   # or sum(y)/len(y)
ssreg2 = np.sum((yhat2-ybar2)**2)			   # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot2 = np.sum((inv_yhat2 - ybar2)**2)			 # or sum([ (yi - ybar)**2 for yi in y])
r2 = ssreg2 / sstot2'''
################################################Potreiro 20 Mirapasto#############################################################
now = datetime.now()

log = open('log.txt', 'a')
log.write("------ Excecução P20 Mirapasto encerrada em: " + str(now) + "\n\n")

now = datetime.now()

log = open('log.txt', 'a')
log.write("------ Excecução P21 Infestado iniciada em: " + str(now) + "\n\n")
################################################Potreiro 21 Infestado############################################################# 
# load dataset
dataset5 = read_csv('entrada_Inf_ndvi.csv', header=0, delimiter=';')
values5 = dataset5.values
# integer encode direction
encoder = LabelEncoder()

dataset6 = read_csv('entclima_ndvi3.csv', header=0, delimiter=';')
values6 = dataset6.values
# integer encode direction
encoder = LabelEncoder()

#Drop NA Values
values5=values5[~np.isnan(values5).any(axis=1)]
values6=values6[~np.isnan(values6).any(axis=1)]

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
values5 = scaled5.values
values6 = scaled6.values
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
if not os.path.exists('M_INF_NDVI_PREVISAO.h5'):
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
	#model.save('M_INF_NDVI_PREVISAO.h5')
	#------------------
else:
	#------------------
	# load the network's architecture
	model = load_model('M_INF_NDVI_PREVISAO.h5')
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
inv_y3 = scaler.inverse_transform(inv_y3)
inv_y3 = inv_y3[:,-1]

print(model.summary())

from keras.utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#from ann_visualizer.visualize import ann_viz

#ann_viz(model, title="My first neural network")

# calculate RMSE
rmse3 = sqrt(mean_squared_error(inv_y3, inv_yhat3))
desvioAmostralpred3 = np.std(inv_yhat3) #desvio padrão populacional
varianciaAmostralpred3 = inv_yhat3.var() #variancia populacional

desvioAmostralreal3 = np.std(inv_y3) #desvio padrão populacional
varianciaAmostralreal3 = inv_y3.var() #variancia populacional

slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(inv_y3, inv_yhat3)
'''
coeffs3 = np.polyfit(inv_y3, inv_yhat3, 5)
p3 = np.poly1d(coeffs3)
# fit values, and mean
yhat3 = p3(inv_y3)							  # or [p(z) for z in x]
ybar3 = np.sum(inv_yhat3)/len(inv_yhat3)		   # or sum(y)/len(y)
ssreg3 = np.sum((yhat3-ybar3)**2)			   # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot3 = np.sum((inv_yhat3 - ybar3)**2)			 # or sum([ (yi - ybar)**2 for yi in y])
r3 = ssreg3 / sstot3'''
################################################Potreiro 21 Infestado#############################################################
now = datetime.now()

log = open('log.txt', 'a')
log.write("------ Excecução P21 Infestado encerrada em: " + str(now) + "\n\n")

now = datetime.now()

log = open('log.txt', 'a')
log.write("------ Excecução P21 Mirapasto iniciada em: " + str(now) + "\n\n")
################################################Potreiro 21 Mirapasto############################################################# 
# load dataset
dataset7 = read_csv('entrada_Mira_ndvi.csv', header=0, delimiter=';')
values7 = dataset7.values
# integer encode direction
encoder = LabelEncoder()

dataset8 = read_csv('entclima_ndvi4.csv', header=0, delimiter=';')
values8 = dataset8.values
# integer encode direction
encoder = LabelEncoder()

#Drop NA Values
values7=values7[~np.isnan(values7).any(axis=1)]
values8=values8[~np.isnan(values8).any(axis=1)]

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
values7 = scaled7.values
values8 = scaled8.values
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
if not os.path.exists('M_MIRA_NDVI_PREVISAO.h5'):
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
	#model.save('M_MIRA_NDVI_PREVISAO.h5')
	#------------------
else:
	#------------------
	# load the network's architecture
	model = load_model('M_MIRA_NDVI_PREVISAO.h5')
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
inv_y4 = scaler.inverse_transform(inv_y4)
inv_y4 = inv_y4[:,-1]

print(model.summary())

from keras.utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#from ann_visualizer.visualize import ann_viz

#ann_viz(model, title="My first neural network")

# calculate RMSE
rmse4 = sqrt(mean_squared_error(inv_y4, inv_yhat4))
desvioAmostralpred4 = np.std(inv_yhat4) #desvio padrão populacional
varianciaAmostralpred4 = inv_yhat4.var() #variancia populacional

desvioAmostralreal4 = np.std(inv_y4) #desvio padrão populacional
varianciaAmostralreal4 = inv_y4.var() #variancia populacional

slope4, intercept4, r_value4, p_value4, std_err4 = stats.linregress(inv_y4, inv_yhat4)
'''
coeffs4 = np.polyfit(inv_y4, inv_yhat4, 5)
p4 = np.poly1d(coeffs4)
# fit values, and mean
yhat4 = p4(inv_y4)							  # or [p(z) for z in x]
ybar4 = np.sum(inv_yhat4)/len(inv_yhat4)		   # or sum(y)/len(y)
ssreg4 = np.sum((yhat4-ybar4)**2)			   # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot4 = np.sum((inv_yhat4 - ybar4)**2)			 # or sum([ (yi - ybar)**2 for yi in y])
r4 = ssreg4 / sstot4'''
################################################Potreiro 21 Mirapasto#############################################################
now = datetime.now()

log = open('log.txt', 'a')
log.write("------ Excecução P21 Mirapasto encerrada em: " + str(now) + "\n\n")
###########################Criação do gráfico de linha#####################################################################

#####Gráfico P20 Infestado#############################################################
arq = read_csv(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\NDVI\e1.csv', header=0, delimiter=';')
dados = arq.values
tam = len(arq)
data = dados[:, 0]#É necessário o arquivo csv estar com o formato da data em: Ano-Mês-Dia
dia = data[(tam-4):,]

arq1 = read_csv(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\NDVI\e1_tx.csv', header=0, delimiter=';')
dados1 = arq1.values
tam1 = len(arq1)
val = dados1[(tam1-4):, -1]
a = '2019-11-06'
fig, ax = pyplot.subplots()
pyplot.ylim(bottom=-50)
pyplot.ylim(top=130)
pyplot.ylabel('kg/ha/dia')
fig.autofmt_xdate()
ax.plot(dia, val, label='Real P20 Inf', linestyle='--', marker='o')
ax.plot(a, inv_yhat1, label='Predito P20 Inf', linestyle='--', marker='o')
#pyplot.plot_date(dia, peso[(tam-12):], label='Taxa de lotação', linestyle='--', marker='o')#[(tam-12):] para pegar o 12 últimos dados da coluna dias
pyplot.legend()
dpi = fig.get_dpi()
pyplot.savefig(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\NDVI\Repeticoes\Futuro\P20Inf\Treinamentos_1.png', dpi=dpi*2)
pyplot.close()

erro = []
for x in inv_yhat1:
	erro = inv_yhat1 - inv_y1
#Boxplot
fig1, ax1 = pyplot.subplots()
pyplot.boxplot([inv_y1, inv_yhat1, erro], labels=['Real', 'Predito', 'Erro'])
pyplot.title('P20 - Infestado')
dpi = fig1.get_dpi()
pyplot.savefig(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\NDVI\Repeticoes\Futuro\P20Inf\Boxplot_1.png', dpi=dpi*2)
pyplot.close()
#Gráfico de Dispersão
pyplot.scatter(inv_y1, inv_yhat1)
range = [inv_y1.min(), inv_yhat1.max()]
pyplot.xlim(left=-50)
pyplot.xlim(right=130)
pyplot.ylim(bottom=-50)
pyplot.ylim(top=130)
pyplot.plot(range, range, 'red')
pyplot.title('P20 - Infestado - Real x Predito')
pyplot.ylabel('Predito')
pyplot.xlabel('Real')
pyplot.savefig(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\NDVI\Repeticoes\Futuro\P20Inf\Dispersao_1.png', dpi=dpi*2)
pyplot.close()

print('P20 - Infestado')
print('Real')
print(inv_y1.round(1))
print('Predito')
print(inv_yhat1.round(1))

file = open(r"C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\NDVI\Repeticoes\Futuro\P20Inf\_1.txt", "w")
file.write('P20 - Infestado' + '\n')
file.write('Real' + '\n')
for a in inv_y1:
	file.write(str(a) + ',' + ' ')
file.write('\n' + 'Predito' + '\n')
for b in inv_yhat1:
	file.write(str(b) + ',' + ' ')
file.close()
#####Gráfico P20 Mirapasto#############################################################
arq = read_csv(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\NDVI\e2.csv', header=0, delimiter=';')
dados = arq.values
tam = len(arq)
data = dados[:, 0]
dia = data[(tam-6):,]

arq2 = read_csv(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\NDVI\e2_tx.csv', header=0, delimiter=';')
dados2 = arq2.values
tam2 = len(arq2)
val = dados2[(tam2-6):, -1]
a = '2020-02-21'

fig, ax = pyplot.subplots()
pyplot.ylim(bottom=-50)
pyplot.ylim(top=130)
pyplot.ylabel('kg/ha/dia')
fig.autofmt_xdate()
ax.plot(dia, val, label='Real P20 MIRA', linestyle='--', marker='o')
ax.plot(a, inv_yhat2, label='Predito P20 MIRA', linestyle='--', marker='o')
#pyplot.plot_date(dia, peso[(tam-12):], label='Taxa de lotação', linestyle='--', marker='o')#[(tam-12):] para pegar o 12 últimos dados da coluna dias
pyplot.legend()
dpi = fig.get_dpi()
pyplot.savefig(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\NDVI\Repeticoes\Futuro\P20Mira\Treinamentos_1.png', dpi=dpi*2)
pyplot.close()

erro = []
for x in inv_yhat2:
	erro = inv_yhat2 - inv_y2  
#Boxplot
fig1, ax1 = pyplot.subplots()
pyplot.boxplot([inv_y2, inv_yhat2, erro], labels=['Real', 'Predito', 'Erro'])
pyplot.title('P20 - Mirapasto')
dpi = fig1.get_dpi()
pyplot.savefig(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\NDVI\Repeticoes\Futuro\P20Mira\Boxplot_1.png', dpi=dpi*2)
pyplot.close()
#Gráfico de Dispersão
pyplot.scatter(inv_y2, inv_yhat2)
range = [inv_y2.min(), inv_yhat2.max()]
pyplot.xlim(left=-50)
pyplot.xlim(right=130)
pyplot.ylim(bottom=-50)
pyplot.ylim(top=130)
pyplot.plot(range, range, 'red')
pyplot.title('P20 - Mirapasto - Real x Predito')
pyplot.ylabel('Predito')
pyplot.xlabel('Real')
pyplot.savefig(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\NDVI\Repeticoes\Futuro\P20Mira\Dispersao_1.png', dpi=dpi*2)
pyplot.close()

print('P20 - Mirapasto')
print('Real')
print(inv_y2.round(1))
print('Predito')
print(inv_yhat2.round(0))

file = open(r"C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\NDVI\Repeticoes\Futuro\P20Mira\_1.txt", "w")
file.write('P20 - Mirapasto' + '\n')
file.write('Real' + '\n')
for a in inv_y2:
	file.write(str(a) + ',' + ' ')
file.write('\n' + 'Predito' + '\n')
for b in inv_yhat2:
	file.write(str(b) + ',' + ' ')
file.close()
#####Gráfico P21 Infestado#############################################################
arq = read_csv(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\NDVI\e3.csv', header=0, delimiter=';')
dados = arq.values
tam = len(arq)
data = dados[:, 0]
dia = data[(tam-4):,]

arq3 = read_csv(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\NDVI\e3_tx.csv', header=0, delimiter=';')
dados3 = arq3.values
tam3 = len(arq3)
val = dados3[(tam3-4):, -1]
a = '2019-11-06'

fig, ax = pyplot.subplots()
pyplot.ylim(bottom=-50)
pyplot.ylim(top=130)
pyplot.ylabel('kg/ha/dia')
fig.autofmt_xdate()
ax.plot(dia, val, label='Real P21 Inf', linestyle='--', marker='o')
ax.plot(a, inv_yhat3, label='Predito P21 Inf', linestyle='--', marker='o')
#pyplot.plot_date(dia, peso[(tam-12):], label='Taxa de lotação', linestyle='--', marker='o')#[(tam-12):] para pegar o 12 últimos dados da coluna dias
pyplot.legend()
dpi = fig.get_dpi()
pyplot.savefig(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\NDVI\Repeticoes\Futuro\P21Inf\Treinamentos_1.png', dpi=dpi*2)
pyplot.close()


erro = []
for x in inv_yhat3:
	erro = inv_yhat3 - inv_y3
#Boxplot
fig1, ax1 = pyplot.subplots()
pyplot.boxplot([inv_y3, inv_yhat3, erro], labels=['Real', 'Predito', 'Erro'])
pyplot.title('P21 - Infestado')
dpi = fig1.get_dpi()
pyplot.savefig(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\NDVI\Repeticoes\Futuro\P21Inf\Boxplot_1.png', dpi=dpi*2)
pyplot.close()
#Gráfico de Dispersão
pyplot.scatter(inv_y3, inv_yhat3)
range = [inv_y3.min(), inv_yhat3.max()]
pyplot.xlim(left=-50)
pyplot.xlim(right=130)
pyplot.ylim(bottom=-50)
pyplot.ylim(top=130)
pyplot.plot(range, range, 'red')
pyplot.title('P21 - Infestado - Real x Predito')
pyplot.ylabel('Predito')
pyplot.xlabel('Real')
pyplot.savefig(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\NDVI\Repeticoes\Futuro\P21Inf\Dispersao_1.png', dpi=dpi*2)
pyplot.close()

print('P21 - Infestado')
print('Real')
print(inv_y3.round(1))
print('Predito')
print(inv_yhat3.round(1))

file = open(r"C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\NDVI\Repeticoes\Futuro\P21Inf\_1.txt", "w")
file.write('P21 - Infestado' + '\n')
file.write('Real' + '\n')
for a in inv_y3:
	file.write(str(a) + ',' + ' ')
file.write('\n' + 'Predito' + '\n')
for b in inv_yhat3:
	file.write(str(b) + ',' + ' ')
file.close()
#####Gráfico P21 Mirapasto#############################################################
arq = read_csv(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\NDVI\e4.csv', header=0, delimiter=';')
dados = arq.values
tam = len(arq)
data = dados[:, 0]
dia = data[(tam-6):,]

arq4 = read_csv(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\NDVI\e4_tx.csv', header=0, delimiter=';')
dados4 = arq4.values
tam4 = len(arq4)
val = dados4[(tam4-6):, -1]
a = '2020-02-21'

fig, ax = pyplot.subplots()
pyplot.ylim(bottom=-50)
pyplot.ylim(top=130)
pyplot.ylabel('kg/ha/dia')
fig.autofmt_xdate()
ax.plot(dia, val, label='Real P21 MIRA', linestyle='--', marker='o')
ax.plot(a, inv_yhat4, label='Predito P21 MIRA', linestyle='--', marker='o')
#pyplot.plot_date(dia, peso[(tam-12):], label='Taxa de lotação', linestyle='--', marker='o')#[(tam-12):] para pegar o 12 últimos dados da coluna dias
pyplot.legend()
dpi = fig.get_dpi()
pyplot.savefig(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\NDVI\Repeticoes\Futuro\P21Mira\Treinamentos_1.png', dpi=dpi*2)
pyplot.close()

erro = []
for x in inv_yhat4:
	erro = inv_yhat4 - inv_y4
#Boxplot
fig1, ax1 = pyplot.subplots()
pyplot.boxplot([inv_y4, inv_yhat4, erro], labels=['Real', 'Predito', 'Erro'])
pyplot.title('P21 - Mirapasto')
dpi = fig1.get_dpi()
pyplot.savefig(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\NDVI\Repeticoes\Futuro\P21Mira\Boxplot_1.png', dpi=dpi*2)
pyplot.close()
#Gráfico de Dispersão
pyplot.scatter(inv_y4, inv_yhat4)
range = [inv_y4.min(), inv_yhat4.max()]
pyplot.xlim(left=-50)
pyplot.xlim(right=130)
pyplot.ylim(bottom=-50)
pyplot.ylim(top=130)
pyplot.plot(range, range, 'red')
pyplot.title('P21 - Mirapasto - Real x Predito')
pyplot.ylabel('Predito')
pyplot.xlabel('Real')
pyplot.savefig(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\NDVI\Repeticoes\Futuro\P21Mira\Dispersao_1.png', dpi=dpi*2)
pyplot.close()

print('P21 - Mirapasto')
print('Real')
print(inv_y4.round(1))
print('Predito')
print(inv_yhat4.round(1))

file = open(r"C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\NDVI\Repeticoes\Futuro\P21Mira\_1.txt", "w")
file.write('P21 - Mirapasto' + '\n')
file.write('Real' + '\n')
for a in inv_y4:
	file.write(str(a) + ',' + ' ')
file.write('\n' + 'Predito' + '\n')
for b in inv_yhat4:
	file.write(str(b) + ',' + ' ')
file.close()
############################################################################################################################

############################Criação arquivo csv das saídas################################################################


with open(r"C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\NDVI\saida_rede_ndvi.csv", 'w', newline='') as file:
    
    writer = csv.writer(file, delimiter=';')
    
    writer.writerow(["Id", "TX"])
    writer.writerow(["Um", inv_yhat1[0]])
    writer.writerow(["Dois", inv_yhat2[0]])
    writer.writerow(["Tres", inv_yhat3[0]])
    writer.writerow(["Quatro", inv_yhat4[0]])
############################################################################################################################
###########################Criação do gráfico de linha#####################################################################
arq1 = read_csv(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\Previsao\lotacao1.csv', header=0, delimiter=';')
val1 = arq1.values
tam1 = len(arq1)
peso1 = val1[:, -1]
data1 = val1[:, 0]#É necessário o arquivo csv estar com o formato da data em: Ano-Mês-Dia
dia1 = data1[(tam1-12):,]

arq2 = read_csv(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\Previsao\lotacao2.csv', header=0, delimiter=';')
val2 = arq2.values
tam2 = len(arq2)
peso2 = val2[:, -1]
data2 = val2[:, 0]#É necessário o arquivo csv estar com o formato da data em: Ano-Mês-Dia
dia2 = data2[(tam2-12):,]

arq3 = read_csv(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\Previsao\lotacao3.csv', header=0, delimiter=';')
val3 = arq3.values
tam3 = len(arq3)
peso3 = val3[:, -1]
data3 = val3[:, 0]#É necessário o arquivo csv estar com o formato da data em: Ano-Mês-Dia
dia3 = data3[(tam3-12):,]

arq4 = read_csv(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\Previsao\lotacao4.csv', header=0, delimiter=';')
val4 = arq4.values
tam4 = len(arq4)
peso4 = val4[:, -1]
data4 = val4[:, 0]#É necessário o arquivo csv estar com o formato da data em: Ano-Mês-Dia
dia4 = data4[(tam4-12):,]

p1 = (peso1[(tam1-1)] * 0.12)
p2 = (peso2[(tam2-1)] * 0.12)
p3 = (peso3[(tam3-1)] * 0.12)
p4 = (peso4[(tam4-1)] * 0.12)
###########################Criação do mapa de vegetação####################################################################
caminho = r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\NDVI\Repeticoes\Futuro'

dados = read_csv(r"C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\NDVI\saida_rede_ndvi.csv", header=0, delimiter=';')

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

arq11 = read_csv(r"C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\NDVI\e1.csv", header=0, delimiter=';')
dados11 = arq11.values
tam11 = len(arq11)
arq21 = read_csv(r"C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\NDVI\e2.csv", header=0, delimiter=';')
dados21 = arq21.values
tam21 = len(arq21)
arq31 = read_csv(r"C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\NDVI\e3.csv", header=0, delimiter=';')
dados31 = arq31.values
tam31 = len(arq31)
arq41 = read_csv(r"C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\NDVI\e4.csv", header=0, delimiter=';')
dados41 = arq41.values
tam41 = len(arq41)


print(dados11[(tam11-1):, -1])
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

m.save(os.path.join(caminho, '_1.html'))

#url = r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\NDVI\NDVI_previsao\Resultado_ndvi.html'

#webbrowser.open(url,new=2)    

now = datetime.now()

log = open('log.txt', 'a')
log.write("------Previsao NDVI Excecução encerrada em: " + str(now) + "\n\n")