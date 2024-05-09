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
log.write("------Uniao ModeloOriginal Excecução iniciada em: " + str(now) + "\n\n")

caminho = r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\ModeloAjustado\Repeticoes\Naylor'
cmh = os.path.join(caminho, 'P21Inf') 
################################################Potreiro 20 Infestado############################################################# 
# load dataset
dataset1 = read_csv('e3.csv', header=0, index_col=0, delimiter=';')
values1 = dataset1.values
# integer encode direction
encoder = LabelEncoder()

#Drop NA Values
values1=values1[~np.isnan(values1).any(axis=1)]


# ensure all data is float
values1 = values1.astype('float32')


real1 = values1[:, -1]

# normalize features
scaler = MinMaxScaler()
scaled1 = scaler.fit_transform(values1)

scaled1 = DataFrame(scaled1)

#print(scaled)
#print(scaled2) 
# split into train and test sets
values1 = scaled1.values

#n_train = 24
n_train = 31
train1 = values1[:n_train, :]
test1 = values1[n_train:, :]
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


coeffs1 = np.polyfit(inv_y1, inv_yhat1, 5)
p1 = np.poly1d(coeffs1)
# fit values, and mean
yhat1 = p1(inv_y1)							  # or [p(z) for z in x]
ybar1 = np.sum(inv_yhat1)/len(inv_yhat1)		   # or sum(y)/len(y)
ssreg1 = np.sum((yhat1-ybar1)**2)			   # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot1 = np.sum((inv_yhat1 - ybar1)**2)			 # or sum([ (yi - ybar)**2 for yi in y])
r1 = ssreg1 / sstot1
################################################Potreiro 20 Infestado############################################################# 
now = datetime.now()

log = open('log.txt', 'a')
log.write("------ Excecução P20 Infestado encerrada em: " + str(now) + "\n\n")

now = datetime.now()

log = open('log.txt', 'a')
log.write("------ Excecução P20 Mirapasto iniciada em: " + str(now) + "\n\n")
################################################Potreiro 20 Mirapasto############################################################# 
# load dataset
dataset3 = read_csv('entrada_Inf.csv', header=0, index_col=0, delimiter=';')
values3 = dataset3.values
# integer encode direction
encoder = LabelEncoder()

dataset4 = read_csv('e3_Teste.csv', header=0, index_col=0, delimiter=';')
values4 = dataset4.values
# integer encode direction
encoder = LabelEncoder()

#Drop NA Values
values3=values3[~np.isnan(values3).any(axis=1)]
values4=values4[~np.isnan(values4).any(axis=1)]

# ensure all data is float
values3 = values3.astype('float32')
values4 = values4.astype('float32')

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
if not os.path.exists('M_INF.h5'):
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
	#model.save('M_INF.h5')
	#------------------
else:
	#------------------
	# load the network's architecture
	model = load_model('M_INF.h5')
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

coeffs2 = np.polyfit(inv_y2, inv_yhat2, 5)
p2 = np.poly1d(coeffs2)
# fit values, and mean
yhat2 = p2(inv_y2)							  # or [p(z) for z in x]
ybar2 = np.sum(inv_yhat2)/len(inv_yhat2)		   # or sum(y)/len(y)
ssreg2 = np.sum((yhat2-ybar2)**2)			   # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot2 = np.sum((inv_yhat2 - ybar2)**2)			 # or sum([ (yi - ybar)**2 for yi in y])
r2 = ssreg2 / sstot2
################################################Potreiro 20 Mirapasto#############################################################
now = datetime.now()

log = open('log.txt', 'a')
log.write("------ Excecução P20 Mirapasto encerrada em: " + str(now) + "\n\n")

now = datetime.now()

log = open('log.txt', 'a')
log.write("------ Excecução P21 Infestado iniciada em: " + str(now) + "\n\n")
################################################Potreiro 21 Infestado############################################################# 
# load dataset
dataset5 = read_csv('entrada_TotalNaylor.csv', header=0, index_col=0, delimiter=';')
values5 = dataset5.values
# integer encode direction
encoder = LabelEncoder()

dataset6 = read_csv('e3_Teste.csv', header=0, index_col=0, delimiter=';')
values6 = dataset6.values
# integer encode direction
encoder = LabelEncoder()

#Drop NA Values
values5=values5[~np.isnan(values5).any(axis=1)]
values6=values6[~np.isnan(values6).any(axis=1)]

# ensure all data is float
values5 = values5.astype('float32')
values6 = values6.astype('float32')

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
if not os.path.exists('M_TOTALNAYLOR.h5'):
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
	#model.save('M_TOTALNAYLOR.h5')
	#------------------
else:
	#------------------
	# load the network's architecture
	model = load_model('M_TOTALNAYLOR.h5')
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

coeffs3 = np.polyfit(inv_y3, inv_yhat3, 5)
p3 = np.poly1d(coeffs3)
# fit values, and mean
yhat3 = p3(inv_y3)							  # or [p(z) for z in x]
ybar3 = np.sum(inv_yhat3)/len(inv_yhat3)		   # or sum(y)/len(y)
ssreg3 = np.sum((yhat3-ybar3)**2)			   # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot3 = np.sum((inv_yhat3 - ybar3)**2)			 # or sum([ (yi - ybar)**2 for yi in y])
r3 = ssreg3 / sstot3
################################################Potreiro 21 Infestado#############################################################
now = datetime.now()

log = open('log.txt', 'a')
log.write("------ Excecução P21 Infestado encerrada em: " + str(now) + "\n\n")

now = datetime.now()

log = open('log.txt', 'a')
log.write("------ Excecução P21 Mirapasto iniciada em: " + str(now) + "\n\n")
################################################Potreiro 21 Mirapasto############################################################# 
################################################Potreiro 21 Mirapasto#############################################################
now = datetime.now()

log = open('log.txt', 'a')
log.write("------ Excecução P21 Mirapasto encerrada em: " + str(now) + "\n\n")
###########################Criação do gráfico de linha#####################################################################

#####Gráfico P20 Infestado#############################################################
arq = read_csv(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\ModeloAjustado\lotacao3.csv', header=0, delimiter=';')
dados = arq.values
tam = len(arq)
data = dados[:, 0]#É necessário o arquivo csv estar com o formato da data em: Ano-Mês-Dia
dia = data[(tam-12):,]
fig, ax = pyplot.subplots()
pyplot.ylim(bottom=-50)
pyplot.ylim(top=130)
pyplot.ylabel('kg/ha/dia')
fig.autofmt_xdate()
pyplot.title('P21 - Infestado')
ax.plot(dia, inv_y1, label='Real', linestyle='--', marker='o')
ax.plot(dia, inv_yhat1, label='P', linestyle='--', marker='o')
ax.plot(dia, inv_yhat2, label='T', linestyle='--', marker='o')
ax.plot(dia, inv_yhat3, label='N', linestyle='--', marker='o')
#pyplot.plot_date(dia, peso[(tam-12):], label='Taxa de lotação', linestyle='--', marker='o')#[(tam-12):] para pegar o 12 últimos dados da coluna dias
pyplot.legend()
dpi = fig.get_dpi()
pyplot.savefig(cmh + '\Treinamentos_10.png', dpi=dpi*2)
pyplot.close()

erro = []
for x in inv_yhat1:
	erro = inv_yhat1 - inv_y1
#Boxplot
fig1, ax1 = pyplot.subplots()
pyplot.boxplot([inv_y1, inv_yhat1, erro], labels=['Real', 'Predito', 'Erro'])
pyplot.title('P21 - Infestado')
dpi = fig1.get_dpi()
pyplot.savefig(cmh + '\Boxplot_P_10.png', dpi=dpi*2)
pyplot.close()
#Gráfico de Dispersão
pyplot.scatter(inv_yhat1, inv_y1)
range = [inv_y1.min(), inv_yhat1.max()]
pyplot.xlim(left=-50)
pyplot.xlim(right=130)
pyplot.ylim(bottom=-50)
pyplot.ylim(top=130)
pyplot.plot(range, range, 'red')
pyplot.title('P21 - Infestado - Real x Predito')
pyplot.ylabel('Real')
pyplot.xlabel('Predito')
pyplot.savefig(cmh + '\Dispersao_P_10.png', dpi=dpi*2)
pyplot.close()

print('P21 - Infestado')
print("R2 linear", r_value ** 2)
print("R2 Polinomial:", r1)
print('Test RMSE: %.3f' % rmse)
print("Desvio Real", desvioAmostralreal1) 
print("Variancia Real", varianciaAmostralreal1)
print("Desvio Predito", desvioAmostralpred1) 
print("Variancia Predito", varianciaAmostralpred1)
print('Real')
print(inv_y1)
print('Predito')
print(inv_yhat1)

file = open(cmh + "\_10.txt", "w")
file.write('P21 - Infestado (P)' + '\n')
file.write('Test RMSE:' + '%.3f' % rmse + '\n')
file.write('R2 linear:' + str(r_value ** 2) + '\n')
file.write("R2 Polinomial:" + str(r1) + '\r')
file.write("Desvio Real:" + str(desvioAmostralreal1) + '\n') 
file.write("Variancia Real:" + str(varianciaAmostralreal1) + '\n')
file.write("Desvio Predito:" + str(desvioAmostralpred1) + '\n') 
file.write("Variancia Predito:" + str(varianciaAmostralpred1) + '\n')
file.write('Real' + '\n')
for a in inv_y1:
	file.write(str('%.2f' % a) + ',' + ' ')
file.write('\n' + 'Predito' + '\n')
for b in inv_yhat1:
	file.write(str('%.2f' % b) + ',' + ' ')
file.close()
#####Gráfico P21 Mirapasto#############################################################
erro = []
for x in inv_yhat2:
	erro = inv_yhat2 - inv_y2  
#Boxplot
fig1, ax1 = pyplot.subplots()
pyplot.boxplot([inv_y2, inv_yhat2, erro], labels=['Real', 'Predito', 'Erro'])
pyplot.title('P21 - Infestado')
dpi = fig1.get_dpi()
pyplot.savefig(cmh + '\Boxplot_T_10.png', dpi=dpi*2)
pyplot.close()
#Gráfico de Dispersão
pyplot.scatter(inv_yhat2, inv_y2)
range = [inv_y2.min(), inv_yhat2.max()]
pyplot.xlim(left=-50)
pyplot.xlim(right=130)
pyplot.ylim(bottom=-50)
pyplot.ylim(top=130)
pyplot.plot(range, range, 'red')
pyplot.title('P21 - Infestado - Real x Predito')
pyplot.ylabel('Real')
pyplot.xlabel('Predito')
pyplot.savefig(cmh + '\Dispersao_T_10.png', dpi=dpi*2)
pyplot.close()

print('P21 - Infestado')
print("R2 linear", r_value2 ** 2)
print("R2 Polinomial:", r2)
print('Test RMSE: %.3f' % rmse2)
print("Desvio Real", desvioAmostralreal2) 
print("Variancia Real", varianciaAmostralreal2)
print("Desvio Predito", desvioAmostralpred2) 
print("Variancia Predito", varianciaAmostralpred2)
print('Real')
print(inv_y2)
print('Predito')
print(inv_yhat2)

file = open(cmh + "\_10.txt", "a")
file.write('\n\r' + 'P21 - Infestado (T)' + '\n')
file.write('Test RMSE:' + '%.3f' % rmse2 + '\n')
file.write('R2 linear:' + str(r_value2 ** 2) + '\n')
file.write("R2 Polinomial:" + str(r2) + '\r')
file.write("Desvio Real:" + str(desvioAmostralreal2) + '\n') 
file.write("Variancia Real:" + str(varianciaAmostralreal2) + '\n')
file.write("Desvio Predito:" + str(desvioAmostralpred2) + '\n') 
file.write("Variancia Predito:" + str(varianciaAmostralpred2) + '\n')
file.write('Real' + '\n')
for a in inv_y2:
	file.write(str('%.2f' % a) + ',' + ' ')
file.write('\n' + 'Predito' + '\n')
for b in inv_yhat2:
	file.write(str('%.2f' % b) + ',' + ' ')
file.close()
#####Gráfico P21 Infestado#############################################################
erro = []
for x in inv_yhat3:
	erro = inv_yhat3 - inv_y3
#Boxplot
fig1, ax1 = pyplot.subplots()
pyplot.boxplot([inv_y3, inv_yhat3, erro], labels=['Real', 'Predito', 'Erro'])
pyplot.title('P21 - Infestado')
dpi = fig1.get_dpi()
pyplot.savefig(cmh + '\Boxplot_N_10.png', dpi=dpi*2)
pyplot.close()
#Gráfico de Dispersão
pyplot.scatter(inv_yhat3, inv_y3)
range = [inv_y3.min(), inv_yhat3.max()]
pyplot.xlim(left=-50)
pyplot.xlim(right=130)
pyplot.ylim(bottom=-50)
pyplot.ylim(top=130)
pyplot.plot(range, range, 'red')
pyplot.title('P21 - Infestado - Real x Predito')
pyplot.ylabel('Real')
pyplot.xlabel('Predito')
pyplot.savefig(cmh + '\Dispersao_N_10.png', dpi=dpi*2)
pyplot.close()

print('P21 - Infestado')
print("R2 linear", r_value3 ** 2)
print("R2 Polinomial:", r3)
print('Test RMSE: %.3f' % rmse3)
print("Desvio Real", desvioAmostralreal3) 
print("Variancia Real", varianciaAmostralreal3)
print("Desvio Predito", desvioAmostralpred3) 
print("Variancia Predito", varianciaAmostralpred3)
print('Real')
print(inv_y3)
print('Predito')
print(inv_yhat3)

file = open(cmh + "\_10.txt", "a")
file.write('\n\r' + 'P21 - Infestado (N)' + '\n')
file.write('Test RMSE:' + '%.3f' % rmse3 + '\n')
file.write('R2 linear:' + str(r_value3 ** 2) + '\n')
file.write("R2 Polinomial:" + str(r3) + '\r')
file.write("Desvio Real:" + str(desvioAmostralreal3) + '\n') 
file.write("Variancia Real:" + str(varianciaAmostralreal3) + '\n')
file.write("Desvio Predito:" + str(desvioAmostralpred3) + '\n') 
file.write("Variancia Predito:" + str(varianciaAmostralpred3) + '\n')
file.write('Real' + '\n')
for a in inv_y3:
	file.write(str('%.2f' % a) + ',' + ' ')
file.write('\n' + 'Predito' + '\n')
for b in inv_yhat3:
	file.write(str('%.2f' % b) + ',' + ' ')
file.close()
############################################################################################################################
now = datetime.now()

log = open('log.txt', 'a')
log.write("------ Excecução Total encerrada em: " + str(now) + "\n\n")