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

caminho = r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\Previsao\Nova pasta'
cmh = os.path.join(caminho, 'Resultado') 
################################################Potreiro 20 Infestado############################################################# 
# load dataset
dataset1 = read_csv('entrada_Inf.csv', header=0, delimiter=';')
values1 = dataset1.values
# integer encode direction
encoder = LabelEncoder()

dataset2 = read_csv('e1_Teste.csv', header=0, delimiter=';')
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

if not os.path.exists('M_INF_PREVISAO_NDVI.h5'):
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
	#model.save('M_INF_PREVISAO_NDVI.h5')
	#------------------
else:
	#------------------
	# load the network's architecture
	model = load_model('M_INF_PREVISAO_NDVI.h5')
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

file = open(r"C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\Previsao\Nova pasta\Repeticoes\ndvi\P20Inf\_1.txt", "w")
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

C = ['2019-09-23', '2019-10-22', '2019-11-06']

############################Criação arquivo csv das saídas################################################################


with open(r"C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\Previsao\saida_rede.csv", 'w', newline='') as file:
	
	writer = csv.writer(file, delimiter=';')
	
	writer.writerow(["Id", "TX"])
	writer.writerow(["Um", "%.2f" % inv_yhat1[0]])
############################################################################################################################
###########################Criação do gráfico de linha#####################################################################
arq1 = read_csv(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\Previsao\Nova pasta\lotacao1.csv', header=0, delimiter=';')
val1 = arq1.values
tam1 = len(arq1)
peso1 = val1[:, -1]
data1 = val1[:, 0]#É necessário o arquivo csv estar com o formato da data em: Ano-Mês-Dia
dia1 = data1[(tam1-2):,]

arq11 = read_csv(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\Previsao\Nova pasta\e1_tx.csv', header=0, delimiter=';')
dados11 = arq11.values
tam11 = len(arq11)
val1 = dados11[(tam11-2):, -1]

p1 = (peso1[(tam1-1)] * 0.12)

#####Gráfico P20 Infestado#############################################################
fig, ax = pyplot.subplots()
pyplot.ylabel('kg/ha/dia')
fig.autofmt_xdate()
ax.plot(dia1, val1, label='Real P20 Inf', linestyle='--', marker='o')
ax.plot(C, inv_yhat1, label='Predito P20 Inf', linestyle='--', marker='o')
#pyplot.plot_date(dia1, peso1[(tam1-12):], label='Taxa de lotação', linestyle='--', marker='o')#[(tam-12):] para pegar o 12 últimos dados da coluna dias
pyplot.legend()
dpi = fig.get_dpi()
pyplot.savefig(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\Previsao\Nova pasta\Repeticoes\ndvi\P20Inf\Treinamentos_1.png', dpi=dpi*2)
pyplot.close()	  
