import os
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score
from datetime import datetime

now = datetime.now()

log = open('log.txt', 'a')
log.write("------Instantanea Cristina ModeloAjustado Excecução iniciada em: " + str(now) + "\n\n")

caminho = r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\ModeloAjustado'
cmh = os.path.join(caminho, 'Potreiro_Cristina') 
###################################################Potreiro 20####################################################################
# load dataset
dataset = read_csv('e1_cristina.csv', header=0, index_col=0, delimiter=';')
values = dataset.values
# integer encode direction
encoder = LabelEncoder()

#Drop NA Values
values=values[~np.isnan(values).any(axis=1)]

# ensure all data is float
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

scaled = DataFrame(scaled)
#print(scaled)
 
# split into train and test sets
values = scaled.values
#n_train = 24
n_train = 39
train = values[:n_train, :]
test = values[n_train:, :]

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))


#print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
#print(train_X.shape[1])

# design network
model = Sequential()
#model.add(LSTM(30, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh', return_sequences = True))#bom para 1 e 2
#model.add(LSTM(15, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh', return_sequences = True))#bom para 1 e 2
#model.add(LSTM(7, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh'))


model.add(LSTM(30, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal',  return_sequences = True))#bom para 1 e 2
model.add(LSTM(15, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal'))

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
history = model.fit(train_X, train_y, epochs=5000, batch_size=72, validation_data=(test_X, test_y), verbose=0, shuffle=False)

# plot history
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.show()
 
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((test_X, yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,-1]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_X, test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,-1]

#print(model.summary())

from keras.utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#from ann_visualizer.visualize import ann_viz

#ann_viz(model, title="My first neural network")

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
desvioAmostralpred = np.std(inv_yhat) #desvio padrão populacional
varianciaAmostralpred = inv_yhat.var() #variancia populacional

desvioAmostralreal = np.std(inv_y) #desvio padrão populacional
varianciaAmostralreal = inv_y.var() #variancia populacional

slope, intercept, r_value, p_value, std_err = stats.linregress(inv_y, inv_yhat)

coeffs1 = np.polyfit(inv_y, inv_yhat, 5)
p1 = np.poly1d(coeffs1)
# fit values, and mean
yhat1 = p1(inv_y)                            # or [p(z) for z in x]
ybar1 = np.sum(inv_yhat)/len(inv_yhat)           # or sum(y)/len(y)
ssreg1 = np.sum((yhat1-ybar1)**2)              # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot1 = np.sum((inv_yhat - ybar1)**2)          # or sum([ (yi - ybar)**2 for yi in y])
r1 = ssreg1 / sstot1

erro = []
for x in inv_yhat:
    erro = inv_yhat - inv_y
    
print('P1')
print('Test RMSE: %.3f' % rmse)
print("R2 linear", r_value ** 2)
print("R2 Polinomial:", r1)
print("Desvio Real:", desvioAmostralreal) 
print("Variancia Real:", varianciaAmostralreal)
print("Desvio Predito:", desvioAmostralpred) 
print("Variancia Predito:", varianciaAmostralpred)
print('Real')
print(inv_y.round(1))
print('Predito')
print(inv_yhat.round(1))

file = open(cmh + "\dados.txt", "w")
file.write('P1' + '\n')
file.write('Test RMSE:' + '%.3f' % rmse + '\n')
file.write('R2 linear:' + str(r_value ** 2) + '\n')
file.write("R2 Polinomial:" + str(r1) + '\r')
file.write("Desvio Real:" + str(desvioAmostralreal) + '\n') 
file.write("Variancia Real:" + str(varianciaAmostralreal) + '\n')
file.write("Desvio Predito:" + str(desvioAmostralpred) + '\n') 
file.write("Variancia Predito:" + str(varianciaAmostralpred) + '\n')
file.write('Real' + '\n')
for a in inv_y:
	file.write(str(a) + ',' + ' ')
file.write('\n' + 'Predito' + '\n')
for b in inv_yhat:
	file.write(str(b) + ',' + ' ')
file.close()

fig1, ax1 = pyplot.subplots()
pyplot.boxplot([inv_y, inv_yhat, erro], labels=['Real', 'Predito', 'Erro'])
pyplot.title('P1')
dpi = fig1.get_dpi()
pyplot.savefig(cmh + '\P1_Boxplot_ModeloAjus.png', dpi=dpi*2)
pyplot.close()

pyplot.scatter(inv_y, inv_yhat)
range = [inv_y.min(), inv_yhat.max()]
pyplot.plot(range, range, 'red')
pyplot.title('P1 - Real x Predito')
pyplot.ylabel('Predito')
pyplot.xlabel('Real')
pyplot.savefig(cmh + '\P1_Dispersao_ModeloAjus.png', dpi=dpi*2)
pyplot.close()

#COPY VALUES

C=dataset.index.values[(test_X.shape[0]*-1):]
Cinv_y=inv_y
Cinv_yhat=inv_yhat

########################################################################################################################
# load dataset
dataset = read_csv('e2_cristina.csv', header=0, index_col=0, delimiter=';')
values = dataset.values
# integer encode direction
encoder = LabelEncoder()

#Drop NA Values
values=values[~np.isnan(values).any(axis=1)]

# ensure all data is float
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

scaled = DataFrame(scaled)
#print(scaled)
 
# split into train and test sets
values = scaled.values
n_train = 39
#n_train = 27
train = values[:n_train, :]
test = values[n_train:, :]

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))


#print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
#print(train_X.shape[1])

# design network
model = Sequential()
#model.add(LSTM(30, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh', return_sequences = True))#bom para 1 e 2
#model.add(LSTM(15, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh', return_sequences = True))#bom para 1 e 2
#model.add(LSTM(7, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh'))


model.add(LSTM(30, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal',  return_sequences = True))#bom para 1 e 2
model.add(LSTM(15, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal'))

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
history = model.fit(train_X, train_y, epochs=5000, batch_size=72, validation_data=(test_X, test_y), verbose=0, shuffle=False)

# plot history
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.show()
 
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((test_X, yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,-1]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_X, test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,-1]

#print(model.summary())

from keras.utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#from ann_visualizer.visualize import ann_viz

#ann_viz(model, title="My first neural network")

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
desvioAmostralpred = np.std(inv_yhat) #desvio padrão populacional
varianciaAmostralpred = inv_yhat.var() #variancia populacional

desvioAmostralreal = np.std(inv_y) #desvio padrão populacional
varianciaAmostralreal = inv_y.var() #variancia populacional

slope, intercept, r_value, p_value, std_err = stats.linregress(inv_y, inv_yhat)

coeffs1 = np.polyfit(inv_y, inv_yhat, 5)
p1 = np.poly1d(coeffs1)
# fit values, and mean
yhat1 = p1(inv_y)                            # or [p(z) for z in x]
ybar1 = np.sum(inv_yhat)/len(inv_yhat)           # or sum(y)/len(y)
ssreg1 = np.sum((yhat1-ybar1)**2)              # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot1 = np.sum((inv_yhat - ybar1)**2)          # or sum([ (yi - ybar)**2 for yi in y])
r1 = ssreg1 / sstot1

erro = []
for x in inv_yhat:
    erro = inv_yhat - inv_y
    
print('P2')
print('Test RMSE: %.3f' % rmse)
print("R2 linear", r_value ** 2)
print("R2 Polinomial:", r1)
print("Desvio Real", desvioAmostralreal) 
print("Variancia Real", varianciaAmostralreal)
print("Desvio Predito", desvioAmostralpred) 
print("Variancia Predito", varianciaAmostralpred)
print('Real')
print(inv_y.round(1))
print('Predito')
print(inv_yhat.round(1))

file = open(cmh + "\dados.txt", "a")
file.write('\n\r' + 'P2' + '\n')
file.write('Test RMSE:' + '%.3f' % rmse + '\n')
file.write('R2 linear:' + str(r_value ** 2) + '\n')
file.write("R2 Polinomial:" + str(r1) + '\r')
file.write("Desvio Real:" + str(desvioAmostralreal) + '\n') 
file.write("Variancia Real:" + str(varianciaAmostralreal) + '\n')
file.write("Desvio Predito:" + str(desvioAmostralpred) + '\n') 
file.write("Variancia Predito:" + str(varianciaAmostralpred) + '\n')
file.write('Real' + '\n')
for a in inv_y:
	file.write(str(a) + ',' + ' ')
file.write('\n' + 'Predito' + '\n')
for b in inv_yhat:
	file.write(str(b) + ',' + ' ')
file.close()

fig1, ax1 = pyplot.subplots()
pyplot.boxplot([inv_y, inv_yhat, erro], labels=['Real', 'Predito', 'Erro'])
pyplot.title('P2')
dpi = fig1.get_dpi()
pyplot.savefig(cmh + '\P2_Boxplot_ModeloAjus.png', dpi=dpi*2)
pyplot.close()

pyplot.scatter(inv_y, inv_yhat)
range = [inv_y.min(), inv_yhat.max()]
pyplot.plot(range, range, 'red')
pyplot.title('P2 - Real x Predito')
pyplot.ylabel('Predito')
pyplot.xlabel('Real')
pyplot.savefig(cmh + '\P2_Dispersao_ModeloAjus.png', dpi=dpi*2)
pyplot.close()
########################################################################################################################
arq = read_csv(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\ModeloAjustado\lotacao_cristina_p1.csv', header=0, delimiter=';')
dados = arq.values
tam = len(arq)
data = dados[:, 0]
dia = data[(tam-12):,]

fig, ax = pyplot.subplots()
pyplot.ylabel('kg/ha/dia')
fig.autofmt_xdate()

pyplot.plot_date(dia, Cinv_y, label='Real P1', linestyle='--', marker='o')
pyplot.plot_date(dia, Cinv_yhat, label='Predito P1', linestyle='--', marker='o')
pyplot.legend()
dpi = fig.get_dpi()
pyplot.savefig(cmh + '\P1_ModeloAjus.png', dpi=dpi*2)
pyplot.close()

arq = read_csv(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\ModeloAjustado\lotacao_cristina_p2.csv', header=0, delimiter=';')
dados = arq.values
tam = len(arq)
data = dados[:, 0]
dia = data[(tam-12):,]

fig, ax = pyplot.subplots()
pyplot.ylabel('kg/ha/dia')
fig.autofmt_xdate()
pyplot.plot_date(dia, inv_y, label='Real P2', linestyle='--', marker='o')
pyplot.plot_date(dia, inv_yhat, label='Predito P2', linestyle='--', marker='o')
pyplot.legend()
dpi = fig.get_dpi()
pyplot.savefig(cmh + '\P2_ModeloAjus.png', dpi=dpi*2)
pyplot.close()

#REVERSE GRAPH
#pyplot.plot_date(C[::-1], Cinv_y, label='Real Inf', linestyle='--', marker='o')
#pyplot.plot_date(C[::-1], Cinv_yhat, label='Predito Inf', linestyle='--', marker='o')

#pyplot.plot_date(dataset.index.values[(test_X.shape[0]*-1):][::-1], inv_y[::-1], label='Real', linestyle='--', marker='o')
#pyplot.plot_date(dataset.index.values[(test_X.shape[0]*-1):][::-1], inv_yhat[::-1], label='Predito', linestyle='--', marker='o')

####################################################Potreiro 21##########################################################################################
# load dataset
dataset = read_csv('e3_cristina.csv', header=0, index_col=0, delimiter=';')
values = dataset.values
# integer encode direction
encoder = LabelEncoder()

#Drop NA Values
values=values[~np.isnan(values).any(axis=1)]

# ensure all data is float
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

scaled = DataFrame(scaled)
#print(scaled)
 
# split into train and test sets
values = scaled.values
#n_train = 24
n_train = 39
train = values[:n_train, :]
test = values[n_train:, :]

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))


#print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
#print(train_X.shape[1])

# design network
model = Sequential()
#model.add(LSTM(30, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh', return_sequences = True))#bom para 1 e 2
#model.add(LSTM(15, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh', return_sequences = True))#bom para 1 e 2
#model.add(LSTM(7, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh'))


model.add(LSTM(30, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal',  return_sequences = True))#bom para 1 e 2
model.add(LSTM(15, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal'))

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
history = model.fit(train_X, train_y, epochs=5000, batch_size=72, validation_data=(test_X, test_y), verbose=0, shuffle=False)

# plot history
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.show()
 
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((test_X, yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,-1]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_X, test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,-1]

#print(model.summary())

from keras.utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#from ann_visualizer.visualize import ann_viz

#ann_viz(model, title="My first neural network")

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
desvioAmostralpred = np.std(inv_yhat) #desvio padrão populacional
varianciaAmostralpred = inv_yhat.var() #variancia populacional

desvioAmostralreal = np.std(inv_y) #desvio padrão populacional
varianciaAmostralreal = inv_y.var() #variancia populacional

slope, intercept, r_value, p_value, std_err = stats.linregress(inv_y, inv_yhat)

coeffs1 = np.polyfit(inv_y, inv_yhat, 5)
p1 = np.poly1d(coeffs1)
# fit values, and mean
yhat1 = p1(inv_y)                            # or [p(z) for z in x]
ybar1 = np.sum(inv_yhat)/len(inv_yhat)           # or sum(y)/len(y)
ssreg1 = np.sum((yhat1-ybar1)**2)              # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot1 = np.sum((inv_yhat - ybar1)**2)          # or sum([ (yi - ybar)**2 for yi in y])
r1 = ssreg1 / sstot1

print('P3')
print('Test RMSE: %.3f' % rmse)
print("R2 linear", r_value ** 2)
print("R2 Polinomial:", r1)
print("Desvio Real", desvioAmostralreal) 
print("Variancia Real", varianciaAmostralreal)
print("Desvio Predito", desvioAmostralpred) 
print("Variancia Predito", varianciaAmostralpred)
print('Real')
print(inv_y.round(1))
print('Predito')
print(inv_yhat.round(1))

file = open(cmh + "\dados.txt", "a")
file.write('\n\r' + 'P3' + '\n')
file.write('Test RMSE:' + '%.3f' % rmse + '\n')
file.write('R2 linear:' + str(r_value ** 2) + '\n')
file.write("R2 Polinomial:" + str(r1) + '\r')
file.write("Desvio Real:" + str(desvioAmostralreal) + '\n') 
file.write("Variancia Real:" + str(varianciaAmostralreal) + '\n')
file.write("Desvio Predito:" + str(desvioAmostralpred) + '\n') 
file.write("Variancia Predito:" + str(varianciaAmostralpred) + '\n')
file.write('Real' + '\n')
for a in inv_y:
	file.write(str(a) + ',' +  ' ')
file.write('\n' + 'Predito' + '\n')
for b in inv_yhat:
	file.write(str(b) + ',' + ' ')
file.close()

erro = []
for x in inv_yhat:
    erro = inv_yhat - inv_y

fig1, ax1 = pyplot.subplots()
pyplot.boxplot([inv_y, inv_yhat, erro], labels=['Real', 'Predito', 'Erro'])
pyplot.title('P3')
dpi = fig1.get_dpi()
pyplot.savefig(cmh + '\P3_Boxplot_ModeloAjus.png', dpi=dpi*2)
pyplot.close()

pyplot.scatter(inv_y, inv_yhat)
range = [inv_y.min(), inv_yhat.max()]
pyplot.plot(range, range, 'red')
pyplot.title('P3 - Real x Predito')
pyplot.ylabel('Predito')
pyplot.xlabel('Real')
pyplot.savefig(cmh + '\P3_Dispersao_ModeloAjus.png', dpi=dpi*2)
pyplot.close()



#COPY VALUES

C=dataset.index.values[(test_X.shape[0]*-1):]
Cinv_y=inv_y
Cinv_yhat=inv_yhat

########################################################################################################################
# load dataset
dataset = read_csv('e4_cristina.csv', header=0, index_col=0, delimiter=';')
values = dataset.values
# integer encode direction
encoder = LabelEncoder()

#Drop NA Values
values=values[~np.isnan(values).any(axis=1)]

# ensure all data is float
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

scaled = DataFrame(scaled)
#print(scaled)
 
# split into train and test sets
values = scaled.values
n_train = 39
#n_train = 27
train = values[:n_train, :]
test = values[n_train:, :]

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))


#print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
#print(train_X.shape[1])

# design network
model = Sequential()
#model.add(LSTM(30, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh', return_sequences = True))#bom para 1 e 2
#model.add(LSTM(15, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh', return_sequences = True))#bom para 1 e 2
#model.add(LSTM(7, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh'))


model.add(LSTM(30, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal',  return_sequences = True))#bom para 1 e 2
model.add(LSTM(15, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal'))

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
history = model.fit(train_X, train_y, epochs=5000, batch_size=72, validation_data=(test_X, test_y), verbose=0, shuffle=False)

# plot history
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.show()
 
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((test_X, yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,-1]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_X, test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,-1]

#print(model.summary())

from keras.utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#from ann_visualizer.visualize import ann_viz

#ann_viz(model, title="My first neural network")

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
desvioAmostralpred = np.std(inv_yhat) #desvio padrão populacional
varianciaAmostralpred = inv_yhat.var() #variancia populacional

desvioAmostralreal = np.std(inv_y) #desvio padrão populacional
varianciaAmostralreal = inv_y.var() #variancia populacional

slope, intercept, r_value, p_value, std_err = stats.linregress(inv_y, inv_yhat)

coeffs1 = np.polyfit(inv_y, inv_yhat, 5)
p1 = np.poly1d(coeffs1)
# fit values, and mean
yhat1 = p1(inv_y)                            # or [p(z) for z in x]
ybar1 = np.sum(inv_yhat)/len(inv_yhat)           # or sum(y)/len(y)
ssreg1 = np.sum((yhat1-ybar1)**2)              # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot1 = np.sum((inv_yhat - ybar1)**2)          # or sum([ (yi - ybar)**2 for yi in y])
r1 = ssreg1 / sstot1

print('P4')
print('Test RMSE: %.3f' % rmse)
print("R2 linear", r_value ** 2)
print("R2 Polinomial:", r1)
print("Desvio Real", desvioAmostralreal) 
print("Variancia Real", varianciaAmostralreal)
print("Desvio Predito", desvioAmostralpred) 
print("Variancia Predito", varianciaAmostralpred)
print('Real')
print(inv_y.round(1))
print('Predito')
print(inv_yhat.round(1))

file = open(cmh + "\dados.txt", "a")
file.write('\n\r' + 'P4' + '\n')
file.write('Test RMSE:' + '%.3f' % rmse + '\n')
file.write('R2 linear:' + str(r_value ** 2) + '\n')
file.write("R2 Polinomial:" + str(r1) + '\r')
file.write("Desvio Real:" + str(desvioAmostralreal) + '\n') 
file.write("Variancia Real:" + str(varianciaAmostralreal) + '\n')
file.write("Desvio Predito:" + str(desvioAmostralpred) + '\n') 
file.write("Variancia Predito:" + str(varianciaAmostralpred) + '\n')
file.write('Real' + '\n')
for a in inv_y:
	file.write(str(a) + ',' + ' ')
file.write('\n' + 'Predito' + '\n')
for b in inv_yhat:
	file.write(str(b) + ',' + ' ')
file.close()

erro = []
for x in inv_yhat:
    erro = inv_yhat - inv_y

fig1, ax1 = pyplot.subplots()
pyplot.boxplot([inv_y, inv_yhat, erro], labels=['Real', 'Predito', 'Erro'])
pyplot.title('P4')
dpi = fig1.get_dpi()
pyplot.savefig(cmh + '\P4_Boxplot_ModeloAjus.png', dpi=dpi*2)
pyplot.close()
    
pyplot.scatter(inv_y, inv_yhat)
range = [inv_y.min(), inv_yhat.max()]
pyplot.plot(range, range, 'red')
pyplot.title('P4 - Real x Predito')
pyplot.ylabel('Predito')
pyplot.xlabel('Real')
pyplot.savefig(cmh + '\P4_Dispersao_ModeloAjus.png', dpi=dpi*2)
pyplot.close()

########################################################################################################################
arq = read_csv(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\ModeloAjustado\lotacao_cristina_p3.csv', header=0, delimiter=';')
dados = arq.values
tam = len(arq)
data = dados[:, 0]
dia = data[(tam-12):,]

fig, ax = pyplot.subplots()
pyplot.ylabel('kg/ha/dia')
fig.autofmt_xdate()
pyplot.plot_date(dia, Cinv_y, label='Real P3', linestyle='--', marker='o')
pyplot.plot_date(dia, Cinv_yhat, label='Predito P3', linestyle='--', marker='o')
pyplot.legend()
dpi = fig.get_dpi()
pyplot.savefig(cmh + '\P3_ModeloAjus.png', dpi=dpi*2)
pyplot.close()

arq = read_csv(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\ModeloAjustado\lotacao_cristina_p4.csv', header=0, delimiter=';')
dados = arq.values
tam = len(arq)
data = dados[:, 0]
dia = data[(tam-12):,]

fig, ax = pyplot.subplots()
pyplot.ylabel('kg/ha/dia')
fig.autofmt_xdate()
pyplot.plot_date(dia, inv_y, label='Real P4', linestyle='--', marker='o')
pyplot.plot_date(dia, inv_yhat, label='Predito P4', linestyle='--', marker='o')
pyplot.legend()
dpi = fig.get_dpi()
pyplot.savefig(cmh + '\P4_ModeloAjus.png', dpi=dpi*2)
pyplot.close()

#REVERSE GRAPH
#pyplot.plot_date(C[::-1], Cinv_y, label='Real Inf', linestyle='--', marker='o')
#pyplot.plot_date(C[::-1], Cinv_yhat, label='Predito Inf', linestyle='--', marker='o')

#pyplot.plot_date(dataset.index.values[(test_X.shape[0]*-1):][::-1], inv_y[::-1], label='Real', linestyle='--', marker='o')
#pyplot.plot_date(dataset.index.values[(test_X.shape[0]*-1):][::-1], inv_yhat[::-1], label='Predito', linestyle='--', marker='o')

###################################################Potreiro 20####################################################################
# load dataset
dataset = read_csv('e5_cristina.csv', header=0, index_col=0, delimiter=';')
values = dataset.values
# integer encode direction
encoder = LabelEncoder()

#Drop NA Values
values=values[~np.isnan(values).any(axis=1)]

# ensure all data is float
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

scaled = DataFrame(scaled)
#print(scaled)
 
# split into train and test sets
values = scaled.values
#n_train = 24
n_train = 39
train = values[:n_train, :]
test = values[n_train:, :]

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))


#print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
#print(train_X.shape[1])

# design network
model = Sequential()
#model.add(LSTM(30, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh', return_sequences = True))#bom para 1 e 2
#model.add(LSTM(15, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh', return_sequences = True))#bom para 1 e 2
#model.add(LSTM(7, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh'))


model.add(LSTM(30, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal',  return_sequences = True))#bom para 1 e 2
model.add(LSTM(15, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal'))

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
history = model.fit(train_X, train_y, epochs=5000, batch_size=72, validation_data=(test_X, test_y), verbose=0, shuffle=False)

# plot history
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.show()
 
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((test_X, yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,-1]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_X, test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,-1]

#print(model.summary())

from keras.utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#from ann_visualizer.visualize import ann_viz

#ann_viz(model, title="My first neural network")

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
desvioAmostralpred = np.std(inv_yhat) #desvio padrão populacional
varianciaAmostralpred = inv_yhat.var() #variancia populacional

desvioAmostralreal = np.std(inv_y) #desvio padrão populacional
varianciaAmostralreal = inv_y.var() #variancia populacional

slope, intercept, r_value, p_value, std_err = stats.linregress(inv_y, inv_yhat)

coeffs1 = np.polyfit(inv_y, inv_yhat, 5)
p1 = np.poly1d(coeffs1)
# fit values, and mean
yhat1 = p1(inv_y)                            # or [p(z) for z in x]
ybar1 = np.sum(inv_yhat)/len(inv_yhat)           # or sum(y)/len(y)
ssreg1 = np.sum((yhat1-ybar1)**2)              # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot1 = np.sum((inv_yhat - ybar1)**2)          # or sum([ (yi - ybar)**2 for yi in y])
r1 = ssreg1 / sstot1

erro = []
for x in inv_yhat:
    erro = inv_yhat - inv_y
    
print('P5')
print('Test RMSE: %.3f' % rmse)
print("R2 linear", r_value ** 2)
print("R2 Polinomial:", r1)
print("Desvio Real:", desvioAmostralreal) 
print("Variancia Real:", varianciaAmostralreal)
print("Desvio Predito:", desvioAmostralpred) 
print("Variancia Predito:", varianciaAmostralpred)
print('Real')
print(inv_y.round(1))
print('Predito')
print(inv_yhat.round(1))

file = open(cmh + "\dados.txt", "a")
file.write('\n\r' + 'P5' + '\n')
file.write('Test RMSE:' + '%.3f' % rmse + '\n')
file.write('R2 linear:' + str(r_value ** 2) + '\n')
file.write("R2 Polinomial:" + str(r1) + '\r')
file.write("Desvio Real:" + str(desvioAmostralreal) + '\n') 
file.write("Variancia Real:" + str(varianciaAmostralreal) + '\n')
file.write("Desvio Predito:" + str(desvioAmostralpred) + '\n') 
file.write("Variancia Predito:" + str(varianciaAmostralpred) + '\n')
file.write('Real' + '\n')
for a in inv_y:
	file.write(str(a) + ',' + ' ')
file.write('\n' + 'Predito' + '\n')
for b in inv_yhat:
	file.write(str(b) + ',' + ' ')
file.close()

fig1, ax1 = pyplot.subplots()
pyplot.boxplot([inv_y, inv_yhat, erro], labels=['Real', 'Predito', 'Erro'])
pyplot.title('P5')
dpi = fig1.get_dpi()
pyplot.savefig(cmh + '\P5_Boxplot_ModeloAjus.png', dpi=dpi*2)
pyplot.close()

pyplot.scatter(inv_y, inv_yhat)
range = [inv_y.min(), inv_yhat.max()]
pyplot.plot(range, range, 'red')
pyplot.title('P5 - Real x Predito')
pyplot.ylabel('Predito')
pyplot.xlabel('Real')
pyplot.savefig(cmh + '\P5_Dispersao_ModeloAjus.png', dpi=dpi*2)
pyplot.close()

#COPY VALUES

C=dataset.index.values[(test_X.shape[0]*-1):]
Cinv_y=inv_y
Cinv_yhat=inv_yhat

########################################################################################################################
# load dataset
dataset = read_csv('e6_cristina.csv', header=0, index_col=0, delimiter=';')
values = dataset.values
# integer encode direction
encoder = LabelEncoder()

#Drop NA Values
values=values[~np.isnan(values).any(axis=1)]

# ensure all data is float
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

scaled = DataFrame(scaled)
#print(scaled)
 
# split into train and test sets
values = scaled.values
n_train = 39
#n_train = 27
train = values[:n_train, :]
test = values[n_train:, :]

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))


#print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
#print(train_X.shape[1])

# design network
model = Sequential()
#model.add(LSTM(30, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh', return_sequences = True))#bom para 1 e 2
#model.add(LSTM(15, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh', return_sequences = True))#bom para 1 e 2
#model.add(LSTM(7, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh'))


model.add(LSTM(30, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal',  return_sequences = True))#bom para 1 e 2
model.add(LSTM(15, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal'))

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
history = model.fit(train_X, train_y, epochs=5000, batch_size=72, validation_data=(test_X, test_y), verbose=0, shuffle=False)

# plot history
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.show()
 
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((test_X, yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,-1]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_X, test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,-1]

#print(model.summary())

from keras.utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#from ann_visualizer.visualize import ann_viz

#ann_viz(model, title="My first neural network")

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
desvioAmostralpred = np.std(inv_yhat) #desvio padrão populacional
varianciaAmostralpred = inv_yhat.var() #variancia populacional

desvioAmostralreal = np.std(inv_y) #desvio padrão populacional
varianciaAmostralreal = inv_y.var() #variancia populacional

slope, intercept, r_value, p_value, std_err = stats.linregress(inv_y, inv_yhat)

coeffs1 = np.polyfit(inv_y, inv_yhat, 5)
p1 = np.poly1d(coeffs1)
# fit values, and mean
yhat1 = p1(inv_y)                            # or [p(z) for z in x]
ybar1 = np.sum(inv_yhat)/len(inv_yhat)           # or sum(y)/len(y)
ssreg1 = np.sum((yhat1-ybar1)**2)              # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot1 = np.sum((inv_yhat - ybar1)**2)          # or sum([ (yi - ybar)**2 for yi in y])
r1 = ssreg1 / sstot1

erro = []
for x in inv_yhat:
    erro = inv_yhat - inv_y
    
print('P6')
print('Test RMSE: %.3f' % rmse)
print("R2 linear", r_value ** 2)
print("R2 Polinomial:", r1)
print("Desvio Real", desvioAmostralreal) 
print("Variancia Real", varianciaAmostralreal)
print("Desvio Predito", desvioAmostralpred) 
print("Variancia Predito", varianciaAmostralpred)
print('Real')
print(inv_y.round(1))
print('Predito')
print(inv_yhat.round(1))

file = open(cmh + "\dados.txt", "a")
file.write('\n\r' + 'P6' + '\n')
file.write('Test RMSE:' + '%.3f' % rmse + '\n')
file.write('R2 linear:' + str(r_value ** 2) + '\n')
file.write("R2 Polinomial:" + str(r1) + '\r')
file.write("Desvio Real:" + str(desvioAmostralreal) + '\n') 
file.write("Variancia Real:" + str(varianciaAmostralreal) + '\n')
file.write("Desvio Predito:" + str(desvioAmostralpred) + '\n') 
file.write("Variancia Predito:" + str(varianciaAmostralpred) + '\n')
file.write('Real' + '\n')
for a in inv_y:
	file.write(str(a) + ',' + ' ')
file.write('\n' + 'Predito' + '\n')
for b in inv_yhat:
	file.write(str(b) + ',' + ' ')
file.close()

fig1, ax1 = pyplot.subplots()
pyplot.boxplot([inv_y, inv_yhat, erro], labels=['Real', 'Predito', 'Erro'])
pyplot.title('P6')
dpi = fig1.get_dpi()
pyplot.savefig(cmh + '\P6_Boxplot_ModeloAjus.png', dpi=dpi*2)
pyplot.close()

pyplot.scatter(inv_y, inv_yhat)
range = [inv_y.min(), inv_yhat.max()]
pyplot.plot(range, range, 'red')
pyplot.title('P6 - Real x Predito')
pyplot.ylabel('Predito')
pyplot.xlabel('Real')
pyplot.savefig(cmh + '\P6_Dispersao_ModeloAjus.png', dpi=dpi*2)
pyplot.close()
########################################################################################################################
arq = read_csv(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\ModeloAjustado\lotacao_cristina_p5.csv', header=0, delimiter=';')
dados = arq.values
tam = len(arq)
data = dados[:, 0]
dia = data[(tam-12):,]

fig, ax = pyplot.subplots()
pyplot.ylabel('kg/ha/dia')
fig.autofmt_xdate()

pyplot.plot_date(dia, Cinv_y, label='Real P5', linestyle='--', marker='o')
pyplot.plot_date(dia, Cinv_yhat, label='Predito P5', linestyle='--', marker='o')
pyplot.legend()
dpi = fig.get_dpi()
pyplot.savefig(cmh + '\P5_ModeloAjus.png', dpi=dpi*2)
pyplot.close()

arq = read_csv(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\ModeloAjustado\lotacao_cristina_p6.csv', header=0, delimiter=';')
dados = arq.values
tam = len(arq)
data = dados[:, 0]
dia = data[(tam-12):,]

fig, ax = pyplot.subplots()
pyplot.ylabel('kg/ha/dia')
fig.autofmt_xdate()
pyplot.plot_date(dia, inv_y, label='Real P6', linestyle='--', marker='o')
pyplot.plot_date(dia, inv_yhat, label='Predito P6', linestyle='--', marker='o')
pyplot.legend()
dpi = fig.get_dpi()
pyplot.savefig(cmh + '\P6_ModeloAjus.png', dpi=dpi*2)
pyplot.close()

#REVERSE GRAPH
#pyplot.plot_date(C[::-1], Cinv_y, label='Real Inf', linestyle='--', marker='o')
#pyplot.plot_date(C[::-1], Cinv_yhat, label='Predito Inf', linestyle='--', marker='o')

#pyplot.plot_date(dataset.index.values[(test_X.shape[0]*-1):][::-1], inv_y[::-1], label='Real', linestyle='--', marker='o')
#pyplot.plot_date(dataset.index.values[(test_X.shape[0]*-1):][::-1], inv_yhat[::-1], label='Predito', linestyle='--', marker='o')

####################################################Potreiro 21##########################################################################################
# load dataset
dataset = read_csv('e7_cristina.csv', header=0, index_col=0, delimiter=';')
values = dataset.values
# integer encode direction
encoder = LabelEncoder()

#Drop NA Values
values=values[~np.isnan(values).any(axis=1)]

# ensure all data is float
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

scaled = DataFrame(scaled)
#print(scaled)
 
# split into train and test sets
values = scaled.values
#n_train = 24
n_train = 39
train = values[:n_train, :]
test = values[n_train:, :]

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))


#print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
#print(train_X.shape[1])

# design network
model = Sequential()
#model.add(LSTM(30, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh', return_sequences = True))#bom para 1 e 2
#model.add(LSTM(15, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh', return_sequences = True))#bom para 1 e 2
#model.add(LSTM(7, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh'))


model.add(LSTM(30, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal',  return_sequences = True))#bom para 1 e 2
model.add(LSTM(15, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal'))

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
history = model.fit(train_X, train_y, epochs=5000, batch_size=72, validation_data=(test_X, test_y), verbose=0, shuffle=False)

# plot history
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.show()
 
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((test_X, yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,-1]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_X, test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,-1]

#print(model.summary())

from keras.utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#from ann_visualizer.visualize import ann_viz

#ann_viz(model, title="My first neural network")

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
desvioAmostralpred = np.std(inv_yhat) #desvio padrão populacional
varianciaAmostralpred = inv_yhat.var() #variancia populacional

desvioAmostralreal = np.std(inv_y) #desvio padrão populacional
varianciaAmostralreal = inv_y.var() #variancia populacional

slope, intercept, r_value, p_value, std_err = stats.linregress(inv_y, inv_yhat)

coeffs1 = np.polyfit(inv_y, inv_yhat, 5)
p1 = np.poly1d(coeffs1)
# fit values, and mean
yhat1 = p1(inv_y)                            # or [p(z) for z in x]
ybar1 = np.sum(inv_yhat)/len(inv_yhat)           # or sum(y)/len(y)
ssreg1 = np.sum((yhat1-ybar1)**2)              # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot1 = np.sum((inv_yhat - ybar1)**2)          # or sum([ (yi - ybar)**2 for yi in y])
r1 = ssreg1 / sstot1

print('P7')
print('Test RMSE: %.3f' % rmse)
print("R2 linear", r_value ** 2)
print("R2 Polinomial:", r1)
print("Desvio Real", desvioAmostralreal) 
print("Variancia Real", varianciaAmostralreal)
print("Desvio Predito", desvioAmostralpred) 
print("Variancia Predito", varianciaAmostralpred)
print('Real')
print(inv_y.round(1))
print('Predito')
print(inv_yhat.round(1))

file = open(cmh + "\dados.txt", "a")
file.write('\n\r' + 'P7' + '\n')
file.write('Test RMSE:' + '%.3f' % rmse + '\n')
file.write('R2 linear:' + str(r_value ** 2) + '\n')
file.write("R2 Polinomial:" + str(r1) + '\r')
file.write("Desvio Real:" + str(desvioAmostralreal) + '\n') 
file.write("Variancia Real:" + str(varianciaAmostralreal) + '\n')
file.write("Desvio Predito:" + str(desvioAmostralpred) + '\n') 
file.write("Variancia Predito:" + str(varianciaAmostralpred) + '\n')
file.write('Real' + '\n')
for a in inv_y:
	file.write(str(a) + ',' + ' ')
file.write('\n' + 'Predito' + '\n')
for b in inv_yhat:
	file.write(str(b) + ',' + ' ')
file.close()

erro = []
for x in inv_yhat:
    erro = inv_yhat - inv_y

fig1, ax1 = pyplot.subplots()
pyplot.boxplot([inv_y, inv_yhat, erro], labels=['Real', 'Predito', 'Erro'])
pyplot.title('P7')
dpi = fig1.get_dpi()
pyplot.savefig(cmh + '\P7_Boxplot_ModeloAjus.png', dpi=dpi*2)
pyplot.close()

pyplot.scatter(inv_y, inv_yhat)
range = [inv_y.min(), inv_yhat.max()]
pyplot.plot(range, range, 'red')
pyplot.title('P7 - Real x Predito')
pyplot.ylabel('Predito')
pyplot.xlabel('Real')
pyplot.savefig(cmh + '\P7_Dispersao_ModeloAjus.png', dpi=dpi*2)
pyplot.close()
#COPY VALUES

C=dataset.index.values[(test_X.shape[0]*-1):]
Cinv_y=inv_y
Cinv_yhat=inv_yhat

########################################################################################################################
# load dataset
dataset = read_csv('e8_cristina.csv', header=0, index_col=0, delimiter=';')
values = dataset.values
# integer encode direction
encoder = LabelEncoder()

#Drop NA Values
values=values[~np.isnan(values).any(axis=1)]

# ensure all data is float
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

scaled = DataFrame(scaled)
#print(scaled)
 
# split into train and test sets
values = scaled.values
n_train = 39
#n_train = 27
train = values[:n_train, :]
test = values[n_train:, :]

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))


#print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
#print(train_X.shape[1])

# design network
model = Sequential()
#model.add(LSTM(30, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh', return_sequences = True))#bom para 1 e 2
#model.add(LSTM(15, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh', return_sequences = True))#bom para 1 e 2
#model.add(LSTM(7, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh'))


model.add(LSTM(30, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal',  return_sequences = True))#bom para 1 e 2
model.add(LSTM(15, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal'))

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
history = model.fit(train_X, train_y, epochs=5000, batch_size=72, validation_data=(test_X, test_y), verbose=0, shuffle=False)

# plot history
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.show()
 
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((test_X, yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,-1]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_X, test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,-1]

#print(model.summary())

from keras.utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#from ann_visualizer.visualize import ann_viz

#ann_viz(model, title="My first neural network")

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
desvioAmostralpred = np.std(inv_yhat) #desvio padrão populacional
varianciaAmostralpred = inv_yhat.var() #variancia populacional

desvioAmostralreal = np.std(inv_y) #desvio padrão populacional
varianciaAmostralreal = inv_y.var() #variancia populacional

slope, intercept, r_value, p_value, std_err = stats.linregress(inv_y, inv_yhat)

coeffs1 = np.polyfit(inv_y, inv_yhat, 5)
p1 = np.poly1d(coeffs1)
# fit values, and mean
yhat1 = p1(inv_y)                            # or [p(z) for z in x]
ybar1 = np.sum(inv_yhat)/len(inv_yhat)           # or sum(y)/len(y)
ssreg1 = np.sum((yhat1-ybar1)**2)              # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot1 = np.sum((inv_yhat - ybar1)**2)          # or sum([ (yi - ybar)**2 for yi in y])
r1 = ssreg1 / sstot1

print('P8')
print('Test RMSE: %.3f' % rmse)
print("R2 linear", r_value ** 2)
print("R2 Polinomial:", r1)
print("Desvio Real", desvioAmostralreal) 
print("Variancia Real", varianciaAmostralreal)
print("Desvio Predito", desvioAmostralpred) 
print("Variancia Predito", varianciaAmostralpred)
print('Real')
print(inv_y.round(1))
print('Predito')
print(inv_yhat.round(1))

file = open(cmh + "\dados.txt", "a")
file.write('\n\r' + 'P8' + '\n')
file.write('Test RMSE:' + '%.3f' % rmse + '\n')
file.write('R2 linear:' + str(r_value ** 2) + '\n')
file.write("R2 Polinomial:" + str(r1) + '\r')
file.write("Desvio Real:" + str(desvioAmostralreal) + '\n') 
file.write("Variancia Real:" + str(varianciaAmostralreal) + '\n')
file.write("Desvio Predito:" + str(desvioAmostralpred) + '\n') 
file.write("Variancia Predito:" + str(varianciaAmostralpred) + '\n')
file.write('Real' + '\n')
for a in inv_y:
	file.write(str(a) + ',' + ' ')
file.write('\n' + 'Predito' + '\n')
for b in inv_yhat:
	file.write(str(b) + ',' + ' ')
file.close()

erro = []
for x in inv_yhat:
    erro = inv_yhat - inv_y
  
fig1, ax1 = pyplot.subplots()
pyplot.boxplot([inv_y, inv_yhat, erro], labels=['Real', 'Predito', 'Erro'])
pyplot.title('P8')
dpi = fig1.get_dpi()
pyplot.savefig(cmh + '\P8_Boxplot_ModeloAjus.png', dpi=dpi*2)
pyplot.close()
  
pyplot.scatter(inv_y, inv_yhat)
range = [inv_y.min(), inv_yhat.max()]
pyplot.plot(range, range, 'red')
pyplot.title('P8 - Real x Predito')
pyplot.ylabel('Predito')
pyplot.xlabel('Real')
pyplot.savefig(cmh + '\P8_Dispersao_ModeloAjus.png', dpi=dpi*2)
pyplot.close()
########################################################################################################################
arq = read_csv(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\ModeloAjustado\lotacao_cristina_p7.csv', header=0, delimiter=';')
dados = arq.values
tam = len(arq)
data = dados[:, 0]
dia = data[(tam-12):,]

fig, ax = pyplot.subplots()
pyplot.ylabel('kg/ha/dia')
fig.autofmt_xdate()

pyplot.plot_date(dia, Cinv_y, label='Real P7', linestyle='--', marker='o')
pyplot.plot_date(dia, Cinv_yhat, label='Predito P7', linestyle='--', marker='o')
pyplot.legend()
dpi = fig.get_dpi()
pyplot.savefig(cmh + '\P7_ModeloAjus.png', dpi=dpi*2)
pyplot.close()

arq = read_csv(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\ModeloAjustado\lotacao_cristina_p8.csv', header=0, delimiter=';')
dados = arq.values
tam = len(arq)
data = dados[:, 0]
dia = data[(tam-12):,]

fig, ax = pyplot.subplots()
pyplot.ylabel('kg/ha/dia')
fig.autofmt_xdate()
pyplot.plot_date(dia, inv_y, label='Real P8', linestyle='--', marker='o')
pyplot.plot_date(dia, inv_yhat, label='Predito P8', linestyle='--', marker='o')
pyplot.legend()
dpi = fig.get_dpi()
pyplot.savefig(cmh + '\P8_ModeloAjus.png', dpi=dpi*2)
pyplot.close()

#REVERSE GRAPH
#pyplot.plot_date(C[::-1], Cinv_y, label='Real Inf', linestyle='--', marker='o')
#pyplot.plot_date(C[::-1], Cinv_yhat, label='Predito Inf', linestyle='--', marker='o')

#pyplot.plot_date(dataset.index.values[(test_X.shape[0]*-1):][::-1], inv_y[::-1], label='Real', linestyle='--', marker='o')
#pyplot.plot_date(dataset.index.values[(test_X.shape[0]*-1):][::-1], inv_yhat[::-1], label='Predito', linestyle='--', marker='o')

###################################################Potreiro 20####################################################################
# load dataset
dataset = read_csv('e9_cristina.csv', header=0, index_col=0, delimiter=';')
values = dataset.values
# integer encode direction
encoder = LabelEncoder()

#Drop NA Values
values=values[~np.isnan(values).any(axis=1)]

# ensure all data is float
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

scaled = DataFrame(scaled)
#print(scaled)
 
# split into train and test sets
values = scaled.values
#n_train = 24
n_train = 39
train = values[:n_train, :]
test = values[n_train:, :]

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))


#print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
#print(train_X.shape[1])

# design network
model = Sequential()
#model.add(LSTM(30, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh', return_sequences = True))#bom para 1 e 2
#model.add(LSTM(15, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh', return_sequences = True))#bom para 1 e 2
#model.add(LSTM(7, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal', activation='tanh'))


model.add(LSTM(30, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal',  return_sequences = True))#bom para 1 e 2
model.add(LSTM(15, input_shape=(train_X.shape[1], train_X.shape[2]),  kernel_initializer='normal'))

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
history = model.fit(train_X, train_y, epochs=5000, batch_size=72, validation_data=(test_X, test_y), verbose=0, shuffle=False)

# plot history
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.show()
 
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((test_X, yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,-1]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_X, test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,-1]

#print(model.summary())

from keras.utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#from ann_visualizer.visualize import ann_viz

#ann_viz(model, title="My first neural network")

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
desvioAmostralpred = np.std(inv_yhat) #desvio padrão populacional
varianciaAmostralpred = inv_yhat.var() #variancia populacional

desvioAmostralreal = np.std(inv_y) #desvio padrão populacional
varianciaAmostralreal = inv_y.var() #variancia populacional

slope, intercept, r_value, p_value, std_err = stats.linregress(inv_y, inv_yhat)

coeffs1 = np.polyfit(inv_y, inv_yhat, 5)
p1 = np.poly1d(coeffs1)
# fit values, and mean
yhat1 = p1(inv_y)                            # or [p(z) for z in x]
ybar1 = np.sum(inv_yhat)/len(inv_yhat)           # or sum(y)/len(y)
ssreg1 = np.sum((yhat1-ybar1)**2)              # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot1 = np.sum((inv_yhat - ybar1)**2)          # or sum([ (yi - ybar)**2 for yi in y])
r1 = ssreg1 / sstot1

erro = []
for x in inv_yhat:
    erro = inv_yhat - inv_y
    
print('P9')
print('Test RMSE: %.3f' % rmse)
print("R2 linear", r_value ** 2)
print("R2 Polinomial:", r1)
print("Desvio Real:", desvioAmostralreal) 
print("Variancia Real:", varianciaAmostralreal)
print("Desvio Predito:", desvioAmostralpred) 
print("Variancia Predito:", varianciaAmostralpred)
print('Real')
print(inv_y.round(1))
print('Predito')
print(inv_yhat.round(1))

file = open(cmh + "\dados.txt", "a")
file.write('\n\r' + 'P9' + '\n')
file.write('Test RMSE:' + '%.3f' % rmse + '\n')
file.write('R2 linear:' + str(r_value ** 2) + '\n')
file.write("R2 Polinomial:" + str(r1) + '\r')
file.write("Desvio Real:" + str(desvioAmostralreal) + '\n') 
file.write("Variancia Real:" + str(varianciaAmostralreal) + '\n')
file.write("Desvio Predito:" + str(desvioAmostralpred) + '\n') 
file.write("Variancia Predito:" + str(varianciaAmostralpred) + '\n')
file.write('Real' + '\n')
for a in inv_y:
	file.write(str(a) + ',' + ' ')
file.write('\n' + 'Predito' + '\n')
for b in inv_yhat:
	file.write(str(b) + ',' + ' ')
file.close()

fig1, ax1 = pyplot.subplots()
pyplot.boxplot([inv_y, inv_yhat, erro], labels=['Real', 'Predito', 'Erro'])
pyplot.title('P9')
dpi = fig1.get_dpi()
pyplot.savefig(cmh + '\P9_Boxplot_ModeloAjus.png', dpi=dpi*2)
pyplot.close()
pyplot.scatter(inv_y, inv_yhat)
range = [inv_y.min(), inv_yhat.max()]
pyplot.plot(range, range, 'red')
pyplot.title('P9 - Real x Predito')
pyplot.ylabel('Predito')
pyplot.xlabel('Real')
pyplot.savefig(cmh + '\P9_Dispersao_ModeloAjus.png', dpi=dpi*2)
pyplot.close()

arq = read_csv(r'C:\Users\Ânderson Fischoeder\Desktop\Tese\Modelos\ModeloAjustado\lotacao_cristina_p9.csv', header=0, delimiter=';')
dados = arq.values
tam = len(arq)
data = dados[:, 0]
dia = data[(tam-12):,]

fig, ax = pyplot.subplots()
pyplot.ylabel('kg/ha/dia')
fig.autofmt_xdate()
pyplot.plot_date(dia, inv_y, label='Real P9', linestyle='--', marker='o')
pyplot.plot_date(dia, inv_yhat, label='Predito P9', linestyle='--', marker='o')
pyplot.legend()
dpi = fig.get_dpi()
pyplot.savefig(cmh + '\P9_ModeloAjus.png', dpi=dpi*2)
pyplot.close()

now = datetime.now()

log = open('log.txt', 'a')
log.write("------Instantanea Cristina Excecução Total encerrada em: " + str(now) + "\n\n")