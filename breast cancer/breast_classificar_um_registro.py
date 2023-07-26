import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

classificador = Sequential()

 # adicionando neurônios fullyconnected (todos conectados com o próximo)
 # será feito para a primeira camada oculta, pois o input_din irá adicionar a quantidade de elemnentos na camada de entrada
 # (unidades/quantidade de neuronios, funcao de ativacao, inicializador dos pesos, quantidade de elementos na camada de entrada)
classificador.add(Dense(units= 8, activation= 'relu', 
                         kernel_initializer= 'normal',
                         input_dim= 30))
 # dropout 20%
classificador.add(Dropout(0.2))
 
 # adicionando uma camada oculta sem o dim pq o dim é para a camada de entrada
classificador.add(Dense(units= 8, activation= 'relu', 
                         kernel_initializer= 'normal'))

 # dropout 20%
classificador.add(Dropout(0.2))
 
 # adicionando a camada de saida
classificador.add(Dense(units= 1, activation= 'sigmoid'))

 # implementacao da funcao otimizadora
classificador.compile(optimizer= 'adam', loss= 'binary_crossentropy',
                      metrics= ['binary_accuracy'])

classificador.fit(previsores, classe, batch_size= 10, epochs= 100)

# valores para testar a rede neural inseridos manualmente
novo = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134,
                  0.178, 0.20, 0.05, 1098, 0.07, 4500, 145.2, 0.005,
                  0.04, 0.05, 0.015, 0.03, 0.007, 23.15, 16.64, 178.5,
                  2018, 0.14, 0.105, 0.84, 158, 0.364]])

# 1 = tumor maligno
# 0 = tumor beligno
previsao = classificador.predict(novo)

# para mudar a saída de 0 e 1 para true or false
# if > 0.5 = true
previsao = (previsao > 0.5)