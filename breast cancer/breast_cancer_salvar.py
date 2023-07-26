import pandas as pd
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

# salvar a rede no formato .json (os dados que foram passados como parametro
# no caso)
classificador_json = classificador.to_json()

# salvar em disco (no pc)
# w = write (gravar em disco) , r = read (ler em disco)
with open('classificador_breast.json', 'w') as json_file:
    json_file.write(classificador_json)

# salvar os pesos
classificador.save_weights('classificador_breast.h5')
