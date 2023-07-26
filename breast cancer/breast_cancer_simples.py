import pandas as pd
# import do pandas para a leitura dos arquivos em csv
previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

# faz a divisão da base de dados entre treinamento e teste
from sklearn.model_selection import train_test_split

# executa essa divisao de fato
previroes_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)

import keras

# classe para criação de rede neural
from keras.models import Sequential

# classe para criação das camadas ocultas
from keras.layers import Dense

# criação da rede neural
classificador = Sequential()

# adicionando neurônios fullyconnected (todos conectados com o próximo)
# será feito para a primeira camada oculta, pois o input_din irá adicionar a quantidade de elemnentos na camada de entrada
# (unidades/quantidade de neuronios, funcao de ativacao, inicializador dos pesos, quantidade de elementos na camada de entrada)
classificador.add(Dense(units= 16, activation= 'relu', 
                        kernel_initializer= 'random_uniform', input_dim= 30))

# adicionando uma camada oculta sem o dim pq o dim é para a camada de entrada
classificador.add(Dense(units= 16, activation= 'relu', 
                        kernel_initializer= 'random_uniform'))

# adicionando a camada de saida
classificador.add(Dense(units= 1, activation= 'sigmoid'))

# criado uma função otimizadora para o exercicio
# lr = learn rate
# decay = redução na velocidade de aprendizado
# clipvalue = deixar um valor clipado como referência para não buscar minimos locais e sim minimos globais
otimizador = keras.optimizers.Adam(lr= 0.001, decay= 0.0001, clipvalue= 0.5)

# implementacao da funcao otimizadora
classificador.compile(optimizer= otimizador, loss= 'binary_crossentropy',
                      metrics= ['binary_accuracy'])

# adam = descida do gradiente aristocrático
#classificador.compile(optimizer= 'adam', loss= 'binary_crossentropy',
#                      metrics= ['binary_accuracy'])

# treinamento
# batch size = calcula o erro para X registros e aí atualiza os pesos
# epochs = épocas
classificador.fit(previroes_treinamento, classe_treinamento,
                  batch_size= 10, epochs= 100)

# criação do visualizador dos pesos

# pesos len terá o tamanho de 2 pois há 2 pesos nesse exercicio
# sendo um da camada de entrada para a camada oculta
# e o outro é da unidade de bias criado junto da rede neural (automático)
# mas é possivel fazendo use bias=False no classificador.add(Dense
pesos0 = classificador.layers[0].get_weights()

#print(len(pesos0))

# pesos da primeira camada oculta para a segunda camada oculta
pesos1 = classificador.layers[1].get_weights()

# pesos da segunda camada oculta para a camada de saída
pesos2 = classificador.layers[2].get_weights()

# saida do treinamento (em probabilidade), comparar com a classe_teste para ver se os valores batem
previsoes = classificador.predict(previsores_teste)

# transformar os valores em false ou true para comparar com mais facilidade
previsoes = (previsoes > 0.5)

# 2 formas de fazer a avaliacao da rede neural:
    
# 1º sklearn:
from sklearn.metrics import confusion_matrix, accuracy_score

# precisao de acerto da rede neural em relação a base de teste
precisao = accuracy_score(classe_teste, previsoes)

# avaliacao da rede neural - quantidade de acerto/erro das saidas no seguinte formato:
# (qntd acerto | qntd erro)
# (qntd erro   | qntd acerto)
matriz = confusion_matrix(classe_teste, previsoes)

# 2º Keras:
# valor de erro
# valor de precisao
resultado = classificador.evaluate(previsores_teste, classe_teste)
