import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

# para fazer avaliacao utilizando o wrapper (mais rapido e facil)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

# passando parametros ao invés de inicializar de forma default para criarmos
# a rede neural a forma que quisermos
def criarRede(optimizer, loos, kernel_initializer, activation, neurons):
    
    # criação da rede neural
    classificador = Sequential()

    # adicionando neurônios fullyconnected (todos conectados com o próximo)
    # será feito para a primeira camada oculta, pois o input_din irá adicionar a quantidade de elemnentos na camada de entrada
    # (unidades/quantidade de neuronios, funcao de ativacao, inicializador dos pesos, quantidade de elementos na camada de entrada)
    classificador.add(Dense(units= neurons, activation= activation, 
                            kernel_initializer= kernel_initializer,
                            input_dim= 30))
    # dropout 20%
    classificador.add(Dropout(0.2))
    
    # adicionando uma camada oculta sem o dim pq o dim é para a camada de entrada
    classificador.add(Dense(units= neurons, activation= activation, 
                            kernel_initializer= kernel_initializer))

    # dropout 20%
    classificador.add(Dropout(0.2))
    
    # adicionando a camada de saida
    classificador.add(Dense(units= 1, activation= 'sigmoid'))

    # implementacao da funcao otimizadora
    classificador.compile(optimizer= optimizer, loss= loos,
                          metrics= ['binary_accuracy'])
    return classificador

# executa a rede
classificador = KerasClassifier(build_fn= criarRede)

# basicamente todos os valores que podemos mexer na criacao da rede neural
# feito assim podemos utilizar o grid_search para executar cada um dos
# parametros para podermos ver quais foram os melhores resultados
parametros = {'batch_size': [10, 30],
              'epochs': [50, 100],
              'optimizer': ['adams', 'sgd'],
              'loos': ['binary_crossentropy', 'hinge'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu', 'tanh'],
              'neurons': [16, 8]}

grid_search = GridSearchCV(estimator= classificador,
                           param_grid= parametros,
                           scoring= 'accuracy',
                           cv= 5)

# valores esperados da camada de entrada e saida
grid_search = grid_search.fit(previsores, classe)

# vai indicar 1 dos 2 parametros passados na variavel parametro
melhores_parametros = grid_search.best_params_

# melhor valor atingido pela rede neural utilizando os melhores parametros
melhor_precisao = grid_search.best_score_

# com esses dados farei a configuracao final da rede neural
