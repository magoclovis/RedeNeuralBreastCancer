import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

# scikeras é uma versao mais atualizada aparentemente
# faz a mesma coisa que o import abaixo, porém está atualizado
#from scikeras.wrappers import KerasClassifier

# para fazer avaliacao utilizando o wrapper (mais rapido e facil)
from keras.wrappers.scikit_learn import KerasClassifier

# funcao que vai fazer a divisao da base de dados em treinamento e teste
from sklearn.model_selection import cross_val_score

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

# CTRL+C - CTRL+V Do programa anterior mas dentro de uma funcao
# mais oitmizado e organizado

def criarRede():
    classificador = Sequential()

    # adicionando neurônios fullyconnected (todos conectados com o próximo)
    # será feito para a primeira camada oculta, pois o input_din irá adicionar a quantidade de elemnentos na camada de entrada
    # (unidades/quantidade de neuronios, funcao de ativacao, inicializador dos pesos, quantidade de elementos na camada de entrada)
    classificador.add(Dense(units= 16, activation= 'relu', 
                            kernel_initializer= 'random_uniform', input_dim= 30))
    # dropout 20%
    classificador.add(Dropout(0.2))
    
    # adicionando uma camada oculta sem o dim pq o dim é para a camada de entrada
    classificador.add(Dense(units= 16, activation= 'relu', 
                            kernel_initializer= 'random_uniform'))

    # dropout 20%
    classificador.add(Dropout(0.2))
    
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
    return classificador

# batch size = calcula o erro para X registros e aí atualiza os pesos
# epochs = épocas
# versao desatualizada (sem o scikeras)
classificador = KerasClassifier(build_fn= criarRede, 
                                epochs= 100,
                                batch_size = 10)

# fazer os testes
# resultados vai mostrar o resultado de cada teste
# cv = quantidade de testes (no caso 100 epocas 10 vezes)
# socring = como eu quero retornar os resultados
resultados = cross_val_score(estimator= classificador,
                             X= previsores,
                             y= classe,
                             cv= 10,
                             scoring= 'accuracy')

# media dos resultados da funcao acima
media = resultados.mean()

# calculo do desvio padrao
# quanto maior o valor mais a tendencia que tenha overfitting na base de dados
# overfitting = se adapta demais na base de dados e quando se passa
# dadods totalmente diferentes, ela tem dificuldade em manter boas classificacoes
desvio = resultados.std()
