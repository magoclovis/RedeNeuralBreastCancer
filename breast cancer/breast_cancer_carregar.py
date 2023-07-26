import numpy as np
import pandas as pd

# importar arquivo json
from keras.models import model_from_json

arquivo = open('classificador_breast.json', 'r')

# faz a leitura do arquivo e salva os dados na variavel
estrutura_rede = arquivo.read()

# fecha o arquivo para poupar memória
arquivo.close()

# carrega os dados que foram exportados 
classificador = model_from_json(estrutura_rede)
# carrega os pesos treinados
classificador.load_weights('classificador_breast.h5')

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

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

classificador.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=['binary_accuracy'])

# fazendo a avaliação utilizando a base de dados carregada acima
# 1ª linha - valor da loss function
# 2ª linha - valor da precisao
resultado = classificador.evaluate(previsores, classe)
