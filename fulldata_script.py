import os
import mne
import glob

import numpy as np
import matplotlib.pyplot as plt

from re import search
from subprocess import getoutput as gop
from copy import deepcopy

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split

from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adam
from keras.utils.np_utils import to_categorical

# pastas de dados
folders = {
    'small': 'dataset/small',
    'large_train': 'dataset/large_train',
    'large_test': 'dataset/large_test',
    'full': 'dataset/full',
}

# Testes de cada pessoa da base
tests =  ["S1 obj", "S2 nomatch", "S2 match"]

#filtra a data e remove os eletrodos desnecessários (x, y, z)
def filterData(data):
    # normalização dos dados e remoção x, y e nd
    newData = list()
    scaler = StandardScaler()
    for person in data:
        newEletrodos = list()
        for eletrodos in person:
            eletrodos = np.delete(eletrodos, [31,62,63], axis=0)
            scaler.fit(eletrodos)
            newEletrodos.append(scaler.transform(eletrodos))
        newData.append(np.array(newEletrodos))

    return np.array(newData)

# Aletera o shape da data de (x,y,z,k) para (xy,z,k) = (2,2,3,4) para (4,3,4)
def alterShape(data, tipo):
    # mudança no shape
    newData = list()
    for person in data[0]:
        for eletrodos in person:
            newData.append(eletrodos)
    data[0] = np.array(newData)
    newData = None
    print("data "+tipo+" shape: ",data[0].shape)

    return data

#Função principal que importa os dados
def importNomalized(pathName, exp):
    labelsList = list()
    ch_names = []
    create_ch_name = False
    # carregando pasta "large_train"
    path = gop('ls {}'.format(pathName)).split('\n')
    # 1ª dimensão dos dados contendo os sujeitos. Ex.: C_1, a_m, etc
    subjects = list()
    for types in path:
        if("co2c" in types):
            for i in range (0,10):
                labelsList.append([[0]]*61)
        else:
            for i in range (0,10):
                labelsList.append([[1]]*61)

        files = gop('ls {}/{}'.format(pathName, types)).split('\n')
        # 2ª dimensão dos dados contendo as sessões (trials)
        
        trials = list()
        for f in files:
            arquivo = open('{}/{}/{}'.format(pathName, types, f))
            text = arquivo.readlines()
            # 3ª dimensão dos dados contendo os canais (eletrodos)
            chs = list()
            # 4ª dimensão dos dados contendo os valores em milivolts
            values = list()
            for line in text:
                # ex: "# FP1 chan 0"
                t = search('(?P<ch_name>\w{1,3}) chan \d{1,2}', line)
                # ex: "0 FP1 0 -8.921"
                p = search('^\d{1,2}\ \w{1,3}\ \d{1,3}\ (?P<value>.+$)', line)
                if p:
                    values.append(float(p.group('value')))
                # mudou para outro eletrodo
                elif t:
                    if values:
                        chs.append(values)
                        values = list()
                    if not create_ch_name:
                        ch_names.append(t.group('ch_name').lower())
            create_ch_name = True
            chs.append(values)

            arquivo.seek(32*3)
            line =  arquivo.readline()
            arquivo.close()

            if exp in line:
                trials.append(chs)

        subjects.append(trials)
    data = np.array(subjects)
    
    return [filterData(data),np.asarray(labelsList)]

# cria o modelo
def createModel(dataTrain):
#   Cria um modelo sequencial com 3 camadas sendo a ultíma com saída binária
    model = Sequential()
    model.add(Dense(units = 100, activation='relu', input_shape = (61,256)))
    model.add(Dense(units = 50, activation='relu'))
    model.add(Dense(units = 1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#   Printa o modelo (camadas)
    model.summary()

#   74%
    history = model.fit(dataTrain[0], dataTrain[1], nb_epoch=30)
    
    return model

# Testa os modelos construídos
def testModel(model, dataTest):
    return [(model.predict(dataTest[0]) > 0.5), (dataTest[1] == 1)]
    

# Função que testa a precisão
def accuracy(prev, labels, test):
    acerto = 0
    erro = 0
    eletrodo = 0

    # labels = (labels == 1)

    for pre, lab in zip(prev, labels):
        for x1, x2 in zip(pre, lab):
            if(x1[0] == x2[0]):
                eletrodo+=1
        if(eletrodo > 30):
            acerto+=1
        else:
            erro+=1
        eletrodo =0

    print("Accuracy "+test+": "+str((100*(acerto))/(acerto+erro)))

# ________________________________________________________________________


def main():
    data = list()
    dataTst = list()
    models = list()
    results = list()

    for tst in tests:
        print("EXP: "+tst)
        data.append(alterShape(importNomalized(folders['large_train'], tst), "treino"))
        dataTst.append(alterShape(importNomalized(folders['large_test'], tst), "teste"))

    for i in range(0, 3):
        models.append(createModel(data.pop(0)))
    
    for model in models:
        results.append(testModel(model, dataTst.pop(0)))
    
    for acc, test in zip(results,tests):
        accuracy(acc[0],acc[1], test)


if __name__ == '__main__':
    main()





# ________________________________________________________________ TESTES E AFINS:

# Prepara info para os eletrodos selecionados
# primeira sessão, primeiro trial
# Média de cada eletrodo (256 -> 1) ________________________:
# 
# newData = list()
# for person in data:
#     for eletrodos in person:
#         newEletrodos = list()
#         for eletrodo in eletrodos:
#             newEletrodos.append(np.mean(eletrodo))
#         newData.append(newEletrodos)
# 
# print(np.array(newData).shape)

# TESTE ____________________________:
# norm1 = normalize(data, axis=1)
# print(norm1.shape)
# print(norm1)

# PLOT _____________________________:
# 
# 
# pessoa1Trial1 = data[0][0]

# ch_names = ch_names
# ch_types = ['eeg'] * 64

# info = mne.create_info(
#     ch_names=ch_names, 
#     sfreq=256, 
#     ch_types=ch_types)

# raw = mne.io.RawArray(pessoa1Trial1, info)

# raw.drop_channels(['x','nd','y'])
# montage = mne.channels.read_montage('standard_1020')
# raw.set_montage(montage)
# raw.plot_psd()
# print()

# # Grafico no domínio da frequencia
# # plt.plot(np.linspace(0,1,256), raw.get_data()[0])
# # plt.xlabel('tempo (s)')
# # plt.ylabel('Dados EEG (mV/cm²)')

# raw2 = deepcopy(raw)
# raw2.notch_filter(np.arange(60, 121, 60), fir_design='firwin')
# raw2.filter(5., 50., fir_design='firwin')
# raw2.plot_psd(area_mode='range')
# print()

# print("Finalizei com Sucesso!!!")

# Função auxilar que achou a pasta e arquivo com erro na base large (remoção de arquivo necessária)
# for k in range(0,len(data)):
#     for i in range(0,len(data[k])):
#         if(len(data[k][i]) < 64):
#             print("pasta: ",k)
#             print("arquivo: ",i)
#             print("len arquivo: ",len(data[k][i]))
#             print("data", data[k][i])


# if("full" in pathName):
#     for k in range(0,len(data)):
#         for i in range(0,len(data[k])):
#             if(len(data[k][i]) < 64):
#                 print("pasta: ",k)
#                 # print("arquivo: ",i)
#                 # print("len arquivo: ",len(data[k][i]))
#                 # print("data", data[k][i])

#     exit(1)
