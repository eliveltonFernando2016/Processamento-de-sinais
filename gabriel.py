import random

from subprocess import getoutput as gop
import glob

from urllib.request import urlopen, urlretrieve
import os

from re import search
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense

from sklearn.metrics import confusion_matrix, accuracy_score

from mne import set_eeg_reference as car
import mne


# nome de todos os 64 eletrodos presentes na base fornecida
ch_names =  [
            'FP1','FP2','F7','F8','AF1','AF2','FZ','F4','F3','FC6','FC5','FC2','FC1',
            'T8','T7','CZ','C3','C4','CP5','CP6','CP1','CP2','P3','P4','PZ','P8','P7','PO2','PO1',
            'O2','O1','X','AF7','AF8','F5','F6','FT7','FT8','FPZ','FC4','FC3','C6','C5','F2','F1',
            'TP8','TP7','AFZ','CP3','CP4','P5','P6','C1','C2','PO7','PO8','FCZ','POZ','OZ','P2','P1','CPZ','nd','Y'
            ]

# identificando pastas
folders = {
    'small': 'dataset/small',
    'large_train': 'dataset/large_train',
    'large_test': 'dataset/large_test',
    'full': 'dataset/full'
}


# modificação da função fornecida pelo professor afim de recuperar e separar a base pelos testes S1_obj, S2_nomatch, S2_match. 
# cada teste tem 64 eletrodos com 256 leituras
def get_all_datas(files, tests, path_files, types):
    
    trials = list()
    for f in files:
        arquivo = open('{}/{}/{}'.format(folders[path_files], types, f))
        text = arquivo.readlines()
        # 3ª dimensão dos dados contendo os canais (eletrodos)
        chs = list()

        # 4ª dimensão dos dados contendo os valores em milivolts
        values = list()
        for line in text:
            # ex: "# FP1 chan 0"
            t = search('\w{1,3} chan \d{1,2}', line)

            # ex: "0 FP1 0 -8.921"
            p = search('^\d{1,2}\ \w{1,3}\ \d{1,3}\ (?P<value>.+$)', line)
            if p:
                values.append(float(p.group('value')))
            # mudou para outro eletrodo
            elif t and values:
                chs.append(values)
                values = list()
        chs.append(values)
        arquivo.seek(32*3)
        line =  arquivo.readline()
        
        if "S1 obj" in line:
            if len(chs) != 1:   
                tests["S1_obj"].append(chs)

        elif "S2 nomatch" in line:
            if len(chs) != 1:
                tests["S2_nomatch"].append(chs)

        elif "S2 match" in line:
            if len(chs) != 1:
                tests["S2_match"].append(chs)
        
        arquivo.close()

# nesta função são identificados e separados os casos de alcoolicos e controle
def load_bases2(path_files):
    diretory = gop('ls {}'.format(folders[path_files])).split('\n')

    subA = {"S1_obj":[], "S2_nomatch":[], "S2_match":[]}
    subC = {"S1_obj":[], "S2_nomatch":[], "S2_match":[]}

    for types in diretory:
        files = gop('ls {}/{}'.format(folders[path_files], types)).split('\n')
        if 'Co2A' in types.title():
            get_all_datas(files, subA, path_files, types)
        else:
            get_all_datas(files, subC, path_files, types)

    return [subA, subC]


# função auxiliar responsável por realizar a média e entre os 64 eletrodos
def aux_pre_proc(data):
    new_raw = []

    ch_types = ['eeg'] * 64

    info = mne.create_info(ch_names=ch_names, sfreq=256, ch_types=ch_types)

    for i in data:
        raw = mne.io.RawArray(i, info, verbose= False)
        #raw.drop_channels(['X', 'nd', 'Y'])
        inst, data = car(raw, ref_channels='average', verbose= False)
        new_raw.append(data)

    return new_raw

# função resposável por realizar as médias entre os eletrodos, 
# bem como designar as respectivas classes juntamente com o embaralhamento das entradas 
def pre_pros(data):
    # recebe os valores referentes aos alcoolatras
    alco = data[0]

    # valores referentes ao controle
    contro = data[1]

    alco = aux_pre_proc(alco)
    contro = aux_pre_proc(contro)



    # identifica a quantidade de classes (alcoolatras e controles) e mistura todas elas para que seja possível treinar a rede neural
    classes = list()
    for i in range(0, len(alco)):
        classes.append(1)
    for i in range(0, len(contro)):
        classes.append(0)

    total = alco + contro
    
    # combina as classes com as suas respectivas entradas para não perder a posição respectivas de ambos
    combined = list(zip(total, classes))
    random.shuffle(combined)

    total[:], classes[:] = zip(*combined)

    return [np.asarray(total), np.asarray(classes)]


# Função responsável pelo treino da rede neural
# a fim de comparação a mesma configuração foi utilizada com a base "full"
# os ultimos resultado são:
# S1 obj: 88%
# S2 nomatch: 91%
# S2 match: 87%
# Quando o treinamento foi realizado com a base "large_train"
# os resultados cairam drasticamente:
# os ultimos resultados:
# S1 obj: 55%
# S2 nomatch: 62%
# S2 match: 60%
# demonstrando que quando há poucos dados para treino, a rede neural acaba sendo prejudicada.
def training(data, name):
    print(data[0].shape)
    print(data[1].shape)
    print(data[1])
    print(data[1][0])
    exit(0)

    classifier = Sequential()

    classifier.add(Dense(units = 50, activation = 'relu', input_dim = 256))
    classifier.add(Dense(units = 30, activation = 'relu'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    classifier.summary()

    classifier.fit(data[0], data[1], nb_epoch = 15)
    classifier.save(name)


# função responsãvel por realizar os testes nos respectivos modelos
def test_model(test, name_model):

    model = load_model(name_model)
    prev = model.predict(test[0])

    prev = (prev > 0.50)
    print("RESULTADO PARA O TESTE "+name_model+": "+str(accuracy_score(prev, test[1])))
    matrix = confusion_matrix(prev, test[1])



# função que envia os respectivos dados á função de treino
def train(train, name_test):
    name = name_test+".h5"

    train_processed = pre_pros([train[0][name_test], train[1][name_test]])
    training(train_processed, name)

    
# função que envia os respectivos dados á função de teste
def test(test, name_test):
    name = name_test+".h5"

    test_processed = pre_pros([test[0][name_test], test[1][name_test]])
    test_model(test_processed, name)



if __name__ == '__main__':
    tests =  ["S1_obj", "S2_nomatch", "S2_match"]

    data_train = load_bases2('large_train')
    data_test = load_bases2('large_test')


    for i in tests:
        train(data_train, i)
    
    for i in tests:
        test(data_test, i)