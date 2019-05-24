from re import search
import numpy as np
import os
import mne
from subprocess import getoutput as gop
import matplotlib.pyplot as plt
import glob
from copy import deepcopy
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adam
from keras.utils.np_utils import to_categorical

def plot_history(h):
    loss_list = [s for s in h.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in h.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in h.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in h.history.keys() if 'acc' in s and 'val' in s]
    if len(loss_list) == 0:
        print('Custo não está presente no histórico')
        return
    epochs = range(1, len(history.history[loss_list[0]]) + 1)
    # Custo
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, h.history[l], 'b',
                 label='Custo [treinamento] (' + str(str(format(
                    h.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, h.history[l], 'g',
                 label='Custo [validação] (' + str(str(format(
                    h.history[l][-1],'.5f'))+')'))
    plt.title('Custo')
    plt.xlabel('Épocas')
    plt.ylabel('Custo')
    plt.legend()
    # Acurácia
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, h.history[l], 'b',
                 label='Acurácia [treinamento] (' + str(format(
                    h.history[l][-1],'.5f'))+')')
    for l in val_acc_list:
        plt.plot(epochs, h.history[l], 'g',
                 label='Acurácia [validação] (' + str(format(
                    h.history[l][-1],'.5f'))+')')
    plt.title('Acurácia')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.show()


# identificando pastas
folders = {
    'small': 'dataset/small',
    'large_train': 'dataset/large_train',
    'large_test': 'dataset/large_test',
    'full': 'dataset/full',
}

ch_names = []
create_ch_name = False
labels_keras_train = list()

# carregando pasta "large_train"
path = gop('ls {}'.format(folders['large_train'])).split('\n')
# 1ª dimensão dos dados contendo os sujeitos. Ex.: C_1, a_m, etc
subjects = list()
for types in path:
    if("co2c" not in types):
        for i in range (0,30):
            labels_keras.append(0)
    else:
        for i in range (0,30):
            labels_keras.append(1)
    files = gop('ls {}/{}'.format(folders['large_train'], types)).split('\n')
    # 2ª dimensão dos dados contendo as sessões (trials)
    trials = list()
    for f in files:
        arquivo = open('{}/{}/{}'.format(folders['large_train'], types, f))
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
        trials.append(chs)
        arquivo.close()
    subjects.append(trials)
data = np.array(subjects)
print(data.shape)

# normalização dos dados
newData = list()
scaler = StandardScaler()
for person in data:
    newEletrodos = list()
    for eletrodos in person:
        scaler.fit(eletrodos)
        newEletrodos.append(scaler.transform(eletrodos))
    newData.append(np.array(newEletrodos))
data = np.array(newData)
print(data.shape)


# mudança no shape
newData = list()
for person in data:
    for eletrodos in person:
        newData.append(eletrodos)
data = np.array(newData)
newData = None
print(data.shape)

# ________________________________________________________________________

# definição de uma fração do regularizador
l = 0.01

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'
# urlretrieve(url, 'car.txt')
# filedata = open('car.txt')
# data = filedata.read()
# dataset = np.array([s.split(',') for s in data.split('\n')][:-1])
# print(dataset)
# print(len(dataset))
# print(dataset.shape)

# # Transformação dos valores de categórico para numérico
# le = LabelEncoder()
# features = np.array([le.fit_transform(f) for f in dataset[:, :-1].T]).T

# # obtendo a coluna com as respostas
# labels = le.fit_transform(dataset[:, -1])
# categorical_labels = to_categorical(labels, num_classes=len(set(labels)))

# # Dividindo em conjuntos de treino (80%) e teste (20%)
# X_train, X_test, y_train, y_test = train_test_split(
#     features, categorical_labels, test_size=0.3)

# # treino: 80% dos 80% de treino. teste: 20% dos 80% de treino.
# X_train, X_val, y_train, y_val = train_test_split(
#     X_train, y_train, test_size=0.3, shuffle=True)

# desenvolvimento do modelo Keras para uma MLP

labels_keras = np.array(labels_keras)

model = Sequential()
model.add(Dense(570, activation='relu', input_shape=(64,256)))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Aplicação de um modelo de descida de gradiente utilizando o Stocastic Gradient Descendent (SGD)
sgd = SGD(lr=0.05, momentum=0.0)
# Função de otimização da rede: ADAM
adam = Adam(lr=0.005, beta_1=0.9, beta_2=0.999)
# Função de custo baseada em dados originalmente categóricos
model.compile(loss='categorical_crossentropy', optimizer=adam,
              metrics=['accuracy'])


# 
# 
# PAREI AQUI!!!!
# 
# 

# remmover X, nd e Y
# diferenciar controle de não controle 4 caractere pirmeira linha
# utilizar stardtscaler para normalizar


history = model.fit(data, labels_keras , epochs=30)
plot_history(history)

# score = model.evaluate(x_test, y_test, batch_size=128)
# score = model.predict_classes(X_test)
# y_true = [np.where(x == 1)[0][0] for x in y_test]
# print('Acurácia: %0.2f%%' % (accuracy_score(y_true, score) * 100))
# print('Matriz de confusão:')
# print(confusion_matrix(y_true, score))
# print()
# print(classification_report(y_true, score, digits=5))





# tentei assim
# 
# 
# for k in range(0,len(data)):
#     for i in range(0,len(data[k])):
#         for j in range(0,len(data[k][i])):
#             print("data", data[k][i][j])
#             aux = np.mean(data[k][i][j])
#             data[k][i] = aux
#             print("data", data[k][i])
#             break
#         break
#     break


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


# Prepara info para os eletrodos selecionados
# primeira sessão, primeiro trial
# Média de cada eletrodo (256 -> 1)
# newData = list()
# for person in data:
#     for eletrodos in person:
#         newEletrodos = list()
#         for eletrodo in eletrodos:
#             newEletrodos.append(np.mean(eletrodo))
#         newData.append(newEletrodos)

# print(np.array(newData).shape)

# TESTE
# norm1 = normalize(data, axis=1)
# print(norm1.shape)
# print(norm1)


