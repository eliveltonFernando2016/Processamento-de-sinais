from re import search
import numpy as np
import os
import mne
from subprocess import getoutput as gop
import matplotlib.pyplot as plt
import glob
from copy import deepcopy

# identificando pastas
folders = {
    'small': 'dataset/small',
    'large_train': 'dataset/large_train',
    'large_test': 'dataset/large_test',
    'full': 'dataset/full',
}

ch_names = []
create_ch_name = False

# carregando pasta "small"
small_dir = gop('ls {}'.format(folders['large_train'])).split('\n')
# 1ª dimensão dos dados contendo os sujeitos. Ex.: C_1, a_m, etc
subjects = list()
for types in small_dir:
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
newData = list()
for person in data:
    for eletrodos in person:
        newEletrodos = list()
        for eletrodo in eletrodos:
            newEletrodos.append(np.mean(eletrodo))
        newData.append(newEletrodos)

data = newData
newData = None

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



# pessoa1 = data[0][0]

# ch_names = ch_names
# ch_types = ['eeg'] * 64

# info = mne.create_info(
#     ch_names=ch_names, 
#     sfreq=256, 
#     ch_types=ch_types)

# raw = mne.io.RawArray(pessoa1, info)

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

