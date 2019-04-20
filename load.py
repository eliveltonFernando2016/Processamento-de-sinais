from urllib.request import urlopen, urlretrieve
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import mne
from mne.io import RawArray as ra
from mne.time_frequency import psd_multitaper as pm

urls = {
    'small': 'https://archive.ics.uci.edu/ml/machine-learning-databases/eeg-mld/smni_eeg_data.tar.gz',
    'large_train': 'https://archive.ics.uci.edu/ml/machine-learning-databases/eeg-mld/SMNI_CMI_TRAIN.tar.gz',
    'large_test': 'https://archive.ics.uci.edu/ml/machine-learning-databases/eeg-mld/SMNI_CMI_TEST.tar.gz',
    'full': 'https://archive.ics.uci.edu/ml/machine-learning-databases/eeg-mld/eeg_full.tar'
}

# verifica se o diretório dos datasets existe
if not os.path.exists('dataset/'):
    os.mkdir('dataset/')
    for k, v in urls.items():
        fn = v.split('/')[-1]
        print('Baixando:', fn, '...')
        urlretrieve(v, './dataset/{}'.format(fn))
    print('Downlod dos datasets concluído!')
else:
    print('Dataset já baixado!')

from subprocess import getoutput as gop
import glob

# único arquivo somente empacotado (tar)
#os.mkdir('dataset/eeg_full/')
#gop('tar -xvf dataset/eeg_full.tar -C dataset/eeg_full')
#os.remove('dataset/eeg_full.tar')

while glob.glob('dataset/**/*.gz', recursive=True):
    # quando o arquivo está empacotado (tar) e compactado (gz)
    for f in glob.iglob('dataset/**/*.tar.gz', recursive=True):
        gop('tar -zxvf {} -C {}'.format(f, f[:f.rindex('/')]))
        os.remove(f)
    # quando o arquivo está somente compactado (gz)
    for f in glob.iglob('dataset/**/*.gz', recursive=True):
        gop('gzip -d {}'.format(f))
print('Descompactações finalizadas!')

# organizando melhor as pastas
#os.rename('dataset/smni_eeg_data', 'dataset/small')
#os.rename('dataset/eeg_full', 'dataset/full')
#os.rename('dataset/SMNI_CMI_TRAIN/', 'dataset/large_train/')
#os.rename('dataset/SMNI_CMI_TEST/', 'dataset/large_test/')
print(gop('ls -l dataset/'))

from re import search
import numpy as np

# identificando pastas
folders = {
    'small': 'dataset/small',
    'large_train': 'dataset/large_train',
    'large_test': 'dataset/large_test',
    'full': 'dataset/full',
}
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
        trials.append(chs)
        arquivo.close()
    subjects.append(trials)
data = np.array(subjects)
print(data.shape)

d1 = list()
d2 = list()

for e in range(64):
    for i, t in enumerate(np.linspace(0, 1, 256)):
        d1.append([e, t, data[0][0][e][i]])
        d2.append([e, t, data[1][0][e][i]])
d1 = np.array(d1)
d2 = np.array(d2)
x1, y1, z1 = d1[:,0], d1[:,1], d1[:,2]
x2, y2, z2 = d2[:,0], d2[:,1], d2[:,2]

# fig = plt.figure()

# ax = fig.add_subplot(1, 2, 1, projection='3d')
# surf = ax.plot_trisurf(x1, y1, z1, cmap=cm.inferno, linewidth=1)
# ax.set_xlabel('Canais')
# ax.set_ylabel('Tempo (seg.)')
# ax.set_zlabel('Milivolts')

# ax = fig.add_subplot(1, 2, 2, projection='3d')
# surf = ax.plot_trisurf(x2, y2, z2, cmap=cm.inferno, linewidth=1)
# ax.set_xlabel('Canais')
# ax.set_ylabel('Tempo (seg.)')
# ax.set_zlabel('Milivolts')

# fig.colorbar(surf)
# fig.tight_layout()
# plt.show()

info = mne.create_info(
    ch_names=['O2'],
    ch_types=['eeg'],
    sfreq = 256
)
# print(len(data[1][1][1]))
# print(len(data[1][1]))
# print(len(data[1]))
# print(len(data))

# raw = ra(data, info)


# psd, freq = pm(raw, 5, 240)

for k in range(0,len(data)):
    for i in range(0,len(data[k])):
        if(len(data[k][i]) < 64):
            print("arquivo: ",k)
            print("eletrodo: ",i)
            print("len arquivo: ",len(data[k][i]))