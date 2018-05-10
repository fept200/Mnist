# -*- coding: utf-8 -*-
""""
DNN AutoEncoder
時系列(OneHot)のデータの特徴値を次元削減で教師なし学習する
"""
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
import chainer.functions as F
import chainer.links as L
import copy, sys, glob, os, re, csv, random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import pdb


class DnnAutoEncoder(chainer.Chain):
    def __init__(self):

        super(DnnAutoEncoder, self).__init__()
        with self.init_scope():
            self.e1 = L.Linear(784, 900) 
            self.e2 = L.Linear(None, 600)
            self.e3 = L.Linear(None, 300)

            self.ed = L.Linear(None, 10)

            self.d3 = L.Linear(None, 300)
            self.d2 = L.Linear(None, 600)
            self.d1 = L.Linear(None, 900)
            self.d0 = L.Linear(None, 784)

            self.l1 = L.Linear(None, 10)
            self.l2 = L.Linear(None, 2)

    def __call__(self, x):
        h = F.relu(self.e1(x))
        h = F.relu(self.e2(h))
        h = F.relu(self.e3(h))

        hed = self.ed(h)

        h = F.relu(self.d3(hed))
        h = F.relu(self.d2(h))
        h = F.relu(self.d1(h))
        hd = self.d0(h)

        h = F.relu(self.l1(hed))
        hl = self.l2(h)

        return hed, hd, hl

    def csvsaves(self, file, data):
        with open('./' + file, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(data)

    def csvload(self, filename):
        data = []
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                data.append(row)
        return data

    def load_csv_file(self, filename):
        data = np.array(self.csvload(filename))
        return data

    def loaddir(self, dir):
        dirs = sorted(glob.glob(dir + '/*'))
        ls = []
        for filename in dirs:
            pattern = dir +'/(.*)'
            matchs = re.finditer(pattern, filename)
            for m in matchs:
                ls.append(m.groups()[0])
        return ls

    def loss_mean_squared(self, y, t):
        self.loss = None
        #self.lossfun = F.mean_absolute_error
        self.lossfun = F.mean_squared_error
        ## 各BatchのDecorder出力の誤差を取る
        ## Batch毎に処理するため遅い
        for yi, ti in zip(y, t):
            #print('yi: ', yi.data)
            #print('ti: ', ti.data)
            loss = self.lossfun(yi, ti)
            self.loss = loss if self.loss is None else self.loss + loss
            #print(self.xp.argmax(yi.data, axis=1).astype('i'))

        ## 累計loss
        return self.loss / 784

""" 
__main__
"""
if __name__ == '__main__':
    """
    初期値
    """
    gpu = 0
    predict_datadir = './data/csv.28x28/predict/09'

    """
    モデル作成
    """
    model = DnnAutoEncoder()
    optimizer = optimizers.Adam(); OPTIMIZERNAME = 'Adam'
    optimizer.setup(model)

    """
    Load/Save
    """
    import copy
    ## Save
    #serializers.save_npz('init_model.npz', copy.deepcopy(model).to_cpu())
    #serializers.save_npz('init_optimizer.npz', optimizer) ## 再学習時に必要
    ## Load
    serializers.load_npz('model_unsupervised.npz', model)

    """
    データ作成
    """
    ## GPU
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()  # Make the GPU current
        model.to_gpu()
        xp = cuda.cupy
    else:
        xp = np
        
    predict_datafiles = model.loaddir(predict_datadir)

    """
    predict
    """
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        result = []
        accum_hed = np.empty((0, 10), int)
        for f in predict_datafiles:
            re = []
            if f[0:1] == '0':
                label = 0
            else:
                label = 1
            data = model.load_csv_file(predict_datadir + '/' + f)
            xs = xp.asarray(data, dtype=np.float32) ## Linear入力値はfloat32
            xs = xs.reshape(1,784)
            hed, hd, hl = model(xs)
            re.append(f)
            re.append(float(model.loss_mean_squared(hd, xs).data))
            re.append(float(F.softmax(hl)[0][0].data))
            re.append(label)
            result.append(re)
            hed_data = chainer.cuda.to_cpu(hed.data)
            accum_hed = np.append(accum_hed, hed_data, axis=0)
        model.csvsaves("result_unsupervised" , result)
        tsne_data = TSNE(n_components=2, random_state=0).fit_transform(accum_hed)

        plt.figure(1)
        for i in range(tsne_data.shape[0]):
            if i < 400:
                plt.scatter(tsne_data[i,0], tsne_data[i,1], alpha=0.7, color='blue' ,s=2)
            else:
                plt.scatter(tsne_data[i,0], tsne_data[i,1], alpha=0.7, color='orange', s=2)
        plt.show()
        sys.exit()
