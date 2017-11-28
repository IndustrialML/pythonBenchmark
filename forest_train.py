import numpy as np
import os
import struct
import array
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from time import time

#picture = (784,) with int(0,255) where 0=white
def load_mnist(dataset="training", digits=np.arange(10), path="data", size = 60000):
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = array.array("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = array.array("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = size #int(len(ind) * size/100.)
    images = np.zeros((N, rows, cols), dtype=np.uint8)
    labels = np.zeros((N, 1), dtype=np.int8)
    for i in range(N): #int(len(ind) * size/100.)):
        images[i] = np.array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ])\
            .reshape((rows, cols))
        labels[i] = lbl[ind[i]]
    labels = [label[0] for label in labels]
    return np.array(images).reshape(-1,784), np.array(labels)

if __name__=="__main__":    
    images_train, labels_train = load_mnist("training")
    n_estimators = 50

    clf = RandomForestClassifier(n_estimators = n_estimators, max_features=28, criterion="gini")
    t0 = time()
    print("Starting the training...")
    clf.fit(images_train, labels_train)
    print("Training finished in %ss"%(round(time()-t0,2)))

    images_test, labels_test = load_mnist("testing")
    accuracy = clf.score(images_test, labels_test)
    print(accuracy)
    
    filename = "forestClassifier_"
    joblib.dump(clf,filename+str(n_estimators)+".pkl")
    print("Pickled the classifier to: " + filename+str(n_estimators)+".pkl")
