import numpy as np
from flask import Flask, request
from sklearn.externals import joblib

import os
import struct
import array

app = Flask(__name__)
model = None

@app.route("/example", methods=["GET"])
def predict_example():
    images,labels = load_mnist()    

    image = np.array(images[0])
    print(labels[0])
    pred = model.predict(image.reshape(1,-1))
    return "%s"%pred[0]

@app.route("/predict", methods=["POST"])
def predict_digit():
    request_data = request.get_json()
    image = np.array(request_data)

    pred = model.predict(image.reshape(1,-1))
    return "%s"%pred

def load_mnist(dataset="training", digits=np.arange(10), path="data", size = 60000):
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    
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

def load_model():
    clf = joblib.load("forestClassifier_50.pkl")

    global model
    model = clf

if __name__=="__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000)
