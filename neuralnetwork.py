import os
import numpy as np
import json

def softmax(preacts):
    m = np.max(preacts, axis=0, keepdims=True)
    e = np.exp(preacts - m)
    return e / np.sum(e, axis=0, keepdims=True)

class NeuralNetwork:
    def __init__(self, layerSizes, saveFile):
        self._layerSizes = layerSizes
        self._saveFile = saveFile
        self._weights = []
        self._biases = []

        self._recentWeights = []
        self._recentBiases = []

        if os.path.exists(self._saveFile):
            self.load()
        else:
            self.initRandomWeights()

    @property
    def layerSizes(self):
        return self._layerSizes

    @property
    def saveFile(self):
        return self._saveFile

    @property
    def weights(self):
        return self._weights

    @property
    def biases(self):
        return self._biases

    def initRandomWeights(self):
        self._weights = []
        self._biases = []
        for i in range(len(self._layerSizes) - 1):
            insize = self._layerSizes[i]
            outsize = self._layerSizes[i + 1]
            HeInit = np.sqrt(2.0 / insize)
            W = np.random.randn(outsize, insize) * HeInit
            b = np.zeros((outsize, 1))
            self._weights.append(W)
            self._biases.append(b)

    def forprop(self, x):
        a = x
        for i in range(len(self._weights) - 1):
            z = self._weights[i] @ a + self._biases[i]
            a = np.maximum(z, 0.0)
        final = self._weights[-1] @ a + self._biases[-1]
        return softmax(final)

    def backprop(self, inp, exp):
        a = inp
        activations = [inp]
        preacts = []
        for i in range(len(self._weights) - 1):
            z = self._weights[i] @ a + self._biases[i]
            preacts.append(z)
            a = np.maximum(z, 0.0)
            activations.append(a)

        z = self._weights[-1] @ a + self._biases[-1]
        preacts.append(z)
        exp_vec = np.zeros((3, 1), dtype=np.float32)
        exp_vec[int(exp)] = 1.0
        p = softmax(z)
        delta = p - exp_vec
        dW = [delta @ activations[-1].T]
        db = [delta]

        for i in range(len(self._weights) - 2, -1, -1):
            delta = self._weights[i + 1].T @ delta
            delta = delta * (preacts[i] > 0).astype(np.float32)
            dW.insert(0, delta @ activations[i].T)
            db.insert(0, delta)
        return dW, db

    def record_grads(self, dW, db):
        if len(self._recentWeights) == 10:
            self._recentWeights = []
            self._recentBiases = []
        self._recentWeights.append(dW)
        self._recentBiases.append(db)

    def apply_grads(self, lr):
        if len(self._recentWeights) != 10:
            return

        sum_dW = [np.zeros_like(w) for w in self._weights]
        sum_db = [np.zeros_like(b) for b in self._biases]

        for dW in self._recentWeights:
            for i in range(len(sum_dW)):
                sum_dW[i] += dW[i]

        for db in self._recentBiases:
            for i in range(len(sum_db)):
                sum_db[i] += db[i]

        avg_dW = [dw / 10 for dw in sum_dW]
        avg_db = [db / 10 for db in sum_db]

        for i in range(len(self._weights)):
            self._weights[i] -= lr * avg_dW[i]
            self._biases[i] -= lr * avg_db[i]

    def save(self):
        with open(self._saveFile, "w") as f:
            json.dump({
                "weights": [w.tolist() for w in self._weights],
                "biases": [b.tolist() for b in self._biases]
            }, f)

    def load(self):
        with open(self._saveFile, "r") as f:
            data = json.load(f)
            self._weights = [np.array(w) for w in data["weights"]]
            self._biases = [np.array(b) for b in data["biases"]]
