import os
import numpy as np
import json

def softmax(preacts):
    m = np.max(preacts, axis=0, keepdims=True)
    e = np.exp(preacts - m)
    return e / np.sum(e, axis=0, keepdims=True)

class NeuralNetwork:
    def __init__(self, layerSizes, saveFile):
        self.layerSizes = layerSizes
        self.saveFile = saveFile
        self.weights = []
        self.biases = []

        if os.path.exists(self.saveFile):
            self.load()
        else:
            self.initRandomWeights()

    def initRandomWeights(self):
        self.weights = []
        self.biases = []
        for i in range(len(self.layerSizes) - 1):
            numsIn = self.layerSizes[i]
            numsOut = self.layerSizes[i+1]
            HeInit = np.sqrt(2.0 / numsIn)
            W = np.random.randn(numsOut, numsIn) * HeInit
            b = np.zeros((numsOut, 1))
            self.weights.append(W)
            self.biases.append(b)

    def forprop(self, x,):
        a = x
        for i in range(len(self.weights) - 1):
            z = self.weights[i] @ a + self.biases[i]
            a = np.maximum(z, 0.0)
        raw_out = self.weights[-1] @ a + self.biases[-1]
        return softmax(raw_out)
    
    def backprop(self, inp, exp):
        a = inp
        activations = [inp]
        preact = []
        for i in range(len(self.weights) - 1):
            z = self.weights[i] @ a + self.biases[i]
            preact.append(z)
            a = np.maximum(z, 0.0)
            activations.append(a)
        z = self.weights[-1] @ a + self.biases[-1]
        preact.append(z)
        exp_vec = np.zeros((3, 1), dtype=np.float32)
        exp_vec[int(exp)] = 1.0
        p = softmax(z)
        delta = p - exp_vec

        deltaw = [delta @ activations[-1].T]
        deltab = [delta]
        for i in range(len(self.weights) - 2, -1, -1):
            delta = self.weights[i + 1].T @ delta
            delta = delta * (preact[i] > 0).astype(np.float32)
            deltaw.insert(0, delta @ activations[i].T)
            deltab.insert(0, delta)
        return deltaw, deltab

    def save(self):
        with open(self.saveFile, "w") as f:
            json.dump({"weights": [w.tolist() for w in self.weights], "biases": [b.tolist() for b in self.biases]}, f)

    def load(self):
        with open(self.saveFile, "r") as f:
            data = json.load(f)
            self.weights = [np.array(w) for w in data["weights"]]
            self.biases = [np.array(b) for b in data["biases"]]