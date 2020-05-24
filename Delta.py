import os
import random
import numpy as np
from PIL import Image

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoidDeriv(x):
    value = sigmoid(x)
    return value * (1 - value)


def train(data, stop, maxError):
    N, n = data.shape
    learningRate = 0.8
    globalErrorCount = 0
    weights = np.random.randn(1, n - 1)
    gradXErrors = []

    for epoch in range(0, stop):
        globalErrorCount = 0
        for case in data:
            tempData = []
            gradXErrors = []
            caseData = case[:-1]
            xWeights = np.matmul(weights, case[:-1])
            xWeights = xWeights[0]
            output = sigmoid(xWeights)
            error = output - case[-1]
            globalErrorCount += (error ** 2)
            gradXErrors.append(2 * error * sigmoidDeriv(xWeights))
            tempData.append(caseData)

        tempData = np.asarray(tempData)
        gr = tempData * np.asarray(gradXErrors)

        weights = weights - learningRate * gr

        if globalErrorCount < maxError:
            break

    print(globalErrorCount)
    return weights


def loadImages(path, label):
    data = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            im = Image.open(dirpath + '\\' + filename, 'r').convert('LA').resize((64, 64))
            pixelValues = [x[0] / 255.0 for x in im.getdata()]

            pixelValues.append(1)
            pixelValues.append(label)

            data.append(np.array(pixelValues))

    return np.asarray(data)


def main():
    
    trainingSize = 700

    data1 = loadImages("./brain", 0)
    trainingData = data1[0:trainingSize]
    testData = data1[trainingSize:]

    data2 = loadImages("./bonsai", 1)
    trainingData2 = data2[0:trainingSize]
    testData2 = data2[trainingSize:]

    trainingData = np.concatenate((trainingData, trainingData2))
    np.random.shuffle(trainingData)

    weights = train(trainingData, 10000, 2)


if __name__ == '__main__':
    main()