import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import random 
import math
# the data used in this program can be found at this link
# for object name file1. to store the weghts and learning rates
file1 = open("MyFile.txt","a")
class dataCleaningandSplitting:
    global minmax
    minmax= list()
    def __init__(self,data) -> None:
        self.data=data
    def dataCleaning(self):
        data.info()
        #data.dropna() # we do not need this as our data does not contain any kind null data
        for i in range (len(data)):
            if data.loc[i,"X5"]>10 or data.loc[i,"X5"]<1:
                data.loc[i,"X5"]=data.mode()  
    def plotting(self):
        x=[]
        y=[]
        x=list(data["X3"])
        y=list(data['Y1'])
        print(x)
        print(y)
        #data.plot(x,y)
        plt.show()
    #max and min in the dataset
    def dataset_minmax(self):
        stats=[]
        for column in (data.columns):
            stats.append([min(data[column]),max(data[column])])
        return stats
    # Rescale dataset columns to the range 0-1
    def normalize_dataset(self):
        minmax=self.dataset_minmax()
        #convertinf all the data integers into floats
        for column in data.columns:
            data[column] = data[column].astype(float)
        """figure1=plt.figure()
        plt.plot(list(data["X5"]),list(data['Y2']))
        plt.xlabel("floor area")
        plt.ylabel=("heat load")
        plt.title("energy effciency")
        plt.grid()
        plt.show()"""
        #data.plot(kind='scatter', x=data["X3"],y=data['Y1'])
        for x in range(len(data)):
            for y in range(len(data.columns)):
                data.loc[x,data.columns[y]] = (data.loc[x,data.columns[y]] - minmax[y][0]) / (minmax[y][1] - minmax[y][0])
        figure2=plt.figure()
        """plt.plot(list(data["X5"]),list(data['Y2']))
        plt.xlabel("floor area")
        plt.ylabel=("heat load")
        plt.title("energy effciency")
        plt.grid()
        plt.show()"""
    def trainig_data(self):
        self.normalize_dataset()
        trainingSet=np.matrix(data.iloc[:550,:])
        trainingSet=np.asfarray(trainingSet, int)
        return  trainingSet
    def testing_data(self):
        self.normalize_dataset()
        testingSet=np.matrix(data.iloc[550:,:])
        testingSet=np.asfarray(testingSet, int)
        return testingSet
    
    
   
class Backpropagation:
    def initialize(self, nInputs, nHidden, nOutputs):
        network = list()
        hiddenLayer = [{'weights': [random() for i in range(nInputs + 1)]} for i in range(nHidden)]
        network.append(hiddenLayer)
        outputLayer = [{'weights': [random() for i in range(nHidden + 1)]} for i in range(nOutputs)]
        network.append(outputLayer)
        return network

    # Propagate forward
    def activate(self, inputs, weights):
        activation = weights[-1]
        for i in range(len(weights) - 1):
            activation += weights[i] * inputs[i]
        return activation

    def transfer(self, activation):
        return 1.0 / (1.0 + math.exp(-activation))

    def forwardPropagate(self, network, row):
        inputs = row
        for layer in network:
            newInputs = []
            print(layer)
            for neuron in layer:
                activation = self.activate(inputs, neuron['weights'])
                neuron['output'] = self.transfer(activation)
                newInputs.append(neuron['output'])
            inputs = newInputs
        return inputs

    # Propagate backwards
    def transferDerivative(self, output):
        return output * (1.0 - output)

    def backwardPropagateError(self, network, expected):
        for i in reversed(range(len(network))):
            layer = network[i]
            errors = list()
            if (i != len(network) - 1):
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * self.transferDerivative(neuron['output'])

    # For train network
    def updateWeights(self, network, row, learningRate, nOutputs):
        nOutputs = nOutputs * -1
        for i in range(len(network)):
            inputs = row[:nOutputs]
            if (i != 0):
                inputs = [neuron['output'] for neuron in network[i - 1]]
            for neuron in network[i]:
                for j in range(len(network[i])):
                    neuron['weights'][j] += learningRate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += learningRate * neuron['delta']

    def updateLearningRate(self, learningRate, decay, epoch):
        return learningRate * 1 / (1 + decay * epoch)

    def trainingNetwork(self, network, train, learningRate, nEpochs, nOutputs, expectedError):
        sumError = 10000.0
        for epoch in range(nEpochs):
            if (sumError <= expectedError):
                break
            if(epoch % 100 == 0):
                learningRate = self.updateLearningRate(learningRate, learningRate/nEpochs, float(epoch))

            sumError = 0
            for row in train:
                outputs = self.forwardPropagate(network, row)
                expected = self.getExpected(row, nOutputs)
                sumError += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
                self.backwardPropagateError(network, expected)
                self.updateWeights(network, row, learningRate, nOutputs)
            print('> epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learningRate, sumError))

    def getExpected(self, row, nOutputs):
        expected = []
        for i in range(nOutputs):
            temp = (nOutputs - i) * - 1
            expected.append(row[temp])
        return expected
    # For predict result
    def predict(self, network, row):
        outputs = self.forwardPropagate(network, row)
        return outputs
data=pd.read_excel("C:/Users/Admin/OneDrive/Yahya/energy effieciency .xlsx")
#dataCleaningandSplitting(data).plotting()
#print(dataCleaningandSplitting(data).dataCleaning())
#print(dataCleaningandSplitting(data).normalize_dataset())

#back propagation 
nOutputs = int(input('Insert the number Neurons into Output Layer: '))
nEpochs = int(input('Insert the number of Epochs: '))
nHiddenLayer = int(input('Insert the number Neurons into Hidden Layer: '))
learningRate = float(input('Insert Learning Rate: '))
expectedError = float(input('Insert Expected Error: '))
###
backpropagation = Backpropagation()
nInputs = len(dataCleaningandSplitting(data).trainig_data()[0]) - nOutputs
network = backpropagation.initialize(nInputs, nHiddenLayer, nOutputs)
backpropagation.forwardPropagate(network,nInputs)
"""backpropagation.trainingNetwork(network, dataCleaningandSplitting(data).trainig_data(), learningRate, nEpochs, nOutputs, expectedError)
def writeWieghtsToFile(network):
    my_df = pd.DataFrame(network)
    my_df.to_csv('my_array.csv',header = False, index= False)
writeWieghtsToFile(network)
input('\nPress enter to view Result...')


##testing the the network
for row in dataCleaningandSplitting(data).testing_data():
    prediction = backpropagation.predict(network, row)
    # print('Input =', (row), 'Expected = ', backpropagation.getExpected(row, nOutputs), 'Result =', (prediction))
    print('Expected = ', backpropagation.getExpected(row, nOutputs), 'Result =', (prediction))
"""