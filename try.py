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
    
    def testing(self):
        self.normalize_dataset()
        testingSet=np.matrix(data)
        testingSet=np.asfarray(testingSet, int)
        return testingSet
   
class Backpropagation:
    def initialize(self, nInputs, nHidden, nOutputs):
        network = list()
        hiddenLayer =[ {'weights': [-0.7366323621711862, -4.49097085123432, -28.54654531839023, 19.373164789387104, -1.0303448421628223, 0.027669632796500438, -0.7390958120087957, 0.05318227612272745, 12.383790920379612], 'output': 1.1376510402130908e-08, 'delta': -1.50913053409066e-10}\
        ,{'weights': [-3.5477335151233564, -2.745456401346205, 0.7775863677154443, -2.715903985792684, -3.4762815639169364, 0.058771164638413635, -1.2588032321410136, -0.07403383016098276, 6.518031556734741], 'output': 0.3809164151007468, 'delta': -0.008485605904016645}\
        ,{'weights': [-1.694387470622674, -1.5266036540587342, -2.686273166392583, -1.1613342614609221, 0.6148844748370678, 7.876959876983966, 0.5786421569657115, -5.078922864450515, -4.1566073216920385], 'output': 0.0037183334856952244, 'delta': 4.971373133069474e-06}\
        ,{'weights': [3.6314514409132603, -3.2223368212802965, -3.600859916520262, -1.0599034373195906, 1.8902446547063145, 1.4096197081797177, -0.5812811340341704, -0.5123082369919456, -0.05644278940371786], 'output': 0.091318608878092, 'delta': 2.1264537372038006e-05}\
        ,{'weights': [5.743835265871421, -2.2887306636487157, -12.892234229501213, 7.871740332950767, -1.9112172024195848, 0.3669167625781644, -2.034070327196821, -0.40490807956631336, 3.3828816182660812], 'output': 1.065649882660366e-05, 'delta': 1.35964483075543e-08}\
        ,{'weights': [-13.53690588928662, 7.965973076469028, 22.013430627952435, -8.161329657548329, 1.4813141310181916, 0.02990081437928742, 0.8513419735031575, 0.4093452589229939, -7.437876763859067], 'output': 0.999999630777417, 'delta': 1.1315949278132708e-09}\
        ,{'weights': [-4.0797294739044085, 3.6221066991682846, 10.490935886805003, -3.9937275807395585, 5.370495126044702, 0.4145484902830487, 1.765488682955673, 1.420109288290127, -1.5266904083111381], 'output': 0.9999999168964517, 'delta': 2.5771366853345447e-10}\
        ,{'weights': [-12.342496725532312, 8.23074606131437, 28.32877840549726, -11.405532855393853, -0.37293990503275515, 0.23460835466247465, -0.1504564445034528, -0.07503386492792558, -7.566545693710697], 'output': 0.9999999892704493, 'delta': 3.487165050343497e-11}]
        network.append(hiddenLayer)
        outputLayer =[ {'weights': [-1.6605831974685625, -4.492355146299357, 0.14884275786776757, 0.006001536855830714, 0.1034500571484761, 0.49544613858342057, 0.28437601409580027, 0.26728393724353905, 2.5244764557349852], 'output': 0.8654042837493412, 'delta': 0.0058480260655592595}\
            ,{'weights': [-1.6040340390861492, -4.382610980623644, 0.2128113719808521, 0.0998137593998456, 0.302790986199051, 0.07555958273283869, 0.6490161553922325, 0.7613452263877252, 1.5176296971548986], 'output': 0.7931211055494514, 'delta': 0.0022157799997072902}]

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
    def predictY1(self, network, row):
        outputs = self.forwardPropagate(network, row)
        return outputs[0]
    def predictY2(self, network, row):
        outputs = self.forwardPropagate(network, row)
        return outputs[1]
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
backpropagation.trainingNetwork(network, dataCleaningandSplitting(data).trainig_data(), learningRate, nEpochs, nOutputs, expectedError)
input('\nPress enter to view Result...')
##testing the the network
"""for row in dataCleaningandSplitting(data).testing_data():
    prediction = backpropagation.predict(network, row)
    # print('Input =', (row), 'Expected = ', backpropagation.getExpected(row, nOutputs), 'Result =', (prediction))
    print('Expected = ', backpropagation.getExpected(row, nOutputs), 'Result =', (prediction))
"""
def writeWieghtsToFile(network):
    my_df = pd.DataFrame(network)
    my_df.to_csv('my_array.csv',header = False, index= False)
writeWieghtsToFile(network)

Y1=list()
Y2=list()
for row in dataCleaningandSplitting(data).testing():
    Y1.append(backpropagation.predictY1(network, row))
    Y2.append(backpropagation.predictY2(network, row))


fig, ax = plt.subplots()
plt.title("energy effciency")
ax.plot(list(data['Y2']), color = 'green', label = 'Cooling load_real')
ax.plot(Y2, color = 'red', label = 'expected cooling load')
ax.legend(loc = 'upper left')
plt.grid()


fig2 , ax2 = plt.subplots()
plt.title("energy effciency")
ax2.plot(list(data['Y1']), color = 'green', label = 'heating load_real')
ax2.plot(Y1, color = 'red', label = 'expected heating load')
ax2.legend(loc = 'upper left')
plt.grid()
plt.show()