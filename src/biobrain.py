import numpy as np
import json

import utils
import activation

class BiobrainException(BaseException):
    """BiobrainException"""

    def __init__(self, error):
        super(BiobrainException, self).__init__()
        self.error = error


class NeuralNetwork:
    """NeuralNetwork"""

    defaultActivation = 'sigmoid'

    def __init__(self, activation=defaultActivation):
        self._activation = activation
        self._network = (
            [np.random.randn(), np.random.randn()], # weigths
            (np.random.randn())                     # biais
        )

    def train(self, trainingList, learningRate=0.1, chunkSize=0, maxIterations=0):
        self.trainingList = trainingList

        if chunkSize > 0:
            trainingList = utils.chunk(trainingList, chunkSize)

        costs = []
        i = 0

        for subTrainingList in trainingList:
            self._train(subTrainingList, learningRate)
            costs.append(self.getMeanCost(subTrainingList))
            i += 1
            if maxIterations > 0 and i == maxIterations:
                break;

        return costs

    def evaluate(self, inputs):
        return self._activate(self._accumulate(inputs))

    def getMeanCost(self, trainingList):
        cost = 0
        for trainingData in trainingList:
            targetInputs, targetOutputs = trainingData
            cost += self._calcCost(self.evaluate(targetInputs), targetOutputs[0])
        return cost / len(trainingList)

    def save(self, filename, meanPrecision=100):
        try:
            with open(filename, 'w+') as file:
                file.write(json.dumps([self._activation, self._network, self.getMeanCost(self.trainingList[:meanPrecision])]))
                print('Brain saved at \'' + filename + '\'')
        except PermissionError:
            raise BiobrainException('Oops.. Permission denied!')

    def load(self, filename):
        try:
            with open(filename, 'r') as file:
                self._activation, self._network, meanCost = json.load(file)
                print('Brain loaded from \'' + filename + '\'\nEstimated mean cost: ' + str(meanCost))
        except FileNotFoundError:
            raise BiobrainException('Oops.. File not found!')

    def _train(self, trainingList, learningRate):
        for trainingData in trainingList:
            targetInputs, _ = trainingData

            signal = self._accumulate(targetInputs)
            evaluation = self._activate(signal)

            self._learn(trainingData, signal, evaluation, learningRate)

    def _learn(self, trainingData, signal, evaluation, learningRate):
        targetInputs, targetOutputs = trainingData
        costD_predD     = self._calcCostD(targetOutputs[0], evaluation)
        predD_signalD   = self._activate(signal, True)
        costD_zD        = costD_predD * predD_signalD

        def calibrate(value, zD_valueD):
            costD_valueD = costD_zD * zD_valueD
            return value - learningRate * costD_valueD

        def calibrateNeuron(neuron):
            weigths, biais  = self._network
            newWeights      = [calibrate(w, p) for w, p in zip(weigths, targetInputs)]
            biais           = calibrate(biais, 1)

            return newWeights, biais

        self._network = calibrateNeuron(self._network)

    def _calcCost(self, targetOutput, evaluation):
        return np.square(evaluation - targetOutput)

    def _calcCostD(self, targetOutput, evaluation):
        return 2 * (evaluation - targetOutput)

    def _accumulate(self, data):
        weigths, biais = self._network
        return sum([w * d for w, d in zip(weigths, data)]) + biais

    def _activate(self, signal, derivate=False):
        function, derivative = activation.functions.get(self._activation, self.defaultActivation)

        if (derivate):
            return derivative(signal)

        return function(signal)
