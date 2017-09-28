import perceptron
import datasets as data
import runClassifier as run
import pylab
from numpy import *

#13
curve = run.learningCurveSet(perceptron.Perceptron({'numEpoch':10}),data.SentimentData)
run.plotCurve("Perceptron Learning Curve",curve)

#14
h = perceptron.Perceptron({'numEpoch':10})
h.train(data.TwoDDiagonal.X,data.TwoDAxisAligned.Y)
run.plotData(data.TwoDDiagonal.X,data.TwoDAxisAligned.Y)
run.plotClassifier(h.weights,h.bias)
pylab.show()


#15
class DummyData:
    X = array([[-1,2],[1,2],[-1,-2],[1,-2],[-2,1],[-2,-1],[2,1],[2,-1]])
    Y = array([1,1,1,1,-1,-1,-1,-1])

    Xte = X
    Yte = Y

h = perceptron.Perceptron({'numEpoch':10})
h.train(DummyData.X,DummyData.Y)
run.plotData(DummyData.X,DummyData.Y)
run.plotClassifier(h.weights,h.bias)
pylab.show()

curve = run.hyperparamCurveSet(h,'numEpoch', [1,2,3,4,5,6,7,8,9,10],DummyData)
run.plotCurve("Perceptron Learning Curve",curve)

curve = run.learningCurveSet(h,DummyData)
run.plotCurve("Learning Curve for NOn-converge",curve)



#16

class bsX:

    X = array([[10,-1],[10,-0.5],[10,0],[10,0.5],[10,0.9],[10,1],[1,-1], [1,-0.05], [1,0], [1,0.80], [1,0.89], [1, 1]])
    Y= array([1, 1,1,1,1,1, -1,-1, -1, -1, -1, -1])


    Xte = X
    Yte = Y

h = perceptron.Perceptron({'numEpoch':10})
h.train(bsX.X,bsX.Y)
run.plotData(bsX.X,bsX.Y)
run.plotClassifier(h.weights,h.bias)
pylab.show()

curve = run.hyperparamCurveSet(h,'numEpoch', [1,2,3,4,5,6,7,8,9,10],bsX)

run.plotCurve("Perceptron Prediction",curve)