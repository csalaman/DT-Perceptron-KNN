
import datasets as data
import runClassifier as run
import numpy
import perceptron

curve = run.learningCurveSet(perceptron.Perceptron({'numEpoch':10}),data.TwoDDiagonal)
run.plotCurve("Perceptron Learning Curve on Sentiment Data",curve)

