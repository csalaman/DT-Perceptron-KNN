import dumbClassifiers as du
import datasets as data
import runClassifier as run
import numpy
import knn

#9
curve = run.learningCurveSet(knn.KNN({'isKNN':True,'K':5}),data.DigitData)
run.plotCurve('K-Nearest Neighbor on 5-NN; DIgitsData',curve)