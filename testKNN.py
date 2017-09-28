import dumbClassifiers as du
import datasets as data
import runClassifier as run
import numpy
import knn

#9
curve = run.learningCurveSet(knn.KNN({'isKNN':True,'K':5}),data.DigitData)
run.plotCurve('K-Nearest Neighbor on 5-NN; DIgitsData',curve)

#11
curve = run.hyperparamCurveSet(knn.KNN({'isKNN':True}), 'K', [1,2,3,4,5,6,7,8,9,10],data.DigitData)
run.plotCurve('Hyperparameter Curve on DigitsData',curve)

#12
arr = []
counter = 1
while counter < 20:
    arr.append(counter)
    counter += .5

curve = run.hyperparamCurveSet(knn.KNN({'isKNN':False}), 'eps', arr ,data.DigitData)
run.plotCurve('Hyperparameter Curve on DigitsData',curve)