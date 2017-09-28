import dumbClassifiers as du
import datasets as data
import runClassifier as run
import numpy
import knn

# #9
# curve = run.trainTestSet(knn.KNN({'isKNN':True,'K':5}),data.SentimentData)
# run.plotCurve('K-Nearest Neighbor on 5-NN; DIgitsData',curve)


run.trainTestSet(knn.KNN({'isKNN':False,'eps':0.3}),data.SentimentData)