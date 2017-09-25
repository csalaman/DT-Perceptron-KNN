import dumbClassifiers as du
import datasets as data
import runClassifier as run
import numpy
import dt

# Test the ZDecision Tree
curve = run.learningCurveSet(dt.DT({'maxDepth':5}),data.SentimentData)
run.plotCurve("DT Learning Curve on DAta", curve)