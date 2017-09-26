import dumbClassifiers as du
import datasets as data
import runClassifier as run
import numpy
import dt

# Test the ZDecision Tree
curve = run.learningCurveSet(dt.DT({'maxDepth':6}),data.SentimentData)
run.plotCurve('Decision Tree Learning Curve on Sediment Data',curve)

curve = run.hyperparamCurveSet(dt.DT({}), 'maxDepth', [1,2,4,6,8,12,16],data.SentimentData)
run.plotCurve('Decision Tree Hyperparameter Curve on Sediment Data',curve)