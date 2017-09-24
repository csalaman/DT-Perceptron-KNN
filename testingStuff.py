import dumbClassifiers as du
import datasets as data
import runClassifier as run
import numpy

d = data.TennisData

print("AlwaysPredictOne:")
h = du.AlwaysPredictOne({})
run.trainTestSet(h,d)
print "\n"

print("AlwaysPredictMostFrequent:")
h = du.AlwaysPredictMostFrequent({})
run.trainTestSet(h,d)
print "\n"

print("FirstFeatureClassifier:")
print("FirstFeatureClassifier:")
h = du.FirstFeatureClassifier({})
run.trainTestSet(h,d)


# h.train(d.X,d.Y)
# print numpy.mean((d.Y > 0) == (h.predictAll(d.X) >0))

# h = du.AlwaysPredictOne({})
# run.trainTestSet(h,d)

