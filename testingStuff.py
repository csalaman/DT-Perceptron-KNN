import dumbClassifiers as du
import datasets as data
import runClassifier as run
import numpy


def runTest(d):

    print "AlwaysPredictOne: "
    h = du.AlwaysPredictOne({})
    run.trainTestSet(h,d)

    print "AlwaysPredictMostFrequent:"
    h = du.AlwaysPredictMostFrequent({})
    run.trainTestSet(h,d)

    print "FirstFeatureClassifier:"
    h = du.FirstFeatureClassifier({})
    run.trainTestSet(h,d)



runTest(data.TennisData)
print '\n'
runTest(data.SentimentData)



# h.train(d.X,d.Y)
# print numpy.mean((d.Y > 0) == (h.predictAll(d.X) >0))

# h = du.AlwaysPredictOne({})
# run.trainTestSet(h,d)

