import numpy as np
from simplenet import simplenet

epochs = 1
inputWidth = 128
inputHeight = 72
savedTrainSet = np.load("trainSet.npy")
trainSet = savedTrainSet[:-1500]
valSet = savedTrainSet[-1500:]
# each image is reshaped from a matrix into a single array
trainImageSet = np.array([i[0] for i in trainSet]).reshape(-1, inputWidth, inputHeight, 1)
trainSolutionSet = np.array([i[1] for i in trainSet])

valImageSet = np.array([i[0] for i in valSet]).reshape(-1, inputWidth, inputHeight, 1)
valSolutionSet = np.array([i[1] for i in valSet])

modelName = 'simplenet.model'
model = simplenet()


savedEvalSet = np.load("evalSet.npy")
evalImageSet = np.array([i[0] for i in savedEvalSet]).reshape(-1, inputWidth, inputHeight, 1)
evalSolutionSet = np.array([i[1] for i in savedEvalSet])


for epochs in range (1,100):
    model.fit({'input': trainImageSet}, {'targets': trainSolutionSet}, n_epoch=1, validation_set=({'input': valImageSet}, {'targets': valSolutionSet}),
              snapshot_step=500, show_metric=True, run_id=modelName)


    evalScore = model.evaluate(evalImageSet, evalSolutionSet)[0]
    print(evalScore)

    valScore = model.evaluate(valImageSet, valSolutionSet)[0]
    print(valScore)

    if(evalScore > 0.1) and (valScore> 0.1):
        print("{},{}".format(format(evalScore, '.4f'),format(valScore, '.4f')))
        f = open('epochslog.csv','a')
        f.write("{},{},{}\n".format(format(epochs),format(evalScore, '.4f'),format(valScore, '.4f')))
        f.close()
        model.save(modelName)
