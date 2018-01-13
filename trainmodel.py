import numpy as np
from simplenet import simplenet

epochs = 1000
inputWidth = 8
inputHeight = 6

modelName = 'simplenet.model'

model = simplenet()

savedTrainSet = np.load("trainSet.npy")
trainSet = savedTrainSet[:-750]
valSet = savedTrainSet[-750:]

# each image is reshaped from a matrix into a single array
trainImageSet = np.array([i[0] for i in trainSet]).reshape(-1, inputWidth, inputHeight, 1)
trainSolutionSet = np.array([i[1] for i in trainSet])

valImageSet = np.array([i[0] for i in valSet]).reshape(-1, inputWidth, inputHeight, 1)
valSolutionSet = np.array([i[1] for i in valSet])

model.fit({'input': trainImageSet}, {'targets': trainSolutionSet}, n_epoch=epochs, validation_set=({'input': valImageSet}, {'targets': valSolutionSet}),
          snapshot_step=500, show_metric=True, run_id=modelName)

savedEvalSet = np.load("evalSet.npy")
evalImageSet = np.array([i[0] for i in savedEvalSet]).reshape(-1, inputWidth, inputHeight, 1)
evalSolutionSet = np.array([i[1] for i in savedEvalSet])

print(model.evaluate(evalImageSet, evalSolutionSet))
print(model.evaluate(valImageSet, valSolutionSet))
print(model.evaluate(trainImageSet,trainSolutionSet))
model.save(modelName)
