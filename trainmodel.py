import numpy as np
from simplenet import simplenet

epochs = 20
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

#trainSet = savedTrainSet[0:750]
#valSet = savedTrainSet[750:1500]

model.fit({'input': trainImageSet}, {'targets': trainSolutionSet}, n_epoch=epochs, validation_set=({'input': valImageSet}, {'targets': valSolutionSet}),
          snapshot_step=500, show_metric=True, run_id=modelName)

savedEvalSet = np.load("evalSet.npy")
evalImageSet = np.array([i[0] for i in savedEvalSet]).reshape(-1, inputWidth, inputHeight, 1)
evalSolutionSet = np.array([i[1] for i in savedEvalSet])

evalScore = model.evaluate(evalImageSet, evalSolutionSet)[0]
print(evalScore)
valScore = model.evaluate(valImageSet, valSolutionSet)[0]
print(valScore)

#print("{}, {} \n").format((format(evalScore, '.0f'),format((valScore, '.0f')))) #Give your csv text here.
print("{},{}".format(format(evalScore, '.4f'),format(valScore, '.4f')))
f = open('csvfile.csv','a')
f.write("{},{}\n".format(format(evalScore, '.4f'),format(valScore, '.4f')))
f.close()
model.save(modelName)
model.session = 0
