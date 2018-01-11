import numpy as np
from simplenet import simplenet
from tflearn.data_utils import to_categorical

WIDTH  = 8
HEIGHT = 6
LR = 1e-3
EPOCHS = 30

MODEL_NAME = 'simplenet.model'

model = simplenet(WIDTH,HEIGHT,LR)

savedTrainingData = np.load("trainingData.npy")
savedValidationData = np.load("validationData.npy")


# each image is reshaped from a 60x80 matrix into a single array
X = np.array([i[0] for i in savedTrainingData]).reshape(-1, WIDTH, HEIGHT, 1)
#X = to_categorical(X, nb_classes=1)


Y = np.array([i[1] for i in savedTrainingData])
#Y = to_categorical(Y, nb_classes=1)

test_x = np.array([i[0] for i in savedValidationData]).reshape(-1, WIDTH, HEIGHT, 1)
#test_x = to_categorical(test_x, nb_classes=3)

test_y = np.array([i[1] for i in savedValidationData])
#test_y = to_categorical(test_y, nb_classes=3)


model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}),
          snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

#model.evaluate(test_x,test_y )

model.save(MODEL_NAME)
