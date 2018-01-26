import tflearn
from tflearn.layers.estimator import regression

def simplenet():

    width = 128
    height = 72
    lr = 0.001
    #one input layer with 80x60=4800 Neurons
    net = tflearn.input_data(shape=[None, width,height,1], name='input')
    #one hidden layer with 10 neurons
    net = tflearn.fully_connected(net, 10)
    net = tflearn.fully_connected(net, 10)
    #net = tflearn.fully_connected(net, 10)
    #net = tflearn.fully_connected(net, 10)

    #one output layer with 3 neurons
    net = tflearn.fully_connected(net, 3, activation='softmax')
    net = regression(net, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=lr, name='targets')

    model = tflearn.DNN(net, checkpoint_path='model_simplenet',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='log')
    return model