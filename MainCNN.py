# Convolutional Neural Network
# @Author: Giorgio Roffo
# @year: 2017
# @Description: Simple implementation of a CNN. 
#               Use this as a template for your CNNs.
from architectures import *
from keras.preprocessing.image import ImageDataGenerator


# Set path to the dataset here...
path_trainset = 'dataset/training_set'
path_testset = 'dataset/test_set'


# Convolutional Neural Network Architecture
CNN_model = build_cnn_architecture()


# Data Augmentation - apply some geometric transformation
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Rescaling the testing set in the same way as the training set -> range 0,1
test_datagen = ImageDataGenerator(rescale = 1./255)

#☺ Read the training set and apply transormations
training_set = train_datagen.flow_from_directory(path_trainset,
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# read and scale the testing set
test_set = test_datagen.flow_from_directory(path_testset,
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'binary')

# fit the model to the images 
CNN_model.fit_generator(training_set,
                         steps_per_epoch = training_set.samples,
                         epochs = 1,
                         validation_data = test_set,
                         validation_steps = test_set.samples)

