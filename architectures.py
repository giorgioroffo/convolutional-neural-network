# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

# build_cnn_architecture
def build_cnn_architecture():
    # Initialising the CNN
    net = Sequential()
      
    # Step 1 - Convolution
    input_size = (128, 128)
    net.add(Conv2D(32, (3, 3), input_shape=(*input_size, 3), activation='relu'))
      
    # Step 2 - Pooling
    net.add(MaxPooling2D(pool_size=(2, 2)))  # 2x2 is optimal
  
    # Adding a second convolutional layer
    net.add(Conv2D(32, (3, 3), activation='relu'))
    net.add(MaxPooling2D(pool_size=(2, 2)))
  
    # Adding a third convolutional layer
    net.add(Conv2D(64, (3, 3), activation='relu'))
    net.add(MaxPooling2D(pool_size=(2, 2)))
  
    # Step 3 - Flattening
    net.add(Flatten())
  
    # Step 4 - Full connection
    net.add(Dense(units=64, activation='relu'))
    net.add(Dropout(0.5))
    net.add(Dense(units=1, activation='sigmoid'))
     
    # Compiling the CNN
    net.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
 
    return net




