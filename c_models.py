from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D, ZeroPadding3D


def get_model_tf(summary=False):
    """ Return the Keras model of the network
    """
    model = Sequential()
    # 1st layer group
    model.add(Conv3D(64, 3, 3, 3, activation='relu', padding='same', name='conv1', strides=(1, 1, 1), input_shape=(16, 112, 112, 3)))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid', name='pool1'))
    # 2nd layer group
    model.add(Conv3D(128, 3, 3, 3, activation='relu', padding='same', name='conv2', strides=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool2'))
    # 3rd layer group
    model.add(Conv3D(256, 3, 3, 3, activation='relu', padding='same', name='conv3a', strides=(1, 1, 1)))
    model.add(Conv3D(256, 3, 3, 3, activation='relu', padding='same', name='conv3b', strides=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool3'))
    # 4th layer group
    model.add(Conv3D(512, 3, 3, 3, activation='relu', padding='same', name='conv4a', strides=(1, 1, 1)))
    model.add(Conv3D(512, 3, 3, 3, activation='relu', padding='same', name='conv4b', strides=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool4'))
    # 5th layer group
    model.add(Conv3D(512, 3, 3, 3, activation='relu', padding='same', name='conv5a', strides=(1, 1, 1)))
    model.add(Conv3D(512, 3, 3, 3, activation='relu', padding='same', name='conv5b', strides=(1, 1, 1)))
    model.add(ZeroPadding3D(padding=(0, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool5'))
    model.add(Flatten())
    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(.5))
    model.add(Dense(487, activation='softmax', name='fc8'))
    if summary:
        print(model.summary())
    return model



def get_model_tf_no_final(summary=False):
    """ Return the Keras model of the network
    """
    model = Sequential()
    # 1st layer group
    model.add(Conv3D(64, 3, 3, 3, activation='relu', padding='same', name='conv1', strides=(1, 1, 1), input_shape=(16, 112, 112, 3)))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid', name='pool1'))
    # 2nd layer group
    model.add(Conv3D(128, 3, 3, 3, activation='relu', padding='same', name='conv2', strides=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool2'))
    # 3rd layer group
    model.add(Conv3D(256, 3, 3, 3, activation='relu', padding='same', name='conv3a', strides=(1, 1, 1)))
    model.add(Conv3D(256, 3, 3, 3, activation='relu', padding='same', name='conv3b', strides=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool3'))
    # 4th layer group
    model.add(Conv3D(512, 3, 3, 3, activation='relu', padding='same', name='conv4a', strides=(1, 1, 1)))
    model.add(Conv3D(512, 3, 3, 3, activation='relu', padding='same', name='conv4b', strides=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool4'))
    # 5th layer group
    model.add(Conv3D(512, 3, 3, 3, activation='relu', padding='same', name='conv5a', strides=(1, 1, 1)))
    model.add(Conv3D(512, 3, 3, 3, activation='relu', padding='same', name='conv5b', strides=(1, 1, 1)))
    model.add(ZeroPadding3D(padding=(0, 1, 1), name='zeropad5'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool5'))
    model.add(Flatten())
    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    if summary:
        print(model.summary())
    return model
