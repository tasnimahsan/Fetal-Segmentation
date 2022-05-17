from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Lambda, Dropout, UpSampling2D, BatchNormalization


def createModel(input_size):

    inputs = Input(input_size)

    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    conv10 = BatchNormalization()(conv10)

    model = Model(inputs=inputs, outputs=conv10)

    return model


def createModelUNET(input_size, add_bn=False):

    input_layer = Input(input_size)

    last_layer, start_neurons, rep = input_layer, 64, 4

    conv_layers = []

    for i in range(rep):

        conv = Conv2D(start_neurons * 2**(i+1), (3, 3),
                      activation='relu', padding='same')(last_layer)
        conv = Conv2D(start_neurons * 2**(i+1), (3, 3),
                      activation='relu', padding='same')(conv)
        if add_bn:
            conv = BatchNormalization()(conv)
        pool = MaxPooling2D(pool_size=(2, 2))(conv)
        drop = Dropout(0.5)(pool)
        conv_layers.append(conv)
        last_layer = drop

    conv = Conv2D(start_neurons * 2**4, (3, 3),
                  activation='relu', padding='same')(last_layer)
    conv = Conv2D(start_neurons * 2**4, (3, 3),
                  activation='relu', padding='same')(conv)
    if add_bn:
        conv = BatchNormalization()(conv)
    last_layer = conv

    for i in range(rep):

        conv = Conv2DTranspose(start_neurons * 2**(rep-1-i),
                               (3, 3), strides=(2, 2), padding='same')(last_layer)
        #conv = Conv2D(512, (2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(last_layer))
        merge = concatenate([conv, conv_layers[rep-1-i]])
        drop = Dropout(0.5)(merge)
        conv = Conv2D(start_neurons * 2**(rep-1-i), (3, 3),
                      activation='relu', padding='same')(drop)
        conv = Conv2D(start_neurons * 2**(rep-1-i), (3, 3),
                      activation='relu', padding='same')(conv)
        if add_bn:
            conv = BatchNormalization()(conv)

        last_layer = drop

    output_layer = Conv2D(1, (1, 1), padding="same",
                          activation="sigmoid")(last_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model


def createModelSmall(input_size, add_bn=False):

    input_layer = Input(input_size)

    last_layer, start_neurons, rep = input_layer, 64, 2

    conv_layers = []

    for i in range(rep):

        conv = Conv2D(start_neurons * 2**(i+1), (3, 3),
                      activation='relu', padding='same')(last_layer)
        conv = Conv2D(start_neurons * 2**(i+1), (3, 3),
                      activation='relu', padding='same')(conv)
        if add_bn:
            conv = BatchNormalization()(conv)
        pool = MaxPooling2D(pool_size=(2, 2))(conv)
        drop = Dropout(0.5)(pool)
        conv_layers.append(conv)
        last_layer = drop

    conv = Conv2D(start_neurons * 2**4, (3, 3),
                  activation='relu', padding='same')(last_layer)
    conv = Conv2D(start_neurons * 2**4, (3, 3),
                  activation='relu', padding='same')(conv)
    if add_bn:
        conv = BatchNormalization()(conv)
    last_layer = conv

    for i in range(rep):

        conv = Conv2DTranspose(start_neurons * 2**(rep-1-i),
                               (3, 3), strides=(2, 2), padding='same')(last_layer)
        #conv = Conv2D(512, (2, 2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(last_layer))
        merge = concatenate([conv, conv_layers[rep-1-i]])
        drop = Dropout(0.5)(merge)
        conv = Conv2D(start_neurons * 2**(rep-1-i), (3, 3),
                      activation='relu', padding='same')(drop)
        conv = Conv2D(start_neurons * 2**(rep-1-i), (3, 3),
                      activation='relu', padding='same')(conv)
        if add_bn:
            conv = BatchNormalization()(conv)

        last_layer = drop

    output_layer = Conv2D(1, (1, 1), padding="same",
                          activation="sigmoid")(last_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model