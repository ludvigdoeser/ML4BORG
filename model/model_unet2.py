#this one bottleneck down to 4^3
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Model, Input
from tensorflow.keras.layers import*
import tensorflow.keras as keras

def U_net_3d_2(width, height, depth,lr=0.001,input_ch=1):
    tf.keras.backend.clear_session()

    input_ch = 3
    in1 = keras.Input((width, height, depth, input_ch))

    conv1 = Conv3D(filters=64, kernel_size=2, activation='relu', kernel_initializer='he_normal', padding='same')(in1)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv3D(filters=64, kernel_size=2, activation='relu', kernel_initializer='he_normal', padding='same')(conv1)
    pool1 = MaxPooling3D((2, 2, 2))(conv1)

    conv2 = Conv3D(filters=64, kernel_size=2, activation='relu', kernel_initializer='he_normal', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv3D(filters=64, kernel_size=2, activation='relu', kernel_initializer='he_normal', padding='same')(conv2)
    pool2 = MaxPooling3D((2, 2, 2))(conv2)

    conv3 = Conv3D(filters=64, kernel_size=2, activation='relu', kernel_initializer='he_normal', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv3D(filters=64, kernel_size=2, activation='relu', kernel_initializer='he_normal', padding='same')(conv3)
    pool3 = MaxPooling3D((2, 2, 2))(conv3)

    conv4 = Conv3D(filters=64, kernel_size=2, activation='relu', kernel_initializer='he_normal', padding='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv3D(filters=64, kernel_size=2, activation='relu', kernel_initializer='he_normal', padding='same')(conv4)

    ########################################################
    up1 = concatenate([UpSampling3D((2, 2, 2))(conv4), conv3], axis=-1)
    conv5 = Conv3D(filters=64, kernel_size=2, activation='relu', kernel_initializer='he_normal', padding='same')(up1)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv3D(filters=64, kernel_size=2, activation='relu', kernel_initializer='he_normal', padding='same')(conv5)

    up2 = concatenate([UpSampling3D((2, 2, 2))(conv5), conv2], axis=-1)
    conv6 = Conv3D(filters=64, kernel_size=2, activation='relu', kernel_initializer='he_normal', padding='same')(up2)
    #conv6 = Dropout(0.2)(conv6)
    conv6 = Conv3D(filters=64, kernel_size=2, activation='relu', kernel_initializer='he_normal', padding='same')(conv6)

    up3 = concatenate([UpSampling3D((2, 2, 2))(conv6), conv1], axis=-1)
    conv7 = Conv3D(filters=64, kernel_size=(2, 2, 2), activation='relu', kernel_initializer='he_normal', padding='same')(up3)
    #conv7 = Dropout(0.2)(conv7)
    conv7 = Conv3D(filters=16, kernel_size=(2, 2, 2), activation='relu', kernel_initializer='he_normal', padding='same')(conv7)
    conv7 = Conv3D(filters=3, kernel_size=(2,2,2), kernel_initializer='he_normal', padding='same')(conv7)
    ########################################################
    
    model = Model(inputs=[in1], outputs=[conv7],name='U-net3D')
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=tf.keras.losses.MeanSquaredError(),metrics=[tf.keras.metrics.MeanSquaredError()])
    return model