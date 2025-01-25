import tensorflow as tf
from keras import Model, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout


def create_modified_vgg19():
    # load inceptionv3 model base
    model_base = tf.keras.applications.vgg19.VGG19(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

    # freeze trained layers
    for layer in model_base.layers:
        layer.trainable = False

    # base up until dense layers
    z = model_base.output
    z = Flatten()(z)

    # add own, untrained layers
    z = Dense(units=75, activation='relu')(z)

    # ideas to try out:
    z = BatchNormalization()(z)

    # add output layer
    predictions = tf.keras.layers.Dense(units=10, activation='softmax')(z)
    model = Model(inputs=model_base.input, outputs=predictions)

    # compile with optimizer and loss
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def create_custom_model():
    i = tf.keras.Input(shape=(224, 224, 3))

    units = 64
    kernel_size = (3, 3)

    # Block 1
    # x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(i)
    x = Conv2D(units, kernel_size, activation='relu', padding='same', name='block1_conv1')(i)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool1')(x)
    x = Conv2D(units, kernel_size, activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool2')(x)

    # Block 2
    # x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(units*2, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool1')(x)
    x = Conv2D(units*2, kernel_size, activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool2')(x)

    # Block 3
    # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(units*4, kernel_size, activation='relu', padding='same', name='block3_conv1')(x)
    # x = Conv2D(units*4, kernel_size, activation='relu', padding='same', name='block3_conv2')(x)
    # x = Conv2D(units*4, kernel_size, activation='relu', padding='same', name='block3_conv3')(x)
    # x = Conv2D(units*4, kernel_size, activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool1')(x)

    # # Block 4
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Top
    x = Flatten(name='flatten')(x)
    x = Dense(30, activation='relu', name='dense_1')(x)
    x = BatchNormalization()(x)

    predictions = Dense(units=10, activation='softmax')(x)

    model = Model(inputs=i, outputs=predictions)

    # compile with optimizer and loss
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def recreate_vgg19_with_our_top():
    model_input = tf.keras.Input(shape=(256, 256, 3))

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(model_input)
    # x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # # Block 4
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    #
    # # Block 5
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Top
    x = Flatten(name='flatten')(x)
    x = Dense(50, activation='relu', name='dense50')(x)
    x = BatchNormalization()(x)

    predictions = Dense(units=10, activation='softmax')(x)

    model = Model(inputs=model_input, outputs=predictions)

    # compile with optimizer and loss
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def new_model_from_pretrained(pretrained_model_name):
    # Input
    i = Input(shape=(256, 256, 3))

    # block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(i)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # block 3
    # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # our block 3 addition
    # x = Dropout(rate=0.1)(x)

    # block 4
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # block 5
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # block 6
    x = Flatten(name='block5_flatten')(x)
    x = Dense(20, activation='relu', name='block5_dense50_1')(x)
    x = BatchNormalization(name='block5_batchnorm1')(x)

    # output
    o = Dense(units=10, activation='softmax', name='output_dense10')(x)

    # create model instance
    model = Model(inputs=i, outputs=o)

    # compile with optimizer and loss
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.load_weights(filepath='trained_models/' + str(pretrained_model_name),
                       by_name=True)

    # for layer in model.layers[:-4]:
    #     layer.trainable = False

    return model
