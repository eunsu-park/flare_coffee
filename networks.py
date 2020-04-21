import tensorflow as tf
import tensorflow.keras as keras

conv_init=tf.random_normal_initializer(0., 0.02)
gamma_init=tf.random_normal_initializer(1., 0.02)

def down_conv(nb_feature, *a, **k):
    return keras.layers.Conv2D(filters=nb_feature, *a, **k, kernel_initializer=conv_init, use_bias=False)

def batch_norm():
    return keras.layers.BatchNormalization(momentum=0.9, axis=-1, epsilon=1.01e-5, gamma_initializer=gamma_init)

def dropout(*a, **k):
    return keras.layers.Dropout(*a, **k)

def activation(*a, **k):
    return keras.layers.Activation(*a, **k)

def concatenate(*a, **k):
    return keras.layers.Concatenate(*a, **k)

def average_pooling(*a, **k):
    return keras.layers.AveragePooling2D(*a, **k)

def max_pooling(*a, **k):
    return keras.layers.MaxPooling2D(*a, **k)

def dense(*a, **k):
    return keras.layers.Dense(*a, **k)

def flatten(*a, **k):
    return keras.layers.Flatten(*a, **k)


class block:

    def __init__(self, nb_feature, growth_rate):
        self.nb_feature = nb_feature
        self.growth_rate = growth_rate

    def __call__(self, layer):
        nb_inception = self.nb_feature + self.growth_rate

        layer1 = down_conv(nb_inception, kernel_size=(1, 1), strides=1, padding='same') (layer)
        layer1 = batch_norm() (layer1)
        layer1 = activation('relu') (layer1)

        layer2 = down_conv(nb_inception, kernel_size=(1, 3), strides=1, padding='same') (layer)
        layer2 = batch_norm() (layer2)
        layer2 = activation('relu') (layer2)

        layer3 = down_conv(nb_inception, kernel_size=(3, 1), strides=1, padding='same') (layer)
        layer3 = batch_norm() (layer3)
        layer3 = activation('relu') (layer3)

        layer4 = down_conv(nb_inception, kernel_size=(3, 3), strides=1, padding='same') (layer)
        layer4 = batch_norm() (layer4)
        layer4 = activation('relu') (layer4)

        layer = concatenate(-1) ([layer1, layer2, layer3, layer4])
        layer = down_conv(nb_inception*4, kernel_size=(1, 1), strides=1, padding='same') (layer)
        layer = activation('relu') (layer)
        layer = down_conv(nb_inception, kernel_size=(3, 3), strides=1, padding='same') (layer)
#        layer = activation('relu') (layer)        
    
        return layer, nb_inception

class module:

    def __init__(self, nb_feature, nb_block, growth_rate) :
        self.nb_feature = nb_feature
        self.nb_block = nb_block
        self.growth_rate = growth_rate

    def __call__(self, layer_input):
        layer = layer_input
        for n in range(self.nb_block):
            layer, self.nb_feature = block(self.nb_feature, self.growth_rate) (layer)
            layer = concatenate(-1) ([layer_input, layer])
        return layer, self.nb_feature
        

def coffee(isize, ch_input, ch_output, growth_rate=16, nb_module=2, nb_block=5):

    input_A = keras.layers.Input(shape=(isize, isize, ch_input), dtype=tf.float32)

    nb_feature = 64
    layer = down_conv(nb_feature, kernel_size=(3, 3), strides=1, padding='same') (input_A)
    layer = activation('relu') (layer)

    for n in range(nb_module):
        nb_feature += growth_rate
        layer, nb_feature = module(nb_feature, nb_block, growth_rate) (layer)
        layer = down_conv(nb_feature, kernel_size=(1, 1), strides=1, padding='same') (layer)
        layer = batch_norm() (layer)
        layer = activation('relu') (layer)
        layer = max_pooling(pool_size=(3, 3), strides=2, padding='same') (layer)

    shape = layer.get_shape().as_list()
    layer = average_pooling(pool_size=(shape[1], shape[2]), strides=(shape[1], shape[2]), padding='same') (layer)
    layer = flatten() (layer)

    layer = dense(ch_output) (layer)
    layer = activation('softmax') (layer)

    return keras.models.Model(inputs=input_A, outputs=layer)


if __name__ == '__main__' :
    model = coffee(256, 1, 2)
    model.summary()
