import tensorflow as tf


class Network(object):
    def __init__(self, prefix, n_layer):
        self.layer_list = []
        self.prefix = prefix
        self.n_layer = n_layer

    def make_layer_list(self, n_input, n_output, n_unit):
        pass

    def build(self):
        return tf.keras.Sequential(self.layer_list)


class FCN(Network):
    def __init__(self, prefix, n_layer, n_input, n_output, n_unit):
        super(FCN, self).__init__(prefix, n_layer)

        self.layer_list = self.make_layer_list(n_input, n_output, n_unit)

    def make_layer_list(self, n_input, n_output, n_unit):
        layers = []
        layers.append(tf.keras.layers.InputLayer((n_input,)))

        for i in range(self.n_layer):
            layer_name = self.prefix + '/dense' + str(i + 1)
            layers.append(tf.keras.layers.Dense(n_unit, activation='relu', name=layer_name))

        layers.append(tf.keras.layers.Dense(n_output, name=self.prefix + '/dense' + str(self.n_layer + 1)))

        return layers


class Main(FCN):
    def __init__(self, d_in):
        super(Main, self).__init__("main", 2, d_in, 10, 50)


class HM(FCN):
    def __init__(self):
        super(HM, self).__init__("meta", 2, 2, 1, 30)


class SeparateMain(FCN):
    def __init__(self, d_in):
        super(SeparateMain, self).__init__("main", 2, d_in, 20, 50)


class MultiFCN(FCN):
    def __init__(self, prefix, n_layer, n_input, n_output, n_unit, n_layer_main):
        super(MultiFCN, self).__init__(prefix, n_layer, n_input, n_output, n_unit)

        self.net_list = []
        for j in range(n_layer_main):
            prefix = str(j) + "_" + prefix
            net = self.make_layer_list(n_input, n_output, n_unit)
            self.net_list.append(net)


class BaseCNN(Network):
    def __init__(self, prefix, n_layer, n_input, n_output, n_filter, n_channel):
        super(BaseCNN, self).__init__(prefix, n_layer)
        self.n_filter = n_filter
        self.n_channel = n_channel
        self.layer_list = self.make_layer_list(n_layer, n_input, n_output)

    def make_layer_list(self, n_layer, n_input, n_output):
        layers = []
        layers.append(tf.keras.layers.Conv2D(self.n_filter, kernel_size=(3, 3), activation='relu',
                                             input_shape=(n_input, n_input, self.n_channel)))

        for i in range(n_layer):
            layers.append(tf.keras.layers.MaxPooling2D((2, 2)))
            layers.append(tf.keras.layers.Conv2D(self.n_filter, kernel_size=(3, 3), activation='relu'))

        layers.append(tf.keras.layers.Flatten())
        layers.append(tf.keras.layers.Dense(256))
        layers.append(tf.keras.layers.Dense(n_output))

        return layers

    def build(self):
        return tf.keras.Sequential(self.layer_list)


class MainCNN(BaseCNN):
    def __init__(self):
        super(MainCNN, self).__init__("main", 1, 32, 10, 32, 3)


class MobileNet(object):
    def build(self):
        return tf.keras.applications.MobileNetV3Small(input_shape=(32, 32, 3), classes=10, weights=None)
