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


class Meta(FCN):
    def __init__(self):
        super(Meta, self).__init__("meta", 2, 2, 1, 50)


class MetaAlpha(FCN):
    def __init__(self):
        super(MetaAlpha, self).__init__("meta", 2, 4, 1, 20)


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
    def __init__(self, prefix, n_layer, n_input, n_output, n_unit):
        super(BaseCNN, self).__init__(prefix, n_layer)

        self.layer_list = self.make_layer_list(n_input, n_output, n_unit)

    def make_layer_list(self, n_layer, n_output, n_unit):
        layers = []
        layers.append(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

        for i in range(n_layer):
            layer_name = self.prefix + '/conv' + str(i + 1)
            layers.append(tf.keras.layers.MaxPooling2D((2, 2), layer_name=layer_name))
            layers.append(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', layer_name=layer_name))

        layers.append(tf.keras.layers.Dense(n_output, name=self.prefix + '/dense' + str(n_layer + 1)))

        return layers

    def build(self):
        return tf.keras.Sequential(self.layer_list)


