import tensorflow as tf


class FCN(tf.keras.Model):
    def __init__(self, prefix, n_layer, n_input, n_output, n_unit):
        super(FCN, self).__init__()

        self.layer_list = self.make_layer_list(prefix, n_layer, n_input, n_output, n_unit)

    def call(self, inputs):
        for layer in self.layer_list:
            temp = layer(inputs)
            inputs = temp

        return temp

    @staticmethod
    def make_layer_list(prefix, n_layer, n_input, n_output, n_unit):
        layers = []
        layers.append(tf.keras.layers.InputLayer((n_input,)))
        for i in range(n_layer):
            layer_name = prefix + '/dense' + str(i + 1)
            layers.append(tf.keras.layers.Dense(n_unit, activation='relu', name=layer_name))

        layers.append(tf.keras.layers.Dense(n_output, name=prefix + '/dense' + str(n_layer + 1)))

        return layers


class MultiFCN(FCN):
    def __init__(self, prefix, n_layer, n_input, n_output, n_unit, n_layer_main):
        super(MultiFCN, self).__init__(prefix, n_layer, n_input, n_output, n_unit)

        self.net_list = []
        for j in range(n_layer_main):
            prefix = str(j) + "_" + prefix
            net = self.make_layer_list(prefix, n_layer, n_input, n_output, n_unit)
            self.net_list.append(net)

    def call(self, inputs):
        return self.net(inputs)
