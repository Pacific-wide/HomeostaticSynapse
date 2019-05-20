

class FCN(tf.keras.models.Model):
    def __init__(self):
        super(FCN, self).__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.InputLayer((784,)),
            tf.keras.layers.Dense(100, activation='relu', name='dense1'),
            tf.keras.layers.Dense(100, activation='relu', name='dense2'),
            tf.keras.layers.Dense(10, name='dense3')])

    def call(self, inputs):
        return self.net(inputs)

class MetaFCN(tf.keras.models.Model):
    def __init__(self):
        super(MetaFCN, self).__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.InputLayer((1,)),
            tf.keras.layers.Dense(30, activation='relu', name='dense1'),
            tf.keras.layers.Dense(30, activation='relu', name='dense2'),
            tf.keras.layers.Dense(30, activation='relu', name='dense3'),
            tf.keras.layers.Dense(30, activation='relu', name='dense4'),
            tf.keras.layers.Dense(1, name='dense5')])

    def call(self, inputs):
        return self.net(layer_to_flat(inputs))
