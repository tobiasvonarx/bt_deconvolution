import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.nn import softmax, relu
# from tensorflow.keras.regularizers import L1L2
from constants import *
# from util import RMSEloss

class Encoder(tf.keras.Model):
    def __init__(self, N, hp=None):
        super(Encoder, self).__init__()
        if len(sys.argv)>1 and sys.argv[1]=="gbm_neftel":
            activation = softmax
        else:
            activation = softmax
        # self.bottleneck = Dense(K, input_shape=(N,), activation=activation, kernel_regularizer="l1",bias_regularizer="l1_l2", name="bottleneck")
        self.dropout = Dropout(hp.suggest_float("dropout1",0,0.8) if STUDY else DROPOUT_RATE_ENCODER)
        self.bottleneck = Dense(K, input_shape=(N,), activation=activation, name="bottleneck")

    @tf.function
    def call(self, x, training=False):
        X = self.dropout(x)
        X = self.bottleneck(X)
        return X

class Decoder(tf.keras.Model):
    def __init__(self, N, hp=None):
        super(Decoder, self).__init__()
        self.dropout = Dropout(hp.suggest_float("dropout2",0,0.5) if STUDY else DROPOUT_RATE_DECODER)
        # self.decoded = Dense(N, kernel_regularizer="l1",bias_regularizer="l1_l2",activation=relu)
        self.decoded = Dense(N, activation=relu)
    
    @tf.function
    def call(self, x, training=False):
        X = self.dropout(x)
        X = self.decoded(X)
        return X
    
class AE(tf.keras.Model):
    def __init__(self, N, hp=None):
        super(AE, self).__init__()
        self.encoder = Encoder(N, hp)
        self.decoder = Decoder(N, hp)
    @tf.function
    def call(self, x):
        latent = self.encoder(x)
        out = self.decoder(latent)
        return out
    @tf.function
    def train_step(self, x):
        # x[0] is X, x[1] is Y. they are identical for the AE
        x = x[0]
        with tf.GradientTape() as tape:#, tf.GradientTape() as dec_tape:
            latent = self.encoder(x, training=True)
            pred = self.decoder(latent, training=True)
            loss = self.compiled_loss(x, pred)
            # print(x,pred)
            # rmse = RMSEloss(x, pred)
        # self.optimizer.minimize(loss, var_list=self.encoder.trainable_variables, tape=enc_tape)
        # self.optimizer.minimize(loss, var_list=self.decoder.trainable_variables, tape=dec_tape)
        self.optimizer.minimize(loss, var_list=self.trainable_variables, tape=tape)

        return self.compute_metrics(x, x, pred, None)
