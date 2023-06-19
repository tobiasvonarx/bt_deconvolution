import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Layer, Concatenate, BatchNormalization, LeakyReLU, ReLU
from tensorflow.nn import softmax, relu, leaky_relu
from tensorflow.keras.activations import linear
# from tensorflow.keras.regularizers import L1L2
from constants import *
from util import KLloss, MSEloss, RMSEloss
from tensorflow.keras.metrics import Mean
import tensorflow.keras.backend as KB

# source: keras docs
class Sampling(Layer):
    def call(self, x):
        z_mean, z_logvar = x
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_logvar) * epsilon

# identity transform that adds the kl divergence loss
class KLDivergence(Layer):
    def __init__(self, i):
        super(KLDivergence, self).__init__()
        self.kl_tracker = Mean(name=f"kl_loss_{i}")

    @property
    def metrics(self):
        return [
            self.kl_tracker
        ]

    def call(self, x):
        mean, logvar = x
        kl_loss = KLloss(mean, logvar)
        self.kl_tracker.update_state(kl_loss)
        self.add_loss(KB.mean(kl_loss), inputs=x)
        return x

class Module(tf.keras.Model):
    def __init__(self,i,N, t=None):
        super(Module, self).__init__()
        # N_COL in
        # hidden layer
        # variational part with latent_dim = K
        # sampling from that
        self.hidden = Dense(M1, input_shape=(N,), name="m1")
        self.batchnorm = BatchNormalization()
        self.activation = ReLU()
        self.dropout = Dropout(t.suggest_float("dropout1",0,0.9) if STUDY else DROPOUT_RATE_ENCODER)
        self.z_mean = Dense(K, name="z_mean")
        self.z_logvar = Dense(K, name="z_logvar")
        self.kl_divergence = KLDivergence(i)
        self.z = Sampling()

    def call(self, x, training=False):
        # return K sampled values
        X = self.hidden(x)
        X = self.batchnorm(X)
        X = self.activation(X)
        if training:
            X = self.dropout(X)
        z_mean = self.z_mean(X)
        z_logvar = self.z_logvar(X)
        if training:
            z_mean, z_logvar = self.kl_divergence([z_mean, z_logvar])
        return self.z([z_mean, z_logvar])
    
class Decoder(tf.keras.Model):
    def __init__(self, N, t=None):
        super(Decoder, self).__init__()
        # K*K in
        # hidden layer
        # K*N_COL reconstructed all marker genes output
        self.hidden = Dense(M2, input_shape=(K*K,), name="m2")
        self.batchnorm = BatchNormalization()
        self.activation = ReLU()
        self.dropout = Dropout(t.suggest_float("dropout2",0,0.5) if STUDY else DROPOUT_RATE_DECODER)
        self.reconstructed = Dense(K*N, activation=linear, name="out")
    
    def call(self, x, training=False):
        X = self.hidden(x)
        X = self.batchnorm(X)
        X = self.activation(X)
        if training:
            X = self.dropout(X)
        return self.reconstructed(X)

class VAE(tf.keras.Model):
    def __init__(self, N, t=None):
        super(VAE, self).__init__()
        # create K modular encoders
        self.encoders = [Module(i,N,t) for i in range(K)]
        self.concat = Concatenate()
        # one shared decoder
        self.decoder = Decoder(N,t)
        #self.total_loss_tracker = Mean(name="total_loss")
        #self.reconstruction_loss_tracker = Mean(name="reconstruction_loss")
        #self.kl_loss_tracker = Mean(name="kl_loss")
    
    def call(self, x, training=False):
        # x is a K-tuple
        z = self.concat([encoder(x[i]) for i,encoder in enumerate(self.encoders)])
        out = self.decoder(z)
        return out
    """
    @tf.function
    def train_step(self, x):
        # x[0] is X, x[1] is Y. they are identical for the AE
        x = x[0]
        all_x = self.concat(x)
        print("x",x)
        print("all_x",x)
        assert False

        with tf.GradientTape() as tape:#, tf.GradientTape() as dec_tape:
            zs = []
            kl_losses = 0
            for i in range(K):
                z, z_mean, z_logvar = self.encoders[i](x[i], training=True)
                kl_loss = KLloss(z_mean, z_logvar)
                zs.append(z)
                kl_losses += kl_loss
            reconstructed = self.decoder(self.concat(zs), training=True)
            reconstruction_loss = MSEloss(all_x, reconstructed)
            loss = kl_losses + reconstruction_loss
        self.optimizer.minimize(loss, var_list=self.trainable_weights, tape=tape)
        self.total_loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_losses)

        # return self.compute_metrics(all_x, all_x, reconstructed, None)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
        """