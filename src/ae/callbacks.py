from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow import summary
from preprocess import get_true_proportions
from inference import infer_cell_proportions
from util import RMSEnum, RMSEloss
from constants import log_dir, PATIENCE, EPOCHS, LR, TENSORBOARD
from tensorflow.math import exp
from tensorflow.keras.metrics import Mean

class LatentValues(Callback):
    def __init__(self, model, x_train, x_test, prop_train, prop_test):
        super(LatentValues, self).__init__()
        self.true_proportions = get_true_proportions()
        self.model = model
        self.x_train = x_train
        self.x_test = x_test
        self.prop_train = prop_train
        self.prop_test = prop_test
        self.train_writer = summary.create_file_writer(log_dir+"/train/")
        self.train_writer.set_as_default()
        
    def on_epoch_end(self, epoch, logs=None):
        reord_proportions = infer_cell_proportions(self.model, self.x_train, self.prop_train, epoch, "train")
        loss = RMSEnum(reord_proportions, self.prop_train)
        # with self.train_writer.as_default():
        summary.scalar("epoch_latent_loss_train", data=loss, step=epoch)
        reord_proportions = infer_cell_proportions(self.model, self.x_test, self.prop_test, epoch, "test")
        loss = RMSEnum(reord_proportions, self.prop_test)
        # with self.test_writer.as_default():
        summary.scalar("epoch_latent_loss_test", data=loss, step=epoch)
        loss = RMSEnum(self.model(self.x_test), self.x_test)
        summary.scalar("epoch_loss_test", data=loss, step=epoch)

def init_callbacks(model, x_train, x_test, prop_train, prop_test):
    tensorboard = TensorBoard(log_dir=log_dir)
    earlystopping = EarlyStopping(monitor="epoch_loss_test", patience=PATIENCE)
    latentvalues = LatentValues(model, x_train, x_test, prop_train, prop_test)
    # scheduler = lr_polynomial_decay(epochs=EPOCHS, initial_learning_rate=0.01, power=1)
    if TENSORBOARD:
        return [tensorboard, latentvalues]
    else:
        return []
    return [tensorboard, earlystopping, latentvalues]
    return [LearningRateScheduler(lr_time_based_decay)]

def lr_time_based_decay(epoch, lr):
        return lr * 1 / (1 + LR/EPOCHS * epoch)
