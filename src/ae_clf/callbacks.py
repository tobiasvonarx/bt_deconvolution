from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow import summary
from util import RMSEnum, RMSEloss
from constants import log_dir, PATIENCE, EPOCHS, LR, TENSORBOARD
from tensorflow.math import exp
from tensorflow.keras.metrics import Mean

class TestLoss(Callback):
    def __init__(self, model, x_test, prop_test):
        super(TestLoss, self).__init__()
        self.model = model
        self.x_test = x_test
        self.prop_test = prop_test
        self.train_writer = summary.create_file_writer(log_dir+"/train/")
        self.train_writer.set_as_default()
        
    def on_epoch_end(self, epoch, logs=None):
        loss = RMSEnum(self.model(self.x_test), self.x_test)
        summary.scalar("epoch_loss_test", data=loss, step=epoch)

def init_callbacks(model, x_test, prop_test):
    tensorboard = TensorBoard(log_dir=log_dir)
    earlystopping = EarlyStopping(monitor="epoch_loss_test", patience=PATIENCE)
    testloss = TestLoss(model, x_test, prop_test)
    # scheduler = lr_polynomial_decay(epochs=EPOCHS, initial_learning_rate=0.01, power=1)
    if TENSORBOARD:
        return [tensorboard, testloss]
    else:
        return []
    return [tensorboard, earlystopping, testloss]
    return [LearningRateScheduler(lr_time_based_decay)]

def lr_time_based_decay(epoch, lr):
        return lr * 1 / (1 + LR/EPOCHS * epoch)
