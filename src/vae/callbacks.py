from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, LearningRateScheduler
from tensorflow import summary
from preprocess import get_true_proportions
from util import RMSEnum
from constants import log_dir, PATIENCE, TENSORBOARD
import pandas as pd

class TestLoss(Callback):
    def __init__(self, model, x_test):
        super(TestLoss, self).__init__()
        self.true_proportions = get_true_proportions()
        self.model = model
        self.x_test = x_test
        self.y_test = pd.concat(x_test, axis=1)
        self.train_writer = summary.create_file_writer(log_dir+"/train/")
        self.train_writer.set_as_default()

        
    def on_epoch_end(self, epoch, logs=None):
        loss = RMSEnum(self.model.predict(self.x_test), self.y_test)
        summary.scalar("epoch_loss_test", data=loss, step=epoch)

def init_callbacks(model, x_test):
    tensorboard = TensorBoard(log_dir=log_dir)
    earlystopping = EarlyStopping(monitor="val_loss", patience=PATIENCE)
    testloss = TestLoss(model, x_test)
    if TENSORBOARD:
        return [tensorboard, testloss]
    else:
        return []
    return [tensorboard, earlystopping, testloss]