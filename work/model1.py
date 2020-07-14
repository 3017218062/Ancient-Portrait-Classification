import numpy as np
import pandas as pd
import PIL.Image as Image
import gc, cv2, warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")
iPath = "../input/"
oPath = "../output/"

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import *
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from efficientnet.tfkeras import EfficientNetB4


class dataSet(object):
    def __init__(self, targetSize=(128, 128)):
        def smoothLabel(labels, factor=0.1):
            return (1 - factor) * labels + (factor / labels.shape[1])

        data = pd.read_csv(iPath + "train.csv")
        filenames = data["image"]
        images = []
        for i in tqdm(filenames):
            image = np.array(Image.open(iPath + "train/" + i).convert("RGB"), dtype=np.float)
            image = cv2.resize(image.astype(np.uint8), targetSize)
            images.append(image)
        images = np.asarray(images, dtype=np.uint8)

        labels = data["gender_status"].values
        labels[labels >= 5] -= 1

        self.xTrain, self.yTrain = images.copy(), pd.get_dummies(pd.Series(labels).astype(np.uint8)).values
        del data, filenames, images, labels
        gc.collect()

        oldLabels = np.argmax(self.yTrain, axis=-1)
        newImages, newLabels = [], []
        indices = np.where(oldLabels == 5)[0]
        for i in range(6):
            newImages.append(self.xTrain[indices].copy())
            newLabels.append(self.yTrain[indices].copy())
        indices = np.where(oldLabels == 6)[0]
        for i in range(3):
            newImages.append(self.xTrain[indices].copy())
            newLabels.append(self.yTrain[indices].copy())
        newImages = np.concatenate(newImages, axis=0)
        newLabels = np.concatenate(newLabels, axis=0)

        self.xTrain = np.concatenate([self.xTrain, newImages], axis=0)
        self.yTrain = np.concatenate([self.yTrain, newLabels], axis=0)
        self.yTrain = smoothLabel(self.yTrain, 0.1)
        del newImages, newLabels
        gc.collect()

        print("Image(train) shape:", self.xTrain.shape)
        print("Label(train) shape:", self.yTrain.shape)

    def getTrain(self):
        return self.xTrain, self.yTrain


class imageGenerator(ImageDataGenerator):
    def flow(self, x, y, batch_size=32, shuffle=True, seed=2020):
        def mixUp(X, Y, alpha):
            b = X.shape[0]
            indices = np.random.permutation(b)
            lam = np.random.beta(alpha, alpha)
            X = X * lam + X[indices] * (1. - lam)
            Y = Y * lam + Y[indices] * (1. - lam)
            return X, Y

        for _x, _y in super().flow(x, y, batch_size=batch_size, shuffle=shuffle, seed=2020):
            if shuffle: _x, _y = mixUp(_x, _y, 1.)
            yield _x, _y


class SWA(Callback):
    def __init__(self, swa_epoch):
        super(SWA, self).__init__()
        self.swa_epoch = swa_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.swa_epoch:
            self.swa_weights = self.model.get_weights()
        elif epoch > self.swa_epoch:
            for i, layer in enumerate(self.model.layers):
                self.swa_weights[i] = (self.swa_weights[i] * (epoch - self.swa_epoch) + self.model.get_weights()[i]) / (
                        (epoch - self.swa_epoch) + 1.)

    def on_train_end(self, logs=None):
        self.model.set_weights(self.swa_weights)
        self.model.save_weights(oPath + "model1_swa.h5")


class lrScheduler(Callback):
    def __init__(self, lrMax=1e-3, lrMin=1e-10, lrBase=1e-5, warmUpEpochs=5, epochs=50):
        super(lrScheduler, self).__init__()
        self.lrMax, self.lrMin, self.lrBase = lrMax, lrMin, lrBase
        self.warmUpEpochs, self.epochs = warmUpEpochs, epochs

    def on_train_begin(self, logs=None):
        self.lr = self.lrBase
        K.set_value(self.model.optimizer.lr, self.lr)

    def on_epoch_begin(self, epoch, logs=None):
        def cosineAnnealing(index):
            value = np.cos(index * np.pi / (self.epochs - self.warmUpEpochs)) + 1
            return value * (self.lrMax - self.lrMin) / 2 + self.lrMin

        if epoch <= self.warmUpEpochs:
            self.lr = (self.lrMax - self.lrBase) * epoch / self.warmUpEpochs + self.lrBase
        else:
            self.lr = cosineAnnealing(epoch - self.warmUpEpochs)
        K.set_value(self.model.optimizer.lr, self.lr)
        print("\nEpoch %02d: LrScheduler reducing lr to %s." % (epoch + 1, self.lr))


class Classifier(object):
    def __init__(self, dataset, inputShape=(128, 128, 3), epochs=100, batchs=32):
        self.xTrain, self.yTrain = dataset.getTrain()
        self.inputShape, self.epochs, self.batchs = inputShape, epochs, batchs

    def modelBuilding(self):
        def focalLoss(alpha=1., gamma=2.):
            alpha = float(alpha)
            gamma = float(gamma)

            def multiCategoryFocalLossFixed(yTrue, yPred):
                yTrue = tf.cast(yTrue, tf.float32)
                yPred = tf.cast(yPred, tf.float32)
                yPred = K.clip(yPred, K.epsilon(), 1. - K.epsilon())
                ce = tf.multiply(yTrue, -K.log(yPred))
                weight = tf.multiply(yTrue, tf.pow(tf.subtract(1., yPred), gamma))
                fl = tf.multiply(alpha, tf.multiply(weight, ce))
                reducedF1 = tf.reduce_max(fl, axis=1)
                return tf.reduce_sum(reducedF1)

            return multiCategoryFocalLossFixed

        self.pretrainedNet = EfficientNetB4(weights="imagenet", include_top=False)
        for layer in self.pretrainedNet.layers: layer.trainable = True

        i = Input(shape=self.inputShape)
        x = self.pretrainedNet(i)
        x = GlobalAveragePooling2D()(x)
        o = Dense(7, activation=softmax, use_bias=True,
                  kernel_initializer=glorot_uniform(seed=2020),
                  bias_initializer=Zeros())(x)

        self.clf = Model(i, o)
        self.clf.compile(
            optimizer=Adam(lr=1e-3),
            loss=focalLoss(alpha=1., gamma=2.),
            metrics=["accuracy"]
        )
        self.clf.summary()

    def modelFitting(self):
        trainGen = imageGenerator(
            rescale=1. / 255, rotation_range=10,
            width_shift_range=0.1, height_shift_range=0.1,
            shear_range=0.1, zoom_range=0.1,
            horizontal_flip=True, vertical_flip=False,
            fill_mode="nearest",
        )
        train = trainGen.flow(self.xTrain, self.yTrain, batch_size=self.batchs, seed=2020)

        self.clf.fit_generator(
            train, steps_per_epoch=self.xTrain.shape[0] // self.batchs,
            epochs=self.epochs, verbose=1,
            callbacks=[
                lrScheduler(lrMax=1e-3, lrBase=1e-5, warmUpEpochs=5, epochs=self.epochs),
                CSVLogger(oPath + "history1.csv", separator=',', append=False),
                SWA(self.epochs - 5),
            ],
        )


if __name__ == "__main__":
    dataset = dataSet(targetSize=(256, 256))
    classifier = Classifier(
        dataset,
        inputShape=(256, 256, 3), epochs=65, batchs=24
    )
    classifier.modelBuilding()
    classifier.modelFitting()
