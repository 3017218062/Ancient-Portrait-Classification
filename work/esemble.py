import numpy as np
import pandas as pd
import PIL.Image as Image
import gc, cv2, os, warnings
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.initializers import *
from efficientnet.tfkeras import EfficientNetB4, EfficientNetB7
from classification_models.tfkeras import Classifiers

warnings.filterwarnings("ignore")
iPath = "../input/"
oPath = "../output/"
gc.collect()


class GeM(Layer):
    def __init__(self):
        super(GeM, self).__init__()
        self.pool = GlobalAveragePooling2D()
        self.e = 3.0

    def build(self, inputShape):
        self.gemExp = self.add_weight("p", (inputShape[-1],), initializer=keras.initializers.constant(self.e),
                                      trainable=True)
        super(GeM, self).build(inputShape)

    def call(self, i):
        x = tf.maximum(i, 1e-9)
        x = tf.pow(x, self.gemExp)
        x = self.pool(x)
        o = tf.pow(x, 1. / self.gemExp)
        return o


def Model1(inputShape):
    pretrainedNet = EfficientNetB4(weights=None, include_top=False)
    i = Input(shape=inputShape)
    x = pretrainedNet(i)
    x = GlobalAveragePooling2D()(x)
    o = Dense(7, activation=softmax)(x)
    clf = Model(i, o)
    clf.summary()
    return clf


def Model2(inputShape):
    pretrainedNet = Classifiers.get("seresnext50")[0](input_shape=inputShape, weights=None, include_top=False)
    i = Input(shape=inputShape)
    x = pretrainedNet(i)
    x = GeM()(x)
    o = Dense(7, activation=softmax)(x)
    clf = Model(i, o)
    clf.summary()
    return clf


def Model3(inputShape):
    pretrainedNet = EfficientNetB7(weights=None, include_top=False)
    i = Input(shape=inputShape)
    x = pretrainedNet(i)
    x = GeM()(x)
    o = Dense(2, activation=softmax)(x)
    clf = Model(i, o)
    clf.summary()
    return clf


def preprocess(targetSize):
    data = pd.read_csv(iPath + "sample_submission.csv")
    filenames = data["image"]

    images = []
    for i in tqdm(filenames):
        image = np.array(Image.open(iPath + "test/" + i).convert("RGB"), dtype=np.float)
        image = cv2.resize(image.astype(np.uint8), targetSize)
        images.append(image)
    images = np.asarray(images, dtype=np.uint8)
    return data, images


def predict(images, model, hard=True):
    yPred = model.predict(images / 255.)
    if hard:
        yPred = np.argmax(yPred, axis=-1).reshape(-1).astype(np.uint8)
    return yPred


def TTA(images, model, hard=True):
    print("Raw images...")
    yPred0 = predict(images, model, hard=False)

    print("Flip images...")
    flipImgs = []
    for i in images:
        flipImgs.append(cv2.flip(i, 1))
    flipImgs = np.asarray(flipImgs)
    yPred1 = predict(flipImgs, model, hard=False)
    del flipImgs
    gc.collect()

    print("Fuzzy images...")
    fuzzyImgs = []
    for i in images:
        fuzzyImgs.append(cv2.resize(cv2.resize(i, (128, 128)), (256, 256)))
    fuzzyImgs = np.asarray(fuzzyImgs)
    yPred2 = predict(fuzzyImgs, model, hard=False)
    del fuzzyImgs
    gc.collect()

    print("Crop images...")
    l = 10
    cropImgs = []
    for i in images:
        cropImgs.append(cv2.resize(i[l:-l, l:-l, :], (256, 256)))
    cropImgs = np.asarray(cropImgs)
    yPred3 = predict(cropImgs, model, hard=False)
    del cropImgs
    gc.collect()

    print("Sum...")
    yPred = (yPred0 + yPred1 + yPred2 + yPred3) / 4.
    if hard:
        yPred = np.argmax(yPred, axis=-1)
    return yPred


def save(data, yPred, name):
    data["gender_status"] = yPred.copy()
    data.to_csv(oPath + "%s.csv" % name, index=False)


if __name__ == "__main__":
    targetSize = (256, 256)
    data, images = preprocess(targetSize)
    model1 = Model1(inputShape=(*targetSize, 3))
    model1.load_weights(oPath + "model1_swa.h5")
    yPred1 = TTA(images, model1, hard=False)
    model2 = Model2(inputShape=(*targetSize, 3))
    model2.load_weights(oPath + "model2_swa.h5")
    yPred2 = TTA(images, model2, hard=False)

    soft = yPred1 * 0.45 + yPred2 * 0.55
    hard = np.argmax(soft, axis=-1)
    hard[hard >= 5] += 1
    save(data, hard, "result1")

    model3 = Model3(inputShape=(*targetSize, 3))
    model3.load_weights(oPath + "model3_swa.h5")
    indices1, indices2 = np.where(hard == 1)[0], np.where(hard == 3)[0]
    indices = np.hstack([indices1, indices2])
    yPred = predict(images[indices], model3, hard=False)
    scale = 0.65
    for i, index in enumerate(indices):
        soft[index][1] += yPred[i][0] * 0.4 * scale
        soft[index][3] += yPred[i][1] * 0.6 * scale
        hard[index] = 1 if soft[index][1] > soft[index][3] else 3
    save(data, hard, "result2")
