from cv2 import rotate
import numpy as np
import os
import skimage.io as io
from skimage.color import rgb2gray
from skimage.transform import resize
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard
from keras import backend as K
#from tqdm import tqdm
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
from model import *
from datetime import datetime
import pdb

print(f'Tensorflow version: {tf.__version__}')

RANDOM_SEED = 42
tf.compat.v1.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(
    graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

TRAIN_RAW_PATH = '../Image/Train/Raw/*.jpg'
TRAIN_MASK_FOLDER = '../Image/Train/Mask'
#TRAIN_CONTRAST_PATH = "Images/Head/head_training_set/hc_contrast/*.png"
TEST_PATH = '../Image/Test/*.jpg'


def processImage(image_path):
    image = io.imread(image_path)
    #w, h, c = image.shape
    #image = rgb2gray(image[(w-IMG_WIDTH)//2:(w+IMG_WIDTH)//2, (h-IMG_HEIGHT)//2:(h+IMG_HEIGHT)//2])
    image = resize(image, (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS),
                   mode='constant', preserve_range=True)
    image = rgb2gray(image)[..., None]
    return image


X_train, Y_train, X_test = [], [], []

print("Resizing training, test images and masks")

for image_path_raw in glob(TRAIN_RAW_PATH):
    image_name, ext = os.path.splitext(os.path.basename(image_path_raw))
    image_path_mask = os.path.join(TRAIN_MASK_FOLDER, image_name + '.png')
    if not os.path.exists(image_path_mask):
        continue
    X_train.append(processImage(image_path_raw))
    Y_train.append(processImage(image_path_mask))

X_train, Y_train = np.array(X_train), np.array(Y_train)
X_test = np.array([processImage(image_path) for image_path in glob(TEST_PATH)])

print("Done with test and train image resizing")

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)


X_train /= 255
X_test /= 255
#Y_train /= 255
Y_train = (Y_train-np.min(Y_train))/(np.max(Y_train)-np.min(Y_train))

indices = np.arange(len(X_test))
np.random.shuffle(indices)
indices = indices[:5]
fig, ax = plt.subplots(len(indices), 2, figsize=(5, 10))
for i, index in enumerate(indices):
    ax[i][0].imshow(X_train[index], cmap=plt.cm.gray)
    ax[i][1].imshow(Y_train[index], cmap=plt.cm.gray)
    ax[i][0].set_ylabel(f'Training Index {index}')
fig.tight_layout()
plt.show()

io.imshow(X_test[index])
plt.title(f'Test Data, index {index}')
plt.show()


def dice_coefficient(y_true, y_pred):
    smooth = K.epsilon()

    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f, axis=-1)
    sums = K.sum(y_true_f, axis=-1) + K.sum(y_pred_f, axis=-1)

    return (2. * intersection + smooth) / (sums + smooth)


def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3])+K.sum(y_pred, [1, 2, 3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def dice_coefficient_loss(y_true, y_pred):
    return 1-dice_coefficient(y_true, y_pred)


model_name = 'model_for_hc_150_epochs'

LOAD_MODEL = True

if not LOAD_MODEL:

    model = createModelUNET((IMG_WIDTH, IMG_HEIGHT, 1))  # add_bn=True)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss=dice_coefficient_loss,
                  metrics=['accuracy', dice_coefficient, iou_coef])
    model.summary()

    log_dir = f'results/{model_name}/logs/fit/{model_name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    checkpoint = ModelCheckpoint(f'results/{model_name}/best_model', save_weights_only=False,
                                 monitor='val_dice_coefficient', mode='max', save_best_only=True)
    callbacks = [TensorBoard(log_dir=f"results/{model_name}/logs"), checkpoint]
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=0.2, random_state=RANDOM_SEED)
    history = model.fit(x=X_train, y=Y_train, validation_data=(
        X_val, Y_val), batch_size=5, epochs=150, callbacks=callbacks, verbose=1)

else:
    model = tf.keras.models.load_model(f'results/{model_name}/best_model', custom_objects={
                                       'accuracy': 'accuracy', 'dice_coefficient': dice_coefficient, 'dice_coefficient_loss': dice_coefficient_loss})


print('Predicting test data...')
Y_pred = model.predict(X_test)

indices = np.arange(len(X_test))
np.random.shuffle(indices)
indices = indices[:5]
fig, ax = plt.subplots(len(indices), 2, figsize=(5, 10))
for i, index in enumerate(indices):
    ax[i][0].imshow(X_test[index], cmap=plt.cm.gray)
    ax[i][1].imshow(Y_pred[index], cmap=plt.cm.gray)
    ax[i][0].set_ylabel(f'Prediction Index {index}')
fig.tight_layout()
plt.show()
