from datasets import *
from datasets import composite_ds
from debugging import *
from analytics import *
from augmentations import *
from models import *
import keras_cv
import keras
import tensorflow as tf

init()

model = efficientnetv2_retinanet()

ds = composite_ds(
    multi_coin_ds([('./datasets/synthetic/annotations.xml', None)], d_img=False, d_bb=False, tsz=(512, 512)),
    multi_coin_ds([('./datasets/synthetic/annotations.xml', None)], d_img=False, d_bb=False, tsz=(512, 512)).map(apply_directional_blur(45, 11)),
    multi_coin_ds([('./datasets/synthetic/annotations.xml', None)], d_img=False, d_bb=False, tsz=(512, 512)).map(apply_gaussian_blur(kernel_size=15))
).batch(32).prefetch(tf.data.AUTOTUNE)


model.fit(
    ds,
    epochs=100,
    validation_data=multi_coin_ds([
        ('./datasets/090/annotations.xml', 'PNG'),
        ('./datasets/190/annotations.xml', 'PNG'),
    ], d_img=False, d_bb=False, tsz=(512, 512)).batch(16).prefetch(tf.data.AUTOTUNE),
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.1, verbose=1)
    ],
)
