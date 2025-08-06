def efficientnetv2_retinanet(optimizer="adam", classification_loss="focal", box_loss="smoothl1"):
    import keras_cv

    backbone = keras_cv.models.EfficientNetV2Backbone.from_preset("efficientnetv2_b0_imagenet", include_rescaling=True)
    backbone.trainable = False

    model = keras_cv.models.RetinaNet(
        num_classes=1,
        bounding_box_format="xywh",
        backbone=backbone,
    )

    model.compile(
        optimizer=optimizer,
        classification_loss=classification_loss,
        box_loss=box_loss
    )

    return model


def init():
    import keras
    import tensorflow as tf

    #keras.mixed_precision.set_global_policy('mixed_float16')

    try:
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for GPU(s)")
    except RuntimeError as e:
        print("Error setting memory growth:", e)