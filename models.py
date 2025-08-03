def efficientnetv2_retinanet():
    import keras_cv

    backbone = keras_cv.models.EfficientNetV2Backbone.from_preset("efficientnetv2_b0_imagenet", include_rescaling=True)
    backbone.trainable = False

    model = keras_cv.models.RetinaNet(
        num_classes=1,
        bounding_box_format="xywh",
        backbone=backbone,
    )

    model.compile(
        optimizer="adam",
        classification_loss="focal",
        box_loss="smoothl1"
    )

    return model

