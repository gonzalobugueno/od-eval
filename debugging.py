def save_images(dataset, c, save_dir='saved_images'):
    """
    Saves the first two images from a Keras dataset to disk.
    Works for both batched and unbatched datasets.
    """

    import os
    import matplotlib.pyplot as plt
    import numpy as np

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Get first two samples from dataset
    for i, sample in enumerate(dataset.take(c)):
        # Handle batched vs unbatched datasets
        if isinstance(sample, tuple):  # (images, labels) format
            images = sample[0]
            if len(images.shape) > 3:  # Batched dataset (batch, h, w, c)
                image = images[0]  # Take first image from batch
            else:  # Unbatched dataset (h, w, c)
                image = images
        else:  # Dataset yields just images
            if len(sample.shape) > 3:  # Batched
                image = sample[0]
            else:  # Unbatched
                image = sample

        # Convert tensor to numpy and handle range
        image = image.numpy()
        if image.dtype == np.float32:  # If normalized to [0,1]
            image = (image * 255).astype(np.uint8)

        # Save image
        plt.imsave(os.path.join(save_dir, f'image_{i}.png'), image)
        print(f"Saved image_{i}.png")


def plot_history(history):
    import matplotlib.pyplot as plt

    # Plot total loss (classification + box)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Plot classification and box losses separately
    plt.plot(history.history["classification_loss"], label="Training Classification Loss")
    plt.plot(history.history["box_loss"], label="Training Box Loss")
    plt.plot(history.history["val_classification_loss"], label="Validation Classification Loss")
    plt.plot(history.history["val_box_loss"], label="Validation Box Loss")
    plt.title("Classification vs. Box Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def visualise_bundle(image_uint8, boxes, classes, class_mapping=None, format="xywh"):
    import numpy as np
    import keras_cv

    keras_cv.visualization.plot_bounding_box_gallery(
        np.expand_dims(image_uint8, axis=0),
        value_range=(0, 255),
        rows=1,
        cols=1,
        y_true={
            "boxes": np.expand_dims(boxes, axis=0),
            "classes": np.expand_dims(classes, axis=0)
        },
        scale=5,
        bounding_box_format=format,
        class_mapping={int(k): f"cls_{k}" for k in classes} if class_mapping is None else class_mapping,
    )

def visualise_sample(ds, boxfmt='xywh', class_mapping=None):
    from datasets import get_real_samples
    import numpy as np
    import keras_cv

    for image, boxes, classes in get_real_samples(ds):
        keras_cv.visualization.plot_bounding_box_gallery(
            np.expand_dims((image.numpy() * 255).astype(np.uint8), axis=0),
            value_range=(0, 255),
            rows=1,
            cols=1,
            y_true={
                "boxes": np.expand_dims(boxes, axis=0),
                "classes": np.expand_dims(classes, axis=0)
            },
            scale=5,
            bounding_box_format=boxfmt,
            class_mapping={int(k): f"cls_{k}" for k in classes.numpy()} if class_mapping is None else class_mapping,
        )


def visualise_sample_bboxes(ds, label=None):
    """
    Requires XYWH boxes!!!
    :param ds:
    :param label: str
    :return:
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from datasets import get_real_samples

    for image, boxez, classez in get_real_samples(ds):

        for box, clazz in zip(boxez, classez):
            x, y, width, height = box

            x_min = int(x)
            y_min = int(y)
            x_max = int(x + width)
            y_max = int(y + height)

            plt.imshow(np.array(image[y_min:y_max, x_min:x_max] * 255, dtype=np.uint8))
            plt.title(
                f"cls={clazz}, h={int(height)}, w={int(width)}" if not label else f"{label} cls={clazz}, h={int(height)}, w={int(width)}")
            plt.axis(False)
            plt.show()


def test_ds(ds):
    from datasets import parse_sample_auto
    import tensorflow as tf

    for i, e in enumerate(ds.repeat(2)):
        print('s', end='')
        img, boxes, classes = parse_sample_auto(e)

        assert img is not None, "Image is None"
        assert boxes is not None, "Boxes are None"
        assert classes is not None, "Classes are None"

        height, width = img.shape[:2]
        assert height == width, f"Image size mismatch: got {width}x{height}"

        for j, (box, cls) in enumerate(zip(boxes, classes)):
            print('.', end='')

            assert not tf.reduce_any(tf.math.is_nan(box)), f"Box {j} contains NaN"
            assert not tf.reduce_any(tf.math.is_inf(box)), f"Box {j} contains Inf"

            x, y, w, h = tf.unstack(box)

            # Basic shape checks
            assert x >= 0, f"Box {j}: x < 0"
            assert y >= 0, f"Box {j}: y < 0"
            assert w > 0, f"Box {j}: w <= 0"
            assert h > 0, f"Box {j}: h <= 0"

            # Bounds check
            assert x + w <= width, f"Box {j}: x+w out of bounds ({x + w} > {width})"
            assert y + h <= height, f"Box {j}: y+h out of bounds ({y + h} > {height})"

            assert cls == 0, f"Unexpected class in box {j}: {cls}"

    print("\nOK")
