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


def visualise_sample(ds, **plotargs):
    from datasets import get_real_samples
    import numpy as np
    import keras_cv

    for image, boxes, classes in get_real_samples(ds):
        image = (image.numpy() * 255).astype(np.uint8)
        # it expects batched shit...

        keras_cv.visualization.plot_bounding_box_gallery(
            np.expand_dims(image, axis=0),
            value_range=(0, 255),
            rows=1,
            cols=1,
            y_true={
                "boxes": np.expand_dims(boxes, axis=0),
                "classes": np.expand_dims(classes, axis=0)
            },
            scale=5,
            bounding_box_format="xywh",
            class_mapping={int(k): f"cls_{k}" for k in classes.numpy()},
            **plotargs,
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
