def grid_search(params):
    import itertools

    keys = list(params.keys())
    for values in itertools.product(*(params[key] for key in keys)):
        yield dict(zip(keys, values))


def is_batched(dataset):
    element_spec = dataset.element_spec
    # For Tensor datasets, check shape
    if hasattr(element_spec, 'shape'):
        return element_spec.shape.rank > 0 and element_spec.shape[0] is None
    # For nested structures (e.g., dicts, tuples), check each component
    elif isinstance(element_spec, (dict, tuple)):
        return any(is_batched(dataset.map(lambda x: x[key])) for key in element_spec)
    return False


def get_real_samples(ds):
    """
    deals with batched pain
    :param ds:
    :return:
    """
    for sample in ds:
        images, boxes, classes = parse_sample_auto(sample)

        if len(images.shape) == 3:  # non-batched
            images = [images]  # or np.expanddims axis = 0
            boxes = [boxes]
            classes = [classes]

        for image, boxes, classes in zip(images, boxes, classes):
            yield image, boxes, classes


def build_output_signature(d_img, d_bb, tsz, use32=True):
    """
    computes output signature based on parameters
    :param d_img: bool
    :param d_bb: bool
    :param tsz: tuple
    :return: tuple
    """

    import tensorflow as tf

    sig_x = {"images": tf.TensorSpec(shape=(*tsz, 3),
                                     dtype=tf.float32 if use32 else tf.float16)} if d_img else tf.TensorSpec(
        shape=(*tsz, 3),
        dtype=tf.float32 if use32 else tf.float16)
    sig_y = {
        "bounding_boxes": {
            "boxes": tf.TensorSpec(shape=(1, 4), dtype=tf.float32),
            "classes": tf.TensorSpec(shape=(1,), dtype=tf.int32)
        }
    } if d_bb else {
        "boxes": tf.TensorSpec(shape=(1, 4), dtype=tf.float32),
        "classes": tf.TensorSpec(shape=(1,), dtype=tf.int32)
    }

    return (sig_x, sig_y)


def parse_sample_auto(sample):
    """
    Automatically parses a sample tuple (x, y) and returns image, box, and class.
    Handles combinations of:
      - x being a plain tensor or a dict with key 'images'
      - y being a flat dict or nested under 'bounding_boxes'

    :param sample: A tuple (x, y) from the dataset
    :return: tuple (image, box, cls)
    """
    x, y = sample

    # Handle image (either raw tensor or dict with 'images')
    if isinstance(x, dict) and "images" in x:
        image = x["images"]
    else:
        image = x

    # Handle y — nested under 'bounding_boxes' or not
    if isinstance(y, dict) and "bounding_boxes" in y:
        box = y["bounding_boxes"]["boxes"]
        cls = y["bounding_boxes"]["classes"]
    else:
        box = y["boxes"]
        cls = y["classes"]

    return image, box, cls


def shape_coin_output(d_img, d_bb, img, box, cls=0):
    """
    matches output signature
    :param d_img: bool
    :param d_bb: bool
    :param img: np.array
    :param box: np.array
    :param cls: int
    :return:
    """

    import numpy as np

    boxes = np.array(box, dtype=np.float32)
    classes = np.array([], dtype=np.int32) if boxes.shape[-1] == 0 else np.array([cls], dtype=np.int32)
    # if boxes classes is empty then we just return an empty array for classes

    x = {"images": img} if d_img else img
    y = {
        "bounding_boxes": {
            "boxes": boxes,
            "classes": classes
        }
    } if d_bb else {
        "boxes": boxes,
        "classes": classes
    }

    return x, y


def multi_coin_ds(xml_paths, nobox=True, togray=False, visibility=None, format='xywh', tsz=(1024, 1024), d_img=True,
                  d_bb=True, shuffle=True, use32=True):
    """

    :param xml_paths: (xml_path, ext) = ext may be none
    :param togray: only when images have transparency. makes the background gray
    :param visibility: filter by visibility
    :param format: any supported by keras_cv.bounding_box.convert_format
    :param tsz: target size for resize (h, w)
    :param d_img:
    :param d_bb:
    :param shuffle: set to shuffle training data
    :param use32: image precision after normalisation: true=32, false=16.
    :return: Generator of image (normalised) and labels.
    """
    from pathlib import Path
    import xml.etree.ElementTree as ET
    import random
    from PIL import Image
    import numpy as np
    import keras_cv
    import tensorflow as tf

    samples = []
    for xml_path, ext in xml_paths:

        image_folder = Path(xml_path).parent / "images"
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for image in root.findall('image'):

            fn = image.get('name') + (('.' + ext) if ext is not None else '')
            full = image_folder / fn

            if not full.exists():
                raise FileNotFoundError(str(full))

            box = image.find('box')

            if box is None:
                if nobox:
                    continue
                else:
                    samples.append((
                        str(full),
                        int(image.get('width')),
                        int(image.get('height')),
                        None,
                    ))

            attr = box.find('attribute')

            vis = attr.text if attr is not None and attr.get('name') == 'visibility' else None
            if visibility is not None and vis not in visibility:
                continue

            samples.append((
                str(full),
                int(image.get('width')),  # Image's height
                int(image.get('height')),  # Image's width
                # Also known as xyxy or xmin, ymin, xmax, ymax
                (
                    float(box.get('xtl')),  # Top left X
                    float(box.get('ytl')),  # Top left Y
                    float(box.get('xbr')),  # Bottom right X
                    float(box.get('ybr'))  # Bottom right Y
                )
            ))

    if shuffle:
        random.shuffle(samples)

    def generator():
        for path, width, height, xyxy in samples:
            img = Image.open(path)

            if togray and img.mode == 'RGBA':
                bg = Image.new("RGB", img.size, (128, 128, 128))
                bg.paste(img, (0, 0), img)
                finalimg = bg
            else:
                finalimg = img.convert('RGB')

            finalimg = finalimg.resize(tsz)
            finalimg = (np.array(finalimg) / 255.0).astype(np.float32 if use32 else np.float16)

            img.close()

            if xyxy is not None:
                xtl, ytl, xbr, ybr = xyxy
                box = np.array(keras_cv.bounding_box.convert_format(
                    np.array([
                        [  # Transforms origin xyxy to resized xyxy
                            xtl * tsz[0] / width,
                            ytl * tsz[1] / height,
                            xbr * tsz[0] / width,
                            ybr * tsz[1] / height]
                    ], dtype=np.float32),
                    'xyxy', format, image_shape=(*tsz, 3)  # Then back to whatever the user wants.
                ),
                    dtype=np.float32)
            else:
                box = np.array([], dtype=np.float32)

            yield shape_coin_output(d_img, d_bb, finalimg, box)

    ds = tf.data.Dataset.from_generator(generator, output_signature=build_output_signature(d_img, d_bb, tsz, use32))
    ds = ds.apply(tf.data.experimental.assert_cardinality(len(samples)))

    return ds


def composite_ds(*generators):
    """
    composite dataset, for usage with augmentations. round robins non-batched datasets
    :param generators: coin datasets
    :return: dataset
    """
    import tensorflow as tf

    ele_spec = generators[0].element_spec
    if not all(ele_spec == g.element_spec for g in generators[1:]):
        raise ValueError("all datasets must have matching element spec")

    def compute_combined_cardinality():
        cardin = 0

        for g in generators:
            c = g.cardinality()

            if c == tf.data.INFINITE_CARDINALITY:
                return tf.data.INFINITE_CARDINALITY

            if c == tf.data.UNKNOWN_CARDINALITY:
                cardin = tf.data.UNKNOWN_CARDINALITY
                continue

            if cardin == tf.data.UNKNOWN_CARDINALITY:
                continue
            else:
                cardin += c.numpy()

        return cardin

    composite_cardinality = compute_combined_cardinality()

    def round_robin_generator():
        iterators = [iter(gen) for gen in generators]
        while iterators:  # Continue until all iterators are exhausted
            for it in list(iterators):  # Make a copy to allow removal
                try:
                    yield next(it)
                except StopIteration:
                    iterators.remove(it)  # Remove exhausted iterators

    ds = tf.data.Dataset.from_generator(round_robin_generator, output_signature=ele_spec)

    if composite_cardinality != tf.data.UNKNOWN_CARDINALITY:
        ds = ds.apply(tf.data.experimental.assert_cardinality(composite_cardinality))

    return ds


def apply_gaussian_blur(kernel_size=15):
    import tensorflow as tf
    import cv2

    def process(img, label):
        blurred_img = tf.numpy_function(
            lambda x: cv2.GaussianBlur(x, (kernel_size, kernel_size), 0),
            [img],
            img.dtype
        )
        blurred_img.set_shape(img.shape)
        return blurred_img, label

    return process


def apply_gaussian_noise(mean=0.0, stddev=0.1, monochromatic=False):
    import tensorflow as tf

    def process(img, label):
        if monochromatic:
            # Generate noise for one channel only
            noise = tf.random.normal(shape=tf.shape(img)[:2], mean=mean, stddev=stddev, dtype=img.dtype)
            noise = tf.expand_dims(noise, axis=-1)  # shape: (height, width, 1)
            noise = tf.broadcast_to(noise, tf.shape(img))  # shape: (height, width, channels)
        else:
            # Generate regular per-channel noise
            noise = tf.random.normal(shape=tf.shape(img), mean=mean, stddev=stddev, dtype=img.dtype)

        # Add noise and clip to [0.0, 1.0]
        noisy_img = tf.clip_by_value(img + noise, 0.0, 1.0)

        return noisy_img, label

    return process


def apply_directional_blur(angle_degrees=0.0, kernel_size=15):
    import tensorflow as tf
    import numpy as np
    import cv2

    def make_motion_blur_kernel(size, angle):
        # Create an empty kernel
        kernel = np.zeros((size, size), dtype=np.float32)
        # Draw a line across the center
        kernel[size // 2, :] = np.ones(size, dtype=np.float32)
        # Normalize
        kernel /= kernel.sum()
        # Rotate the kernel to the desired angle
        center = (size // 2, size // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(kernel, rot_mat, (size, size))
        # Re-normalize in case of interpolation
        rotated /= rotated.sum()
        return rotated

    def process(img, label):
        def blur_fn(x):
            k = make_motion_blur_kernel(kernel_size, angle_degrees)
            return cv2.filter2D(x, -1, k)

        blurred_img = tf.numpy_function(blur_fn, [img], img.dtype)
        blurred_img.set_shape(img.shape)
        return blurred_img, label

    return process
