def apply_gaussian_blur(kernel_size=15):
    import cv2
    import tensorflow as tf

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


def apply_directional_blur(angle_degrees=0.0, kernel_size=15, float_type=None):
    import numpy as np
    import cv2
    import tensorflow as tf

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
