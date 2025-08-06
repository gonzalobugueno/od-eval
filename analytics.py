def metrics(ds, denominator=5):
    from datasets import get_real_samples
    from fractions import Fraction
    import pandas as pd

    data = []

    for image, boxes, classes in get_real_samples(ds):
        height, width = image.shape[:2]

        for box, clazz in zip(boxes, classes):
            xtl, ytl, xbr, ybr = box

            bw = int((xbr - xtl) * width)
            bh = int((ybr - ytl) * height)

            data.append({
                'bbox_width': bw,
                'bbox_height': bh,
                'aspect_ratio': bw / bh
            })

    df = pd.DataFrame(data)
    print(df.describe())

    ar = df['aspect_ratio'].apply(lambda x: Fraction(x).limit_denominator(denominator))

    mean_frac = Fraction(df['aspect_ratio'].mean()).limit_denominator(10)

    print(
        "Mean aspect ratio:"
        f"{mean_frac.numerator}:{mean_frac.denominator}"
    )

    return ar
