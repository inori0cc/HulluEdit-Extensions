"""Cache inputs in expanded prompt coordinates, with exclusive slice ends."""


def cache_spans(image_start, image_end):
    if image_end - image_start != 576 or image_start < 1:
        raise ValueError("Unexpected image span in the expanded prompt")
    return 1, image_end - 34
