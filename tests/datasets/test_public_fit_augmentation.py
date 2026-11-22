import numpy as np
import pytest
from PIL import Image, ImageDraw

from scripts.experiments.public_data.public_fit_augmentation import derivative, geometry


def test_geometry_transforms_a_polygon_and_graduation_point_into_the_same_raster():
    image = Image.new("RGB", (200, 100), (10, 10, 10))
    ImageDraw.Draw(image).rectangle((40, 30, 79, 59), fill=(200, 200, 200))
    obj = dict(bbox=[.2, .3, .4, .6], polygon=[.2, .3, .4, .3, .4, .6, .2, .6],
               landmarks={"piston": [.3, .4]})
    matrix = np.array([[1., 0., 20.], [0., 1., 10.], [0., 0., 1.]])
    after, labels = geometry(image, [obj], matrix)
    assert labels[0]["bbox"] == pytest.approx([.3, .4, .5, .7])
    assert labels[0]["landmarks"]["piston"] == pytest.approx([.4, .5])
    assert after.getpixel((80, 50)) == image.getpixel((60, 40)) == (200, 200, 200)
    assert geometry(image, [obj], np.array([[1., 0., 150.], [0., 1., 0.], [0., 0., 1.]])) is None


def test_capture_keeps_box_supervision_and_masks_visibility_sensitive_fields():
    picture = Image.new("RGB", (200, 100), (90, 90, 90))
    annotation = dict(bbox=[.2, .22, .6, .85], capacity_ml=20, volume_ml=0, liquid_colour="amber",
                      landmarks={"piston": [.3, .4]}, syringe_type="20-scale")
    row = dict(split="public-fit", objects=[annotation])
    image, objects, meta = derivative(picture, row, "capture", 52)
    assert objects[0]["bbox"] == annotation["bbox"]
    assert all(objects[0][key] is None for key in annotation if key != "bbox")
    assert row["objects"][0]["volume_ml"] == 0
    assert image.tobytes() != picture.tobytes()
    assert meta["seed"] == 52
    with pytest.raises(ValueError, match="accepted public"):
        derivative(picture, {**row, "split": "fil-rouge-reference"}, "capture", 52)
