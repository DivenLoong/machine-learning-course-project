import numpy as np
import torch

from drive_vessel.metrics import binary_metrics, summarize_metrics
from drive_vessel.rle import decode_mask, encode_mask


def test_rle_known_example():
    mask = np.zeros((3, 4), dtype=np.uint8)
    mask[0, 0:3] = 1
    mask[2, 1] = 1
    assert encode_mask(mask) == "1 3 10 1"
    assert np.array_equal(decode_mask("1 3 10 1", (3, 4)), mask)


def test_rle_empty_mask():
    mask = np.zeros((8, 8), dtype=np.uint8)
    assert encode_mask(mask) == ""
    assert not decode_mask("", (8, 8)).any()


def test_binary_dice():
    target = np.zeros((10, 10), dtype=bool)
    prediction = np.zeros_like(target)
    target[:2] = True
    prediction[:1] = True
    result = binary_metrics(target, prediction)
    assert np.isclose(result["dice"], 2 / 3)
    assert "id" not in summarize_metrics([{"id": "1", **result}])


def test_model_output_shapes():
    from drive_vessel.models import FRUNet, LadderNet, SmallUNet, TimmUNet

    models = [
        FRUNet(feature_scale=4),
        LadderNet(planes=4, layers=3),
        SmallUNet(base_channels=4, in_channels=3),
        TimmUNet("convnext_atto.d2_in1k", pretrained=False),
    ]
    x = torch.rand(1, 3, 64, 64)
    for model in models:
        model.eval()
        with torch.no_grad():
            output = model(x)
        assert output.shape == (1, 1, 64, 64), (type(model).__name__, output.shape)


def test_fr_grayscale_normalization_precedes_crop():
    from drive_vessel.data import RetinaDataset, Sample

    image = np.zeros((32, 32), dtype=np.float32)
    image[:, 16:] = 1.0
    sample = Sample(image_id="synthetic", image_path=__import__("pathlib").Path("synthetic.png"), mask_path=None)
    dataset = RetinaDataset(
        [sample],
        size=32,
        training=True,
        augment=True,
        cache=True,
        train_patch_size=16,
        input_channels=1,
        normalization="unit",
    )
    dataset._cached[sample.image_id] = (
        image,
        np.zeros((32, 32), dtype=np.float32),
    )
    item = dataset[0]
    assert item["image"].shape == (1, 16, 16)
    assert item["image"].min() >= 0.0
    assert item["image"].max() <= 1.0


def test_mild_augmentation_crops_to_requested_patch():
    from drive_vessel.data import RetinaDataset, Sample

    image = np.zeros((64, 64), dtype=np.float32)
    sample = Sample(
        image_id="synthetic-mild",
        image_path=__import__("pathlib").Path("synthetic.png"),
        mask_path=None,
    )
    dataset = RetinaDataset(
        [sample],
        size=64,
        training=True,
        augment=True,
        augment_profile="mild",
        cache=True,
        train_patch_size=32,
        input_channels=1,
        normalization="unit",
    )
    dataset._cached[sample.image_id] = (
        image,
        np.zeros((64, 64), dtype=np.float32),
    )
    item = dataset[0]
    assert item["image"].shape == (1, 32, 32)


def test_clahe_frangi_produces_two_channels():
    from drive_vessel.data import image_to_tensor, preprocess_image

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[..., 1] = 128
    preprocessed = preprocess_image(image, "clahe_frangi")
    tensor = image_to_tensor(preprocessed, 2, "unit")
    assert tensor.shape == (2, 32, 32)
