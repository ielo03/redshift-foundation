import pytest
import torch

from aion.codecs import ImageCodec
from aion.codecs.config import HF_REPO_ID
from aion.modalities import Image, LegacySurveyCatalog, LegacySurveyImage


def test_load_invalid_modality():
    """Test that loading a modality raises an error."""
    with pytest.raises(TypeError):
        ImageCodec.from_pretrained(HF_REPO_ID, modality=LegacySurveyCatalog)


def test_load_image_codec():
    """Test that loading an image codec raises an error."""
    codec_image = ImageCodec.from_pretrained(HF_REPO_ID, modality=Image)
    codec_legacy_survey_image = ImageCodec.from_pretrained(
        HF_REPO_ID, modality=LegacySurveyImage
    )
    for param_image, param_legacy_survey_image in zip(
        codec_image.parameters(), codec_legacy_survey_image.parameters()
    ):
        assert torch.equal(param_image, param_legacy_survey_image)
