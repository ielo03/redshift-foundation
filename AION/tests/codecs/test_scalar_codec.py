import pytest
import torch

from aion.codecs import GridScalarCodec, LogScalarCodec, MultiScalarCodec, ScalarCodec
from aion.codecs.config import HF_REPO_ID
from aion.modalities import (
    HSCAG,
    HSCAI,
    HSCAR,
    HSCAY,
    HSCAZ,
    Dec,
    GaiaFluxBp,
    # Gaia modalities
    GaiaFluxG,
    GaiaFluxRp,
    GaiaParallax,
    GaiaXpBp,
    GaiaXpRp,
    HSCMagG,
    HSCMagI,
    HSCMagR,
    HSCMagY,
    HSCMagZ,
    HSCShape11,
    HSCShape12,
    HSCShape22,
    LegacySurveyEBV,
    LegacySurveyFluxG,
    LegacySurveyFluxI,
    LegacySurveyFluxR,
    LegacySurveyFluxW1,
    LegacySurveyFluxW2,
    LegacySurveyFluxW3,
    LegacySurveyFluxW4,
    LegacySurveyFluxZ,
    LegacySurveyShapeE1,
    LegacySurveyShapeE2,
    LegacySurveyShapeR,
    Ra,
    Z,
)


@pytest.mark.parametrize(
    "codec_class,modality",
    [
        # LogScalarCodec tests
        (LogScalarCodec, LegacySurveyFluxG),
        (LogScalarCodec, LegacySurveyFluxR),
        (LogScalarCodec, LegacySurveyFluxI),
        (LogScalarCodec, LegacySurveyFluxZ),
        (LogScalarCodec, LegacySurveyFluxW1),
        (LogScalarCodec, LegacySurveyFluxW2),
        (LogScalarCodec, LegacySurveyFluxW3),
        (LogScalarCodec, LegacySurveyFluxW4),
        (LogScalarCodec, LegacySurveyShapeR),
        # Gaia LogScalarCodec tests
        (LogScalarCodec, GaiaFluxG),
        (LogScalarCodec, GaiaFluxBp),
        (LogScalarCodec, GaiaFluxRp),
        (LogScalarCodec, GaiaParallax),
        # ScalarCodec tests
        (ScalarCodec, LegacySurveyShapeE1),
        (ScalarCodec, LegacySurveyShapeE2),
        (ScalarCodec, LegacySurveyEBV),
        (ScalarCodec, HSCMagG),
        (ScalarCodec, HSCMagR),
        (ScalarCodec, HSCMagI),
        (ScalarCodec, HSCMagZ),
        (ScalarCodec, HSCMagY),
        (ScalarCodec, HSCShape11),
        (ScalarCodec, HSCShape22),
        (ScalarCodec, HSCShape12),
        (ScalarCodec, HSCAG),
        (ScalarCodec, HSCAR),
        (ScalarCodec, HSCAI),
        (ScalarCodec, HSCAZ),
        (ScalarCodec, HSCAY),
        # Gaia ScalarCodec tests
        (ScalarCodec, Ra),
        (ScalarCodec, Dec),
        # Gaia MultiScalarCodec tests
        (MultiScalarCodec, GaiaXpBp),
        (MultiScalarCodec, GaiaXpRp),
        # Grid tokenizer
        (GridScalarCodec, Z),
    ],
)
def test_scalar_tokenizer(data_dir, codec_class, modality):
    codec = codec_class.from_pretrained(HF_REPO_ID, modality=modality)
    codec.eval()
    input_batch = torch.load(
        data_dir / f"{modality.name}_codec_input_batch.pt", weights_only=False
    )
    reference_encoded_batch = torch.load(
        data_dir / f"{modality.name}_codec_encoded_batch.pt", weights_only=False
    )
    reference_decoded_batch = torch.load(
        data_dir / f"{modality.name}_codec_decoded_batch.pt", weights_only=False
    )

    with torch.no_grad():
        output = codec.encode(modality(value=input_batch))
        decoded_output = codec.decode(output)

    assert torch.allclose(output, reference_encoded_batch)
    assert torch.allclose(decoded_output.value, reference_decoded_batch)
