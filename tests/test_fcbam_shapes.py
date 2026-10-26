import pytest
torch = pytest.importorskip("torch")
from model.fcbam import FCBAM


@pytest.mark.parametrize("height,width", [(8, 8), (9, 11)])
def test_attention_preserves_shape_and_pool_crop(height, width):
    x = torch.randn(2, 8, height, width)
    pooled = torch.nn.functional.avg_pool2d(x, 2, stride=1, padding=1)[:, :, :height, :width]
    assert pooled.shape == x.shape
    assert FCBAM(8)(x).shape == x.shape
