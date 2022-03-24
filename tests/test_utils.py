import unittest
import torch
import numpy as np
import torchvision.transforms as T

from common.utils import normalize_image


class TestNumpyNormalize(unittest.TestCase):
    def test_torch_np_equal(self):
        img_np = np.random.uniform(low=0.0, high=255.0, size=(2, 2, 3)).astype('uint8')

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.4640, 0.4763, 0.3560], std=[0.0941, 0.1722, 0.1883])
        ])

        mean_np = np.array([0.4640, 0.4763, 0.3560])
        std_np = np.array([0.0941, 0.1722, 0.1883])

        # dtype float32
        ret_tensor = transform(img_np)
        # dtype float64, convert to float32
        ret_np = normalize_image(img_np, mean_np, std_np).astype('float32')

        ret_tensor_np = ret_tensor.numpy().transpose((1, 2, 0))

        np.testing.assert_allclose(ret_np, ret_tensor_np, rtol=1e-06)
