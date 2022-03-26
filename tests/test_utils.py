import unittest
import torch
import numpy as np
import torchvision.transforms as T
import time

from common.utils import normalize_image


def _transform_np_image_to_tensor(imgs, normalize=True):
    tensors = []
    for i in range(imgs.shape[0]):
        img = imgs[i]
        img = img.transpose(2, 0, 1)
        if normalize:
            img = img / 255.

        img_tensor = torch.from_numpy(img)
        tensors.append(img_tensor)

    return torch.stack(tensors, dim=0)


def _transform_tensor(img_tensors, normalize=True):
    tensors = []
    for i in range(img_tensors.size(0)):
        img = img_tensors[i]

        if img.size(0) == 3:
            # correct form, do nothing
            pass
        elif img.size(-1) == 3:
            img = img.permute((2, 0, 1)).contiguous()

        assert img.size(0) == 3

        if normalize:
            img = img / 255.

        tensors.append(img)

    return torch.stack(tensors, dim=0)


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


class TestTransformObservation(unittest.TestCase):
    batch_size = 2

    def _transform_np_image_to_tensor_batch(self, imgs, normalize=True):
        # input should be (number of image, H, W, C)
        new_images = imgs.transpose(0, 3, 1, 2)

        if normalize:
            new_images = new_images / 255.

        return torch.from_numpy(new_images)

    def _transform_tensor_batch(self, img_tensors, normalize=True):
        # input should be (number of image, H, W, C) or (number of image, C, H, W)
        if img_tensors.size(1) == 3:
            pass
        elif img_tensors.size(-1) == 3:
            new_img_tensors = img_tensors.permute((0, 3, 1, 2)).contiguous()

        if normalize:
            new_img_tensors = new_img_tensors / 255.

        return new_img_tensors

    def test_np_batch_loop_equal(self):
        obs = np.random.random((self.batch_size, 64, 64, 3))

        t = time.process_time_ns()
        non_batch_ret = _transform_np_image_to_tensor(obs.copy(), normalize=False)
        elapsed = time.process_time_ns() - t
        print('non batch time', elapsed)

        t = time.process_time_ns()
        batch_ret = self._transform_np_image_to_tensor_batch(obs.copy(), normalize=False)
        elapsed = time.process_time_ns() - t
        print('batch time', elapsed)

        np.testing.assert_allclose(non_batch_ret.numpy(), batch_ret.numpy())

    def test_tensor_batch_loop_equal(self):
        obs = torch.rand((self.batch_size, 64, 64, 3))

        t = time.process_time_ns()
        non_batch_ret = _transform_tensor(torch.clone(obs), normalize=False)
        elapsed = time.process_time_ns() - t
        print('non batch time', elapsed)

        t = time.process_time_ns()
        batch_ret = self._transform_tensor_batch(torch.clone(obs), normalize=False)
        elapsed = time.process_time_ns() - t
        print('batch time', elapsed)

        np.testing.assert_allclose(non_batch_ret.numpy(), batch_ret.numpy())

    def test_np_batch_loop_equal_normalize(self):
        obs = np.random.random((self.batch_size, 64, 64, 3))

        t = time.process_time_ns()
        non_batch_ret = _transform_np_image_to_tensor(obs.copy(), normalize=True)
        elapsed = time.process_time_ns() - t
        print('non batch time', elapsed)

        t = time.process_time_ns()
        batch_ret = self._transform_np_image_to_tensor_batch(obs.copy(), normalize=True)
        elapsed = time.process_time_ns() - t
        print('batch time', elapsed)

        np.testing.assert_allclose(non_batch_ret.numpy(), batch_ret.numpy())

    def test_tensor_batch_loop_equal_normalize(self):
        obs = torch.rand((self.batch_size, 64, 64, 3))

        t = time.process_time_ns()
        non_batch_ret = _transform_tensor(torch.clone(obs), normalize=True)
        elapsed = time.process_time_ns() - t
        print('non batch time', elapsed)

        t = time.process_time_ns()
        batch_ret = self._transform_tensor_batch(torch.clone(obs), normalize=True)
        elapsed = time.process_time_ns() - t
        print('batch time', elapsed)

        np.testing.assert_allclose(non_batch_ret.numpy(), batch_ret.numpy())
