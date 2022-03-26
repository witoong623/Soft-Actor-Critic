import unittest
import cv2
import numpy as np

class TestObservationTransform(unittest.TestCase):
    W = 512
    H = 256

    def test_rgb_bgr_not_equal(self):
        bgr_img = cv2.imread('images/architecture.png')
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        with np.testing.assert_raises(AssertionError):
            np.testing.assert_array_equal(bgr_img, rgb_img)

        with np.testing.assert_raises(AssertionError):
            np.testing.assert_array_equal(bgr_img, bgr_img[:,:,::-1])

    def test_resize_INTER_NEAREST(self):
        bgr_img = cv2.imread('images/architecture.png')
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        bgr_resized_img = cv2.resize(bgr_img, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        rgb_resized_img = cv2.resize(rgb_img, (self.W, self.H), interpolation=cv2.INTER_NEAREST)

        np.testing.assert_array_equal(bgr_resized_img, rgb_resized_img[:,:,::-1])
        np.testing.assert_array_equal(bgr_resized_img[:,:,::-1], rgb_resized_img)

    def test_resize_default(self):
        bgr_img = cv2.imread('images/architecture.png')
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        bgr_resized_img = cv2.resize(bgr_img, (self.W, self.H))
        rgb_resized_img = cv2.resize(rgb_img, (self.W, self.H))

        np.testing.assert_array_equal(bgr_resized_img, rgb_resized_img[:,:,::-1])
        np.testing.assert_array_equal(bgr_resized_img[:,:,::-1], rgb_resized_img)
