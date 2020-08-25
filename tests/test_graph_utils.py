from utils.graph_utils import image_to_vector, vector_to_image
import numpy as np
import torch

from utils.graph_utils import blockify_data, retransform_unblockify_target
from utils.graph_utils import image_to_vector, vector_to_image


def generate_image_with_increasing_elements(n, m, c, b):
    nb_elements = n * m * c * b
    x = np.arange(nb_elements).reshape((b, c, n, m))

    return x


def generate_image_with_increasing_even_elements(n, m, c, b):
    """generate an array with increasing elements where every all odd-numbered numbers
    are masked"""
    image = generate_image_with_increasing_elements(n, m, c, b)

    mask = image[0, 0, ...] % 2 == 0
    image = image * mask

    return image, mask


def checkboardpattern(n, m, d):
    """
    Create a nxn matrix with checkerboard pattern. Based on:
    https://www.tutorialspoint.com/python-program-to-print-check-board-pattern-of-n-n-using-numpy
    Args:
        d:
        n: high
        m:
    """

    x = np.zeros((n, m), dtype=int)
    x[1::2, ::2] = 1
    x[::2, 1::2] = 1

    # copy checkerboard pattern n times in the channel dimension
    x = np.repeat(x[np.newaxis, :, :], d, axis=0)
    return x


class Dataset():
    """dummy dataset class to provide dimensions to retransform_unblockify_target"""

    def __init__(self, n, m):
        self.subsample = False
        self.dimensions = True
        self.n = n
        self.m = m


class TestIncreasingArrayGenerator:
    def test_if_increasing(self):
        c = 32
        n = 10
        m = 4

        for b in [1, 4]:

            increasing_array = generate_image_with_increasing_elements(n, m, c, b)
            old_element = -1
            for b_ix in range(b):
                for c_ix in range(c):
                    for n_ix in range(n):
                        for m_ix in range(m):
                            assert old_element < increasing_array[b_ix, c_ix, n_ix, m_ix]
                            old_element = increasing_array[b_ix, c_ix, n_ix, m_ix]


class TestImageToVector:
    def test_dimensions_and_order(self):
        # define image dimensions
        n = 3
        m = 5
        c = 32

        for b in [1, 4]:
            image = generate_image_with_increasing_elements(n, m, c, b)

            mask = image[0, 0, ...] % 2 == 0
            nn_ixs = np.where(mask)  # pixel indices of non-zero elements in the mask
            image = image * mask

            vector = image_to_vector(image, nn_ixs)

            # first dimension of vector is the number of channels, total elements has to be equal
            # to the non-zero elements in the mask
            assert vector.shape[0] == b
            assert vector.shape[1] == c
            assert vector.shape[2] == nn_ixs[0].size

            # check correct order
            old_element = -1
            for b_ix in range(b):
                for c_ix in range(vector.shape[1]):
                    for n_ix in range(vector.shape[2]):
                        assert old_element < vector[b_ix, c_ix, n_ix]
                        old_element = vector[b_ix, c_ix, n_ix]

    def test_back_and_forth(self):
        # define image dimensions
        n = 3
        m = 5
        c = 32

        for b in [1, 2, 4]:
            image, mask = generate_image_with_increasing_even_elements(n, m, c, b)

            nn_ixs = np.where(mask)  # pixel indices of non-zero elements in the mask

            vector = image_to_vector(image, nn_ixs)
            image_backtransformed = vector_to_image(vec=vector, nn_ixs=nn_ixs, \
                                                    n_feat=c, batch_size=b, n=n, m=m)

            assert np.allclose(image, image_backtransformed)


# --> test blockify unblockify

class Testblockify_data:
    def test_stuff(self):
        n = 3
        m = 5
        c = 32
        c_target = 9

        for b in [1, 2, 5]:
            image, mask = generate_image_with_increasing_even_elements(n, m, c, b)
            target_image, target_mask = generate_image_with_increasing_even_elements(n, m, c_target, b)
            assert np.allclose(mask, target_mask)

            nn_ixs = np.where(mask)  # pixel indices of non-zero elemen            ts in the

            image_vector = torch.from_numpy(image_to_vector(image, nn_ixs))
            target_vector = torch.from_numpy(image_to_vector(target_image, nn_ixs))

            image_block, target_block = blockify_data(image_vector, target_vector, b)

            dataset = Dataset(n, m)

            target_image_backtrans = retransform_unblockify_target(target_block.cpu().detach().numpy(),
                                                                   nn_ixs=nn_ixs,
                                                                   batch_size=b,
                                                                   dataset=dataset)

            assert np.allclose(target_image_backtrans, target_image)

            # check correct order
            old_element = -1
            for b_ix in range(b):
                for c_ix in range(c):
                    for n_ix in range(n + m):
                        assert old_element < image_block[b_ix * (n + m) + n_ix, c_ix]
                        old_element = image_block[b_ix * (n + m) + n_ix, c_ix]

            # check correct order target
            old_element = -1
            for b_ix in range(b):
                for c_ix in range(c_target):
                    for n_ix in range(n + m):
                        assert old_element < target_block[b_ix * (n + m) + n_ix, c_ix]
                        old_element = target_block[b_ix * (n + m) + n_ix, c_ix]
