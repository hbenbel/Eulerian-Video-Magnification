import numpy as np
import tqdm

from processing import idealTemporalBandpassFilter, pyrDown, pyrUp, rgb2yiq


def generateGaussianPyramid(image, kernel, level):
    image_shape = [image.shape[:2]]
    downsampled_image = image.copy()

    for _ in range(level):
        downsampled_image = pyrDown(image=downsampled_image, kernel=kernel)
        image_shape.append(downsampled_image.shape[:2])

    gaussian_pyramid = downsampled_image
    for curr_level in range(level):
        gaussian_pyramid = pyrUp(
                            image=gaussian_pyramid,
                            kernel=kernel,
                            dst_shape=image_shape[level - curr_level - 1]
                        )

    return gaussian_pyramid


def getGaussianPyramids(images, kernel, level):
    gaussian_pyramids = []

    for image in tqdm.tqdm(images,
                           ascii=True,
                           desc='Gaussian Pyramids Generation'):

        gaussian_pyramid = generateGaussianPyramid(
                                    image=rgb2yiq(image),
                                    kernel=kernel,
                                    level=level
                        )
        gaussian_pyramids.append(gaussian_pyramid)

    return np.asarray(gaussian_pyramids)


def filterGaussianPyramids(pyramids,
                           fps,
                           freq_range,
                           alpha,
                           attenuation):

    filtered_pyramids = idealTemporalBandpassFilter(
                            images=pyramids,
                            fps=fps,
                            freq_range=freq_range
                        ).astype(np.float32)

    filtered_pyramids *= alpha
    filtered_pyramids[:, :, :, 1:] *= attenuation

    return filtered_pyramids
