import numpy as np
import tqdm

from processing import (butterBandpassFilter, generatePyramid,
                        idealTemporalBandpassFilter, rgb2yiq)


def getPyramids(images, kernel, level):
    pyramids = []

    for image in tqdm.tqdm(images, ascii=True, desc="Pyramids Generation"):
        pyramid = generatePyramid(rgb2yiq(image), kernel, level)
        pyramids.append(pyramid)

    gaussian_pyramids = list(map(lambda x: x[0], pyramids))
    gaussian_pyramids = np.asarray(gaussian_pyramids, dtype='object')

    laplacian_pyramids = list(map(lambda x: x[1], pyramids))
    laplacian_pyramids = np.asarray(laplacian_pyramids, dtype='object')

    return gaussian_pyramids, laplacian_pyramids


def augmentGaussianPyramids(gaussian_pyramids,
                            level,
                            fps,
                            freq_range,
                            amplification):
    sequences = []

    for pyramid_level in tqdm.tqdm(
                            range(level),
                            ascii=True,
                            desc="Gaussian Pyramids Augmentation"):

        images = np.stack(
                    gaussian_pyramids[:, pyramid_level]
                ).astype(np.float32)

        amp = amplification if pyramid_level > 1 else 1

        filtered_images = idealTemporalBandpassFilter(
                            images=images,
                            fps=fps,
                            freq_range=freq_range
                        ).astype(np.float32)

        filtered_images[:, :, :, 1:] *= amp

        sequences.append(images + filtered_images)

    return sequences


def augmentLaplacianPyramids(gaussian_pyramids,
                             laplacian_pyramids,
                             level,
                             fps,
                             freq_range,
                             amplification):

    sequences = []

    for pyramid_level in tqdm.tqdm(
                            range(level),
                            ascii=True,
                            desc="Laplacian Pyramids Augmentation"):

        gaussian_images = np.stack(
                            gaussian_pyramids[:, pyramid_level]
                        ).astype(np.float32)
        laplacian_images = np.stack(
                            laplacian_pyramids[:, pyramid_level]
                        ).astype(np.float32)

        amp = amplification if pyramid_level > 1 else 0

        filtered_images = butterBandpassFilter(
                            images=laplacian_images,
                            fps=fps,
                            freq_range=freq_range
                        ).astype(np.float32)

        sequences.append(gaussian_images + amp * filtered_images)

    return sequences


def augmentPyramids(gaussian_pyramids,
                    laplacian_pyramids,
                    level,
                    fps,
                    freq_range,
                    amplification,
                    mode):

    if mode == 'gaussian':
        return augmentGaussianPyramids(gaussian_pyramids=gaussian_pyramids,
                                       level=level,
                                       fps=fps,
                                       freq_range=freq_range,
                                       amplification=amplification)

    elif mode == 'laplacian':
        return augmentLaplacianPyramids(gaussian_pyramids=gaussian_pyramids,
                                        laplacian_pyramids=laplacian_pyramids,
                                        level=level,
                                        fps=fps,
                                        freq_range=freq_range,
                                        amplification=amplification)
