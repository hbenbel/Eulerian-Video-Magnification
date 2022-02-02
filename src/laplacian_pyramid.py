import numpy as np
import tqdm

from processing import butterBandpassFilter, pyrDown, pyrUp, rgb2yiq


def generateLaplacianPyramid(image, kernel, level):
    laplacian_pyramid = []
    prev_image = image.copy()

    for _ in range(level):
        downsampled_image = pyrDown(prev_image, kernel)
        upsampled_image = pyrUp(downsampled_image, kernel, image.shape[:2])
        laplacian_pyramid.append(prev_image - upsampled_image)
        prev_image = downsampled_image

    return laplacian_pyramid


def getLaplacianPyramids(images, kernel, level):
    laplacian_pyramids = []

    for image in tqdm.tqdm(images,
                           ascii=True,
                           desc="Laplacian Pyramids Generation"):

        laplacian_pyramid = generateLaplacianPyramid(
                                    image=rgb2yiq(image),
                                    kernel=kernel,
                                    level=level
                        )
        laplacian_pyramids.append(laplacian_pyramid)

    return np.asarray(laplacian_pyramids, dtype='object')


def filterLaplacianPyramids(pyramids,
                            level,
                            fps,
                            freq_range,
                            alpha,
                            lambda_cutoff,
                            attenuation):

    filtered_pyramids = []
    delta = lambda_cutoff / (8 * (1 + alpha))

    for lvl in tqdm.tqdm(range(level),
                         ascii=True,
                         desc="Laplacian Pyramids Augmentation"):

        images = np.stack(pyramids[:, lvl]).astype(np.float32)

        (height, width, _) = images[0].shape
        lambd = (((height ** 2) + (width ** 2)) ** 0.5) / 3
        new_alpha = (lambd / (8 * delta)) - 1

        filtered_images = butterBandpassFilter(
                            images=images,
                            fps=fps,
                            freq_range=freq_range
                        ).astype(np.float32)

        filtered_images *= min(alpha, new_alpha)
        filtered_images[:, :, :, 1:] *= attenuation

        filtered_pyramids.append(filtered_images)

    return filtered_pyramids
