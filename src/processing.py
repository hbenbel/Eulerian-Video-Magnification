import cv2
import numpy as np
import tqdm

from constants import rgb_from_yiq, yiq_from_rgb


def loadVideo(videoPath):
    image_sequence = []
    video = cv2.VideoCapture(videoPath)
    fps = video.get(cv2.CAP_PROP_FPS)

    while video.isOpened():
        ret, frame = video.read()

        if ret is False:
            break

        image_sequence.append(frame[:, :, ::-1])

    video.release()

    return np.asarray(image_sequence), fps


def rgb2yiq(rgb_image):
    image = rgb_image.astype(np.float32)
    return image @ yiq_from_rgb.T


def yiq2rgb(yiq_image):
    image = yiq_image.astype(np.float32)
    return image @ rgb_from_yiq.T


def pyrDown(image, kernel):
    return cv2.filter2D(image, -1, kernel)[::2, ::2]


def pyrUp(image, kernel, dst_source=None):
    dst_height = image.shape[0] + 1
    dst_width = image.shape[1] + 1

    if dst_source is not None:
        dst_height -= (dst_source[0] % image.shape[0] != 0)
        dst_width -= (dst_source[1] % image.shape[1] != 0)

    height_indexes = np.arange(1, dst_height)
    width_indexes = np.arange(1, dst_width)

    upsampled_image = np.insert(image, height_indexes, 0, axis=0)
    upsampled_image = np.insert(upsampled_image, width_indexes, 0, axis=1)

    return cv2.filter2D(upsampled_image, -1, 4 * kernel)


def generatePyramid(image, kernel, level):
    gaussian_pyramid = []
    laplacian_pyramid = []

    for _ in range(level):
        gaussian_pyramid.append(image)
        downsampled_image = pyrDown(image, kernel)

        upsampled_image = pyrUp(downsampled_image, kernel, image.shape[:2])
        laplacian_pyramid.append(image - upsampled_image)

        image = downsampled_image

    gaussian_pyramid = np.asarray(gaussian_pyramid, dtype='object')
    laplacian_pyramid = np.asarray(laplacian_pyramid, dtype='object')

    return gaussian_pyramid, laplacian_pyramid


def idealTemporalBandpassFilter(images,
                                fps=30,
                                freq_range=[0.833, 1],
                                axis=0,
                                amplification=30):

    fft = np.fft.fft(images, axis=axis)
    frequencies = np.fft.fftfreq(images.shape[0], d=1.0/fps)

    low = (np.abs(frequencies - freq_range[0])).argmin()
    high = (np.abs(frequencies - freq_range[1])).argmin()

    fft[:low] = 0
    fft[-high:] = 0
    fft[high:-high] = 0

    filtered_images = np.fft.ifft(fft, axis=0).real
    filtered_images[:, :, :, 1:] *= amplification

    return filtered_images


def normalizeReconstructedImage(image):
    normalized_image = (image - image.min()) / (image.max() - image.min())
    return (255 * normalized_image).astype(np.uint8)


def reconstructImage(pyramid, kernel):
    reconstructed_image = yiq2rgb(pyramid[0]).astype(np.float32)

    for level in range(1, len(pyramid)):
        tmp = yiq2rgb(pyramid[level])
        for curr_level in range(level):
            tmp = pyrUp(tmp, kernel, pyramid[level - curr_level - 1].shape[:2])
        reconstructed_image += tmp.astype(np.float32)

    return normalizeReconstructedImage(reconstructed_image)


def saveVideo(video, saving_path, fps=30):
    (height, width) = video[0].shape[0:2]

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(saving_path, fourcc, fps, (width, height))

    for i in tqdm.tqdm(range(len(video)), ascii=True, desc="Saving Video"):
        writer.write(video[i][:, :, ::-1])

    writer.release()
