import argparse

import cv2
import numpy as np


def load_video(videoPath):
    image_sequence = []
    cap = cv2.VideoCapture(videoPath)

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret is False:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_sequence.append(frame)

    cap.release()

    return image_sequence


def rgb2yiq(rgb_image):
    conversion_matrix = np.array([[0.299,   0.587,  0.114],
                                  [0.596,  -0.274,  -0.322],
                                  [0.211,  -0.523,  0.312]])

    return (rgb_image @ conversion_matrix.T) / 255.0


def yiq2rgb(yiq_image):
    conversion_matrix = np.array([[1, 0.956,  0.621],
                                  [1, -0.272, -0.647],
                                  [1, -1.106, 1.703]])

    return ((yiq_image @ conversion_matrix.T) * 255).astype(np.uint8)


def pyrDown(image, kernel):
    downsized_image = cv2.filter2D(image, -1, kernel)
    return downsized_image[::2, ::2]


def pyrUp(image, kernel):
    upsampled_image = np.insert(arr=image,
                                obj=np.arange(1, image.shape[0] + 1),
                                values=0,
                                axis=0)
    upsampled_image = np.insert(arr=upsampled_image,
                                obj=np.arange(1, upsampled_image.shape[1] + 1),
                                values=0,
                                axis=1)
    return cv2.filter2D(upsampled_image, -1, 4 * kernel)


def lagrangianPyramid(image, kernel, level):
    lagrangian_pyramid = []

    for _ in range(level):
        downsampled_image = pyrDown(image, kernel)
        upsampled_image = pyrUp(downsampled_image, kernel)
        lagrangian_pyramid.append(image - upsampled_image)
        image = downsampled_image

    return lagrangian_pyramid


def main(videoPath, kernel, level):
    image_sequence = load_video(videoPath)
    image_sequence = list(map(lambda x: rgb2yiq(x), image_sequence))

    #5. Apply Temporal filter (with fft)
    #6. Amplify video
    #7. Reconstruct video (pyrup)
    #8. Convert to rgb
    #9. Save video


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Eulerian Video Magnification for heartbeats detection'
    )

    parser.add_argument(
        '--videopath',
        '-v',
        type=str,
        help='Path to the video to be used',
        required=True
    )

    parser.add_argument(
        '--level',
        '-l',
        type=int,
        help='Number of level of the Lagrangian Pyramid',
        required=False,
        default=3
    )

    args = parser.parse_args()
    videopath = args.videopath
    level = args.level
    kernel = np.array([[1,  4,  6,  4, 1],
                       [4, 16, 24, 16, 4],
                       [6, 24, 36, 24, 6],
                       [4, 16, 24, 16, 4],
                       [1,  4,  6,  4, 1]]) / 256

    main(videopath, kernel, level)
