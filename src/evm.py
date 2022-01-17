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


def main(videoPath):
    image_sequence = load_video(videoPath)
    image_sequence = list(map(lambda x: rgb2yiq(x), image_sequence))

    #4. Generate a pyramid (pyrdown)
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

    args = parser.parse_args()
    videopath = args.videopath

    main(videopath)
