import argparse

import numpy as np
import tqdm

from constants import gaussian_kernel
from processing import (generatePyramid, idealTemporalBandpassFilter,
                        loadVideo, reconstructImage, rgb2yiq, saveVideo)


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


def augmentPyramids(pyramids, level, fps, freq_range, amplification):
    sequences = []

    for pyramid_level in tqdm.tqdm(
                            range(level),
                            ascii=True,
                            desc="Pyramids Augmentation"):

        images = np.stack(pyramids[:, pyramid_level]).astype(np.float32)

        filtered_images = idealTemporalBandpassFilter(
                            images=images,
                            fps=fps,
                            freq_range=freq_range,
                            amplification=amplification
                        ).astype(np.float32)

        sequences.append(images + filtered_images)

    return sequences


def getOutputVideo(filtered_images, kernel):
    video = []

    for i in tqdm.tqdm(
                range(filtered_images[0].shape[0]),
                ascii=True,
                desc="Video Reconstruction"):

        image_all_level = list(map(lambda x: x[i], filtered_images))
        reconstructed_image = reconstructImage(image_all_level, kernel)

        video.append(reconstructed_image)

    return video


def main(video_path, kernel, level, freq_range, amplification, saving_path):
    images, fps = loadVideo(video_path)
    gaussian_pyramids, _ = getPyramids(images, kernel, level)

    filtered_pyramids_level = augmentPyramids(
                                pyramids=gaussian_pyramids,
                                level=level,
                                fps=fps,
                                freq_range=freq_range,
                                amplification=amplification
                            )

    output_video = getOutputVideo(filtered_pyramids_level, kernel)

    saveVideo(output_video, saving_path, fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Eulerian Video Magnification for heartbeats detection"
    )

    parser.add_argument(
        "--video_path",
        "-v",
        type=str,
        help="Path to the video to be used",
        required=True
    )

    parser.add_argument(
        "--level",
        "-l",
        type=int,
        help="Number of level of the Gaussian/Laplacian Pyramid",
        required=False,
        default=3
    )

    parser.add_argument(
        "--amplification",
        "-a",
        type=int,
        help="Amplification factor",
        required=False,
        default=30
    )

    parser.add_argument(
        "--min_frequency",
        "-minf",
        type=float,
        help="Minimum allowed frequency",
        required=False,
        default=0.833
    )

    parser.add_argument(
        "--max_frequency",
        "-maxf",
        type=float,
        help="Maximum allowed frequency",
        required=False,
        default=1
    )

    parser.add_argument(
        "--saving_path",
        "-s",
        type=str,
        help="Saving path of the magnified video",
        required=True
    )

    args = parser.parse_args()
    video_path = args.video_path
    level = args.level
    amplification = args.amplification
    frequency_range = [args.min_frequency, args.max_frequency]
    saving_path = args.saving_path

    main(
        video_path=video_path,
        kernel=gaussian_kernel,
        level=level,
        freq_range=frequency_range,
        amplification=amplification,
        saving_path=saving_path
    )
