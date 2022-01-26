import argparse

import tqdm

from constants import gaussian_kernel
from processing import loadVideo, reconstructImage, saveVideo
from pyramids import augmentPyramids, getPyramids


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


def main(video_path,
         kernel,
         level,
         freq_range,
         amplification,
         saving_path,
         mode):

    images, fps = loadVideo(video_path)
    gaussian_pyramids, laplacian_pyramids = getPyramids(images, kernel, level)

    filtered_pyramids_level = augmentPyramids(
                                gaussian_pyramids=gaussian_pyramids,
                                laplacian_pyramids=laplacian_pyramids,
                                level=level,
                                fps=fps,
                                freq_range=freq_range,
                                amplification=amplification,
                                mode=mode
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

    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        help="Type of pyramids to use (gaussian or laplacian)",
        choices=['gaussian', 'laplacian'],
        required=False,
        default='gaussian'
    )

    args = parser.parse_args()
    video_path = args.video_path
    level = args.level
    amplification = args.amplification
    frequency_range = [args.min_frequency, args.max_frequency]
    saving_path = args.saving_path
    mode = args.mode

    main(
        video_path=video_path,
        kernel=gaussian_kernel,
        level=level,
        freq_range=frequency_range,
        amplification=amplification,
        saving_path=saving_path,
        mode=mode
    )
