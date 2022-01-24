import numpy as np


gaussian_kernel = (
    np.array(
        [
            [1,  4,  6,  4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1,  4,  6,  4, 1]
        ]
    )
    / 256
)


yiq_from_rgb = (
    np.array(
            [
                [0.29900000,  0.58700000,  0.11400000],
                [0.59590059, -0.27455667, -0.32134392],
                [0.21153661, -0.52273617,  0.31119955]
            ]
        )
    ).astype(np.float32)


rgb_from_yiq = np.linalg.inv(yiq_from_rgb)
