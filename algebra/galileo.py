import numpy as np
from lie_learn.representations.SO3.pinchon_hoggan.pinchon_hoggan_dense import Jd, rot_mat


def random_rotation():
    # print('generate random rotation')
    α, β, 𝜸 = np.random.randn(3)
    # print('α, β, 𝜸 = {}'.format((α, β, 𝜸)))
    R = rot_mat(alpha=α, beta=β, gamma=𝜸, J=Jd[1], l=1)
    return R


def galilean_transformation(v, R):
    return np.block([
        [1, 0, 0, 0],
        [v,    R   ]
    ])


four_repr = (
    np.array([
        [
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,-1],
            [0,0,1,0]
        ],
        [
            [0,0,0,0],
            [0,0,0,1],
            [0,0,0,0],
            [0,-1,0,0]
        ],
        [
            [0,0,0,0],
            [0,0,-1,0],
            [0,1,0,0],
            [0,0,0,0]
        ]
    ]),
    np.array([
        [
            [0,0,0,0],
            [1,0,0,0],
            [0,0,0,0],
            [0,0,0,0]
        ],
        [
            [0,0,0,0],
            [0,0,0,0],
            [1,0,0,0],
            [0,0,0,0]
        ],
        [
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [1,0,0,0]
        ]
    ])
)

galilean_transformation(np.random.randn(3,1), random_rotation())



three_repr_2d = (
    np.array([
        [
            [0,0,0],
            [0,0,-1],
            [0,1,0],
        ],
        [
            [0,0,0],
            [1,0,0],
            [0,0,0],
        ],
        [
            [0,0,0],
            [0,0,0],
            [1,0,0],
        ]
    ])
)
