import numpy as np

'''
Pick a point uniformly from the unit circle
'''
def circle_uniform_pick(size, out = None):
    if out is None:
        out = np.empty((size, 2))

    angle = 2 * np.pi * np.random.random(size)
    out[:,0], out[:,1] = np.cos(angle), np.sin(angle)    

    return out

def cross_product_matrix_batched(U):
    batch_size = U.shape[0]
    result = np.zeros(shape=(batch_size, 3, 3))
    result[:, 0, 1] = -U[:, 2]
    result[:, 0, 2] = U[:, 1]
    result[:, 1, 0] = U[:, 2]
    result[:, 1, 2] = -U[:, 0]
    result[:, 2, 0] = -U[:, 1]
    result[:, 2, 1] = U[:, 0]
    return result

'''
Von Mises-Fisher distribution, ie. isotropic Gaussian distribution defined over
a sphere.
    mus   => mean directions
    kappa => concentration

Uses numerical tricks described in "Numerically stable sampling of the von 
Mises Fisher distribution on S2 (and other tricks)" by Wenzel Jakob
'''

def sample_von_mises_3d(mus, kappa, out=None):
    size = mus.shape[0]

    # Generate the samples for mu=(0, 0, 1)
    eta = np.random.random(size)
    tmp = 1. - (((eta - 1.) / eta) * np.exp(-2. * kappa))
    W = 1. + (np.log(eta) + np.log(tmp)) / kappa

    V = np.empty((size, 2))
    circle_uniform_pick(size, out = V)
    V *= np.sqrt(1. - W ** 2)[:, None]

    if out is None:
        out = np.empty((size, 3))

    out[:, 0], out[:, 1], out[:, 2] = V[:, 0], V[:, 1], W

    angles = np.arccos(mus[:, 2])
    mask = angles != 0.
    angles = angles[mask]
    mus = mus[mask]
    
    axis = np.zeros(shape=mus.shape)
    axis[:, 0] = -mus[:, 1]
    axis[:, 1] = mus[:, 0]
    
    axis /= np.sqrt(np.sum(axis ** 2, axis=1))[:, None]
    rot = np.cos(angles)[:, None, None] * np.identity(3)[None, :, :]
    rot += np.sin(angles)[:, None, None] * cross_product_matrix_batched(axis)
    rot += (1. - np.cos(angles))[:, None, None] *  np.matmul(axis[:, :, None], axis[:, None, :])
    
    out[mask] = (rot @ out[mask, :, None])[:, :, 0]
    return out


if __name__ == '__main__':
    from math import sqrt
    mus = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
    [1 / sqrt(2), -1 / sqrt(2), 0],
    [-1 / sqrt(2), -1 / sqrt(2), 0],
    [-1 / sqrt(2), 1 / sqrt(2), 0],
    [-1 / sqrt(2), 0., -1 / sqrt(2)],
    [1 / sqrt(2), 0., 1 / sqrt(2)]])
    print(mus)

    sampled = sample_von_mises_3d(mus, 100000)
    print(sampled)

