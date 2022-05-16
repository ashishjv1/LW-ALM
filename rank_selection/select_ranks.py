import numpy as np
from flopco import FlopCo

from collections import defaultdict


def estimate_rank_for_compression_rate(tensor_shape,
                                       rate=2.,
                                       key='tucker2'):
    '''
        Find max rank for which inequality (initial_count / decomposition_count > rate) holds true
    '''
    min_rank = 2

    initial_count = np.prod(tensor_shape)
    if key != 'svd':
        cout, cin, kh, kw = tensor_shape

    if key == 'cp4':
        max_rank = initial_count // (rate * (cin + kh + kw + cout))
        max_rank = max(max_rank, min_rank)

    if key == 'cp3':
        max_rank = initial_count // (rate * (cin + kh * kw + cout))
        max_rank = max(max_rank, min_rank)

    elif key == 'tucker2':
        # tucker2_rank when R4=beta*R3
        if cout > cin:
            beta = 1.6
        else:
            beta = 1.

        a = 1
        b = (cin + beta * cout) / (beta * kh * kw)
        c = -cin * cout / rate / beta

        discr = b ** 2 - 4 * a * c
        max_rank = int((-b + np.sqrt(discr)) / 2 / a)
        # [R4, R3]

        max_rank = max(max_rank, min_rank)
        max_rank = (int(beta * max_rank), max_rank)

    elif key == 'svd':
        max_rank = initial_count // (rate * sum(tensor_shape[:2]))
        max_rank = max(max_rank, min_rank)

    return max_rank


def estimate_ranks_upper_bound(model, lnames_to_compress, nx=1., input_img_size=(1, 3, 224, 224),
                               device='cuda'):
    '''
    Find maximum ranks for all layers in lnames_to_compress list with given compression rate nx

    Parameters:

    model:                  Torch model
    lnames_to_compress:     List with full names of layers
    nx:                     Compression ratio
    input_img_size:         Shape of input image
    device:                 Device that stores model

    Output:
    max_ranks:              Dictionary (key = full layer name, value = max rank for given compression ratio)
    '''

    model_stats = FlopCo(model, img_size=input_img_size, device=device)

    # get maximum ranks for decomposition
    max_ranks = defaultdict()

    for mname, m in model.named_modules():
        if mname in lnames_to_compress:
            lname = mname
            _, cin, _, _ = model_stats.input_shapes[lname][0]
            _, cout, _, _ = model_stats.output_shapes[lname][0]
            kernel_size = model_stats.ltypes[lname]['kernel_size']

            tensor_shape = (cout, cin, *kernel_size)

            if kernel_size != (1, 1):
                r_pr = estimate_rank_for_compression_rate(tensor_shape[:], rate=nx, key='cp3')
            else:
                r_pr = estimate_rank_for_compression_rate(tensor_shape[:2], rate=nx, key='svd')

            max_ranks[lname] = int(r_pr)

    return max_ranks
