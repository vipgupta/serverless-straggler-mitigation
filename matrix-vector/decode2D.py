import numpy as np
import pywren
import time
import numpywren
from numpywren import matrix, matrix_utils
from numpywren import binops
from numpywren.matrix_init import shard_matrix, local_numpy_init, reshard_down
from numpywren.matrix_utils import chunk

def ind1Dto2D(i, len_A_coded, num_parity_blocks):
    if i < len_A_coded:
        return i//num_parity_blocks, i%num_parity_blocks
    else:
        return i - len_A_coded, num_parity_blocks

def ind2Dto1D(i,j, len_A_coded, num_parity_blocks):
    if j < num_parity_blocks:
        return i*num_parity_blocks + j
    else:
        return i + len_A_coded


def peel_row(y_local, i, bitmask, num_parity_blocks, len_A_coded, shard_size ):
    if bitmask[i, num_parity_blocks] == 0:
        ind = ind2Dto1D(i, num_parity_blocks, len_A_coded, num_parity_blocks)
        total = y_local[ind*shard_size:(ind + 1)*shard_size]
        for k in range(num_parity_blocks):
            if bitmask[i, k] == 0:
                ind = ind2Dto1D(i, k, len_A_coded, num_parity_blocks)
                print("row ind used", ind)
                total = total - y_local[ind*shard_size:(ind + 1)*shard_size]
        a = [ind for (ind, val) in enumerate(bitmask[i]) if val == 1]
        a = a[0]
        print("Filling row singleton", ind2Dto1D(i, a, len_A_coded, num_parity_blocks))
        return total, ind2Dto1D(i, a, len_A_coded, num_parity_blocks)
    else:
        total = None
        for k in range(num_parity_blocks):
            if total is None:
                ind = ind2Dto1D(i, k, len_A_coded, num_parity_blocks)
                total = y_local[ind * shard_size:(ind + 1) * shard_size]
            else:
                ind = ind2Dto1D(i, k, len_A_coded, num_parity_blocks)
                total = total + y_local[ind * shard_size:(ind + 1) * shard_size]
        print("Filling row singleton", ind2Dto1D(i, num_parity_blocks, len_A_coded, num_parity_blocks))
        return total, ind2Dto1D(i, num_parity_blocks, len_A_coded, num_parity_blocks)


def peel_col(y_local, j, bitmask, coding_length, len_A_coded, shard_size, num_parity_blocks):
    if bitmask[coding_length, j] == 0:
        ind = ind2Dto1D(coding_length, j, len_A_coded, num_parity_blocks)
        total = y_local[ind*shard_size:(ind + 1)*shard_size]
        for k in range(coding_length):
            if bitmask[k, j] == 0:
                ind = ind2Dto1D(k, j, len_A_coded, num_parity_blocks)
                print("col ind used", ind)
                total = total - y_local[ind * shard_size:(ind + 1) * shard_size]
        a = [ind for (ind, val) in enumerate(bitmask[:, j]) if val == 1]
        a = a[0]
        print("Filling col singleton", ind2Dto1D(a, j, len_A_coded, num_parity_blocks))
        return total, ind2Dto1D(a, j, len_A_coded, num_parity_blocks)
    else:
        total = None
        for k in range(coding_length):
            if total is None:
                ind = ind2Dto1D(k, j, len_A_coded, num_parity_blocks)
                total = y_local[ind * shard_size:(ind + 1) * shard_size]
            else:
                ind = ind2Dto1D(k, j, len_A_coded, num_parity_blocks)
                total = total + y_local[ind * shard_size:(ind + 1) * shard_size]
        print("Filling col singleton", ind2Dto1D(coding_length, j, len_A_coded, num_parity_blocks))
        return total, ind2Dto1D(coding_length, j, len_A_coded, num_parity_blocks)


def decode_vector(Y, num_parity_blocks):
    print("DECODING STARTED")
    block_idxs_exist = set([x[0] for x in Y.block_idxs_exist])
    y_local = np.zeros(Y.shape)
    shard_size = Y.shard_sizes[0]
    vector_length_blocks = int(num_parity_blocks*(Y.shape[0]//shard_size - 1 - num_parity_blocks))//(num_parity_blocks+1)
    coding_length = vector_length_blocks // num_parity_blocks
    len_A_coded = vector_length_blocks + num_parity_blocks
    bitmask = np.ones((coding_length + 1, num_parity_blocks + 1))
    for r in block_idxs_exist:
        y_local[r*shard_size:(r + 1)*shard_size] = Y.get_block(r,0)
        i, j = ind1Dto2D(r, len_A_coded, num_parity_blocks)
        bitmask[i, j] = 0

    while (bitmask.sum() > 0):
        row_sum = bitmask.sum(axis=1)
        r = [ind for (ind, val) in enumerate(row_sum) if val == 1]
        print("row singletons", r)
        for rr in r:
            y_local_block, ind = peel_row(y_local, rr, bitmask, num_parity_blocks, len_A_coded, shard_size)
            y_local[ind * shard_size:(ind + 1) * shard_size] = y_local_block
        bitmask[r] = 0

        col_sum = bitmask.sum(axis=0)
        c = [ind for (ind, val) in enumerate(col_sum) if val == 1]
        print("col singletons", c)
        for cc in c:
            y_local_block,ind = peel_col(y_local, cc, bitmask, coding_length, len_A_coded, shard_size, num_parity_blocks)
            y_local[ind * shard_size:(ind + 1) * shard_size] = y_local_block
        bitmask[:, c] = 0
    y_local = y_local[0:vector_length_blocks*shard_size]
    return y_local