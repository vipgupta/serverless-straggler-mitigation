"""
File:        encoding.py
Authors:     Vipul Gupta and Dominic Carrano
Created:     January 2019

Implementation of encoding for a locally recoverable product code.
"""

import numpy as np
import sys
import pywren
import numpywren
from copy import deepcopy
from numpywren import matrix, matrix_utils
from numpywren import binops
from numpywren.matrix_init import shard_matrix, local_numpy_init, reshard_down
from numpywren.matrix_utils import chunk

def get_block_wrapper(mtx, x):
    a = mtx.get_block(*x)
    return 0

def make_coding_function(X, blocks_per_parity):
    """
    Sum-of-blocks encoding of X, adding a parity block after every blocks_per_parity blocks in the 
    vertical direction. 
    
    If blocks_per_parity does not divide X, then the last len(X._block_idxs(0)) % blocks_per_parity
    blocks will not have a corresponding parity block after them.
    """
    num_normal_blocks = len(X._block_idxs(0))
    num_parity_blocks = num_normal_blocks // blocks_per_parity # to be computed/added by coding_function
    num_total_blocks = num_normal_blocks + num_parity_blocks  # total number after encoding    
    
    async def coding_function(self, loop, i, j):  
        is_parity_block = (i + 1) % (blocks_per_parity + 1) == 0
        if i < 0 or i > num_total_blocks:
            raise ValueError("ERROR: Index is greater than the total number of blocks.")
        elif not is_parity_block:
            # Normal block
            block_index = i - i // (blocks_per_parity + 1)            # The block's index in the uncoded version of X
            return X.get_block(block_index, j)
        else:
            # Parity block, compute and return it
            parity_block_index = i // (blocks_per_parity + 1)          # index of the parity block (1st, 2nd, etc.)
            first_block_index = parity_block_index * blocks_per_parity # index in uncoded X of first block to sum over
                        
            X_sum = None
            for block_index in range(first_block_index, first_block_index + blocks_per_parity):
                if X_sum is None:
                    X_sum = X.get_block(block_index, j)
                else:
                    X_sum = X_sum + X.get_block(block_index, j) 
            self.put_block(X_sum, i, j)
            return X_sum            
    return coding_function

def start_encode_mtx(M, blocks_per_parity, s3_key):
    """ 
    Apply a (blocks_per_parity + 1, blocks_per_parity) MDS code to the matrix M every 
    blocks_per_parity rows by summing up the previous blocks_per_parity rows. 
    
    Params
    ======
    M : numpywren.matrix.BigMatrix
        The matrix to encode.
    
    blocks_per_parity : int
        The number of input blocks sum up in creating each parity block. Note that as
        this number increases, less redundancy is provided.
        
    s3_key : str
        The storage key for Amazon S3.
        
    Returns
    =======
    M_coded : numpywren.matrix.BigMatrix
        The encoded matrix.
        
    futures : list
        List of the pywren futures.
        
    num_workers : int
        The number of workers.
    """
    # Useful definitions
    num_row_blocks = M.shape[0] // M.shard_sizes[0]
    num_col_blocks = M.shape[1] // M.shard_sizes[1]
    num_parity = num_row_blocks // blocks_per_parity   # total number of parity blocks that will be added
    coded_shape = (M.shape[0] + num_parity * M.shard_sizes[0], M.shape[1])
    
    # Ensure no blocks will go uncoded
    if not num_row_blocks % blocks_per_parity == 0:
        raise ValueError("Number of row blocks ({0}) is not divisible \
                         by number of blocks per parity ({1})".format(num_row_blocks, blocks_per_parity))
    
    # Create the coded matrix object
    coding_fn = make_coding_function(M, blocks_per_parity)
    M_coded = matrix.BigMatrix(s3_key, shape=coded_shape, shard_sizes=M.shard_sizes, write_header=True, parent_fn=coding_fn)
    M_coded.delete() # Only needed if you reuse the same s3_key (if the blocks already exist, no work will be done here)
    
    # Generate encoding indices
    encode_idx = []
    for j in range(num_col_blocks):
        for i in range(1, num_parity + 1):
            encode_idx.append((i * (blocks_per_parity + 1) - 1, j))
    num_workers = len(encode_idx)
    
    # Encode the matrix
    pwex = pywren.lambda_executor()
    futures = pwex.map(lambda x: get_block_wrapper(M_coded, x), encode_idx)
    return M_coded, futures, num_workers