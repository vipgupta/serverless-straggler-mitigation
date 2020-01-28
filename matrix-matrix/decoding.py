"""
File:        decoding.py
Authors:     Vipul Gupta and Dominic Carrano
Created:     January 2019

Decoding of a locally recoverable product code.
"""

import numpy as np
import pywren
import numpywren
from copy import deepcopy
from numpywren import matrix, matrix_utils
from numpywren import binops
from numpywren.matrix_init import shard_matrix, local_numpy_init, reshard_down
from numpywren.matrix_utils import chunk

def systematicize(X_coded, blocks_per_parity):
    """
    Create a numpywren BigMatrix using the systematic part of X_coded.
    """
    async def coding_function(self, loop, i, j):  
        row_parity_index = i // blocks_per_parity
        col_parity_index = j // blocks_per_parity
        i_coded = i + row_parity_index
        j_coded = j + col_parity_index
        return X_coded.get_block(i_coded, j_coded)
    return coding_function

def decode_gemm(N, num_parity, C_coded, id):
    """
    Given the product of two coded matrices, C_coded, use a peeling decoder
    to recover the data portion as soon as enough workers have completed.
    
    Note: This function is run by each worker on one block within the resultant
    product of coded matrices, rather than on the ENTIRE resultant matrix. This
    is why the 4th argument, id, is needed - to tell us where C_coded lies in the
    full, larger matrix.
    
    Params
    ======
    N : int
        Total number of row blocks in C_coded.
        
    num_parity : int
        Total number of parity blocks in C_coded.
        
    C_coded : numpywren.matrix.BigMatrix
        A single block of the resultant product, which we decode.
    
    id : tuple
        The index of C_coded within the full matrix it came from.
    """
    import numpy as np
    from copy import deepcopy
    
    # Global indices of this worker's block
    block_row_global, block_col_global = id[0], id[1]
    
    # decoder dimension
    bpp = N // num_parity
        
    def ind_bitmask_to_matrix(i, j):
        return (i + block_row_global * bpp, j + block_col_global * bpp)
    
    def ind_matrix_to_bitmask(x, y):
        return (x - block_row_global * bpp, y - block_col_global * bpp)
    
    def cant_be_decoded(bitmask):
        """
        Given a bitmask where entry (i, j) is set to False (0) iff block (i, j)
        within C_coded has been computed, determine if we can fully decode C_coded
        with the peeling decoder using the currently finished blocks.
        """
        num_missing_blocks = bitmask.sum()
        while num_missing_blocks > 0:
            # zero out row singletons in the bitmask
            row_sum = bitmask.sum(axis=1)
            r = [ind for (ind, val) in enumerate(row_sum) if val == 1]
            bitmask[r] = 0    
            
            # zero out column singletons in the bitmask
            col_sum = bitmask.sum(axis=0)
            c = [ind for (ind, val) in enumerate(col_sum) if val == 1]
            bitmask[:, c] = 0   
            
            # can't peel anything, so we're stuck until more workers finish
            num_singletons = len(r) + len(c)
            if num_singletons == 0:
                return 1
            num_missing_blocks = bitmask.sum()
        
        return 0 # success; we can decode now
    
    def peel_row(x, bitmask):
        """Decode row x of C_coded."""
        a = [ind for (ind, val) in enumerate(np.squeeze(bitmask[x, :])) if val == 1]
        to_fill = ind_bitmask_to_matrix(x, a[0]) 
        
        # Missing block is a data block
        if a[0] != bpp - 1: 
            ind = ind_bitmask_to_matrix(x, bpp - 1)
            total = np.squeeze(C_coded.get_block(*ind))
            for l in range(bpp - 1):
                if bitmask[x, l] == 0:
                    ind = ind_bitmask_to_matrix(x, l)
                    total = total - np.squeeze(C_coded.get_block(*ind))
            C_coded.put_block(total, *to_fill)
            
        # Missing block is a parity block
        else: 
            total = None
            for k in range(bpp - 1):
                ind = ind_bitmask_to_matrix(x, k)
                if total is None:
                    total = np.squeeze(C_coded.get_block(*ind))
                else:
                    total = total + np.squeeze(C_coded.get_block(*ind))
            ind_parity = ind_bitmask_to_matrix(x, bpp-1)
            C_coded.put_block(total, *ind_parity)
        return 
        
    def peel_col(y, bitmask):
        """Decode column y of C_coded."""
        a = [ind for (ind, val) in enumerate(np.squeeze(bitmask[:, y])) if val == 1]
        to_fill = ind_bitmask_to_matrix(a[0], y)
        
        # Missing block is a data block
        if a[0] != bpp - 1:
            ind = ind_bitmask_to_matrix(bpp - 1, y)
            total = np.squeeze(C_coded.get_block(*ind))
            for l in range(bpp-1):
                if bitmask[l,y] == 0:
                    ind = ind_bitmask_to_matrix(l, y)
                    total = total - np.squeeze(C_coded.get_block(*ind))
            C_coded.put_block(total, *to_fill)
        
        # Missing block is a parity block
        else:
            total = None
            for k in range(bpp-1):
                ind = ind_bitmask_to_matrix(k,y)
                if total is None:
                    total = np.squeeze(C_coded.get_block(*ind))
                else:
                    total = total + C_coded.get_block(*ind)
            ind_parity = ind_bitmask_to_matrix(bpp-1,y)
            C_coded.put_block(total, *ind_parity)
        return 
    
    """Decoding stage 1: Wait for enough results to come in for decoding."""
    
    # bitmask[i][j] = 1 if block (i, j) has NOT completed
    bitmask = np.ones((bpp, bpp)) 
    
    # For debugging
    recovered_blocks = []           
    
    # Continue until we're in a state that allows us to decode
    while cant_be_decoded(deepcopy(bitmask)):
        all_existing_blocks = C_coded.block_idxs_exist
        for block in all_existing_blocks:
            # New block came in, convert its global indices to local ones and
            # mark it in the bitmask as recovered
            if block not in recovered_blocks:
                recovered_blocks.append(block)
                try:
                    [i, j] = block
                    if i // bpp == block_row_global and j // bpp == block_col_global:
                        local_idx = ind_matrix_to_bitmask(i, j)
                        bitmask[local_idx] = 0 
                except Exception as e:
                    print(e)
        time.sleep(2) # give workers time to make some progress
        
    """
    Decoding stage 2: Applying a peeling decoder to recover the results.
    
    At this point, even if other workers finish, we don't bother with them, as we're 
    guaranteed to have enough results to fully recover C after STAGE 1 finishes.
    """
    # bitmask[:-1, :-1] represents the systematic part of the code
    num_missing_blocks = bitmask[:-1, :-1].sum() 
    num_rows = np.shape(bitmask)[0]
    num_cols = np.shape(bitmask)[1]
    while num_missing_blocks > 0:
        # Peel along the smaller dimension first
        if num_cols <= num_rows:
            # Peel row singletons
            row_sum = bitmask.sum(axis=1)
            r = np.argwhere(row_sum == 1)
            for x in r:
                x = x[0]
                peel_row(x, bitmask)
                bitmask[x, :] = 0 

            # Peel column singletons
            col_sum = bitmask.sum(axis=0)
            d = np.argwhere(col_sum == 1)
            for x in d:
                x = x[0]
                peel_col(x, bitmask)
                bitmask[:, x] = 0
        else: 
            # Peel column singletons
            col_sum = bitmask.sum(axis=0)
            d = np.argwhere(col_sum == 1)
            for x in d:
                x = x[0]
                peel_col(x, bitmask)
                bitmask[:, x] = 0
            
            # Peel row singletons
            row_sum = bitmask.sum(axis=1)
            r = np.argwhere(row_sum == 1)
            for x in r:
                x = x[0]
                peel_row(x, bitmask)
                bitmask[x, :] = 0 
        
        # Update and repeat
        num_missing_blocks = bitmask[:-1, :-1].sum()
            
    return 1