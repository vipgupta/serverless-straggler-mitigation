"""
File:        matmul.py
Authors:     Vipul Gupta and Dominic Carrano
Created:     January 2019

General matrix multiplication implementations via coding and speculative execution.
"""

import numpy as np
import pywren
import time
from numpywren import matrix, matrix_utils
from encoding import start_encode_mtx
from decoding import systematicize, decode_gemm

ALL_COMPLETED, ANY_COMPLETED, ALWAYS = 1, 2, 3 # Return time options for pywren.wait (see http://pywren.io/docs/)
MIN_ENCODING_COMPLETION_PCT = .90              # Minimum percentage of encoding jobs that must finish before we move on 

def pywren_gemm(id, A, B, C, num_col_blocks):
    """
    Computes the block of (C = A * B.T) specified by id.
    """
    i, j = id[0], id[1]
    Cij = np.zeros((A.shard_sizes[0], B.shard_sizes[0]))
    for x in range(num_col_blocks):
        Cij += A.get_block(i, x).dot(B.get_block(j, x).T)
    C.put_block(Cij, i, j)
    return id

def gemm_coded(A, B, blocks_per_parity, s3_key, completion_pct=.7, encode_A=True, encode_B=True, np_A=-1, np_B=-1):
    """
    Compute A * B.T using a locally recoverable product code for redundancy.

    Params
    ======
    A : numpywren.matrix.BigMatrix
        First input matrix.
        
    B : numpywren.matrix.BigMatrix
        Second input matrix.
        
    blocks_per_parity : int
        Number of blocks to sum up when creating each parity block.
        
    s3_key: str
        Storage key for output matrix.
        
    completion_pct: int
        The fraction of multiplication workers that must finish before moving on to decoding.
        
    encode_A : bool
        Whether or not A needs to be encoded. 
        Allows for the user to pre-encode A if it will be used multiple times.
    
    encode_B : bool
        Whether or not B needs to be encoded.
        Allows for the user to pre-encode B if it will be used multiple times.
    
    np_A : int
        Number of parity blocks in the matrix A. Should be provided if and only if
        encode_A is set to false.
    
    np_B : int
        Number of parity blocks in the matrix B. Should be provided if and only if
        encode_B is set to false. 

    Returns
    =======
    C : numpywren.matrix.BigMatrix
        Resultant matrix product.
        
    t_enc : float
        Encoding time.
        
    t_comp : float
        Computation time.
        
    t_dec : float
        Decoding time.        
    """
    if (not encode_A) and np_A == -1:
        raise ValueError("You must provide the number of parity blocks in A if you pre-encoded it.")
    if (not encode_B) and np_B == -1:
        raise ValueError("You must provide the number of parity blocks in B if you pre-encoded it.")
    
    min_matmul_completion_pct = completion_pct
    
    # Stage 1: Encoding
    start = time.time()
    if encode_A or encode_B:
        num_workers = 0
        if encode_A:
            A_coded, futures_encode_A, num_workers_A = start_encode_mtx(A, blocks_per_parity, "A_coded")
            num_workers += num_workers_A
        if encode_B:
            B_coded, futures_encode_B, num_workers_B = start_encode_mtx(B, blocks_per_parity, "B_coded")
            num_workers += num_workers_B
        
        num_done = 0
        while num_done < MIN_ENCODING_COMPLETION_PCT * num_workers:
            fs_A, fs_B = [], []
            if encode_A:
                fs_A, _ = pywren.wait(futures_encode_A, return_when=ANY_COMPLETED)
            if encode_B:
                fs_B, _ = pywren.wait(futures_encode_B, return_when=ANY_COMPLETED)
            num_done = len(fs_A) + len(fs_B)    
    end = time.time()
    t_enc = end - start # Total encoding time
    
    # Save encoded results
    if not encode_A:
        A_coded = A
    if not encode_B:
        B_coded = B
        
    # Store futures to return
    fs_enc = []
    if encode_A and encode_B:
        fs_enc = encode_A + encode_B
    elif encode_A:
        fs_enc = encode_A
    elif encode_B:
        fs_enc = encode_B
    
    ### Stage 2: Multiply ###
        
    # Initialize output matrix
    if encode_A:
        num_parity_A = (A.shape[0] // A.shard_sizes[0]) // blocks_per_parity
        coded_shape_A = (A.shape[0] + num_parity_A * A.shard_sizes[0], A.shape[1])
    else:
        num_parity_A = np_A
        coded_shape_A = A_coded.shape
    
    if encode_B:
        num_parity_B = (B.shape[0] // B.shard_sizes[0]) // blocks_per_parity
        coded_shape_B = (B.shape[0] + num_parity_B * B.shard_sizes[0], B.shape[1])
    else:
        num_parity_B = np_B
        coded_shape_B = B_coded.shape
        
    shard_sizes_C = (A.shard_sizes[0], B.shard_sizes[0])
    C_coded = matrix.BigMatrix(s3_key + "coded", \
                               shape=(A_coded.shape[0], B_coded.shape[0]), \
                               shard_sizes=shard_sizes_C, \
                               autosqueeze=False, \
                               write_header=True)
    C_coded.delete()
        
    # Generate indices for the output matrix
    num_row_blocks_C, num_col_blocks_C  = C_coded.shape[0] // C_coded.shard_sizes[0], C_coded.shape[1] // C_coded.shard_sizes[1]
    num_cols_coded = A_coded.shape[1] // A_coded.shard_sizes[1]  # Inner dimension of the coded multiplication
    block_idx_C = C_coded.block_idxs
    num_workers = len(block_idx_C)
    
    # Setup multiplication jobs and run
    t_comp_start = time.time()
    np.random.shuffle(block_idx_C) # Avoid bad locality of which blocks straggle via randomization
    pwex = pywren.lambda_executor()

    futures_matmul = pwex.map(lambda x: pywren_gemm(x, A_coded, B_coded, C_coded, num_cols_coded), block_idx_C)
    fs_done_matmul = []
    num_done = 0
    while num_done < completion_pct * num_workers:
        fs_done_matmul, _ = pywren.wait(futures_matmul, return_when=ANY_COMPLETED)
        num_done = len(fs_done_matmul)
    t_comp_end = time.time()
    t_comp = t_comp_end - t_comp_start
        
    ### Stage 3: Decoding ###
    
    # Generate decoding indices
    decode_idx = [(i, j) for i in range(num_parity_A) for j in range(num_parity_B)]
    total_decoding_workers = len(decode_idx)
    
    # Setup and run decoding workers
    t_dec_start = time.time()
    
    #fs_done_decode = map(lambda x: decode_gemm(num_row_blocks_C, num_parity_A, C_coded, x), decode_idx) # try local?
    futures_decode = pwex.map(lambda x: decode_gemm(num_row_blocks_C, num_parity_A, C_coded, x), decode_idx)
    fs_done_decode = []
    completed_decode_workers = 0
    while completed_decode_workers < total_decoding_workers and len(C_coded.block_idxs_not_exist) > 0:
        fs_done_decode, _ = pywren.wait(futures_decode, return_when=ANY_COMPLETED)
        completed_decode_workers = len(fs_done_decode)
    t_dec_end = time.time()
    t_dec = t_dec_end - t_dec_start
    
    ### Stage 4: Systematicize (remove the parity blocks so only the original data is left) ###
    if verbose:
        print("gemm_coded: Stage 4 Starting (Systematicize)")
    
    # Define output dimensions
    if encode_A:
        C_num_rows = A.shape[0]
    else:
        C_num_rows = A.shape[0] - np_A * A.shard_sizes[0]
        
    if encode_B:
        C_num_cols = B.shape[0]
    else:
        C_num_cols = B.shape[0] - np_B * B.shard_sizes[0]
    

    C_shard_sizes = (A.shard_sizes[0], B.shard_sizes[0])    
    get_systematic_part = systematicize(C_coded, blocks_per_parity)
    C = matrix.BigMatrix(s3_key, shape=(C_num_rows, C_num_cols), shard_sizes=C_shard_sizes, parent_fn=get_systematic_part)
    C.delete()

    
    # Run jobs <- TODO this whole section's not needed though...???
    to_read = C.block_idxs
    total_systematicize_workers = len(to_read)
    fs_done_systematicize = []
    futures_systematicize = pwex.map(lambda x: get_block_wrapper(C, x), to_read)
    completed_systematicize_workers = 0
    while completed_systematicize_workers < total_systematicize_workers:
        fs_done_systematicize, _ = pywren.wait(futures_systematicize, return_when=ANY_COMPLETED)
        completed_systematicize_workers = len(fs_done_systematicize)
    
    return C, t_enc, t_comp, t_dec, fs_enc, fs_done_matmul, fs_done_decode, fs_done_systematicize

def gemm_recompute(A, B, thresh, s3_key):
    """    
    Compute A * B.T via speculative execution (i.e., recompute straggling workers).

    Params
    ======
    A : numpywren.matrix.BigMatrix
        First input matrix.
        
    B : numpywren.matrix.BigMatrix
        Second input matrix.
        
    thresh : float (in [0, 1])
        Fraction of workers that should finish before recomputing.
        
    s3_key : str
        Storage key for output matrix.

    Returns
    =======
    C : matrix.BigMatrix
        Resultant matrix product.
        
    t_comp : float
        Time for thresh percentage of the workers to finish.
        
    t_straggle : float
        Time for the remaining 1 - thresh percentage of the workers to finish after
        we begin recomputing.
    """
    if not (0 <= thresh <= 1):
        raise ValueError("thresh must be in the interval [0, 1]")
        
    # Initialize output matrix
    num_col_blocks = A.shape[1] // A.shard_sizes[1]
    shard_sizes = (A.shard_sizes[0], B.shard_sizes[0])
    C = matrix.BigMatrix(s3_key, shape=(A.shape[0], B.shape[0]), shard_sizes=shard_sizes, autosqueeze=False, write_header=True)
    C.delete() # Only needed if you reuse the same s3_key (if the blocks already exist, no work will be done here)

    # Stage 1: Compute "thresh" percentage of the results
    t_comp_start = time.time()
    pwex = pywren.lambda_executor()
    futures = pwex.map(lambda x: pywren_gemm(x, A, B, C, num_col_blocks), C.block_idxs)
    futures_dones = []
    while len(futures_dones) < thresh * len(futures):
        futures_dones, _ = pywren.wait(futures, return_when=ANY_COMPLETED)
    t_comp_end = time.time()
    t_comp = t_comp_end - t_comp_start # Total stage 1 time

    # Stage 2: Recompute straggling workers (the last 1-thresh percent of jobs)
    t_straggle_start = time.time()
    futures_stragglers = pwex.map(lambda x: pywren_gemm(x, A, B, C, num_col_blocks), C.block_idxs_not_exist)
    fs_dones_stragglers = []
    while len(C.block_idxs_not_exist) > 0: 
        futures_dones, _ = pywren.wait(futures, return_when=ALWAYS)
        fs_dones_stragglers, _ = pywren.wait(futures_stragglers, return_when=ALWAYS)
    t_straggle_end = time.time()
    t_straggle = t_straggle_end - t_straggle_start # Total stage 2 time

    # Save final futures for diagnostic
    futures_dones, _ = pywren.wait(futures, return_when=ALWAYS)
    fs_dones_stragglers, _ = pywren.wait(futures_stragglers, return_when=ALWAYS)
    return C, t_comp, t_straggle, futures_dones, fs_dones_stragglers