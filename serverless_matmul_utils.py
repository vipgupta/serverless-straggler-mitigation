# Copied from ServerlessMatmulDebugging.ipynb on 10/10/2019
ALL_COMPLETED = 1
ANY_COMPLETED = 2
ALWAYS = 3

import numpy as np
import sys
import pywren
import numpywren
import time
from copy import deepcopy
from numpywren import matrix, matrix_utils
from numpywren import binops
from numpywren.matrix_init import shard_matrix, local_numpy_init, reshard_down
from numpywren.matrix_utils import chunk
import warnings
warnings.filterwarnings('ignore')

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
            # Parity block, compute and set/return it
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
            print("[cant_be_decoded] Current number of missing blocks:", num_missing_blocks)
            print("[cant_be_decoded] Current bitmask:", bitmask)
            
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
        print("Filling row singleton", to_fill)  
        
        # Missing block is a data block
        if a[0] != bpp - 1: 
            ind = ind_bitmask_to_matrix(x, bpp - 1)
            print("parity used", ind)
            total = np.squeeze(C_coded.get_block(*ind))
            for l in range(bpp - 1):
                if bitmask[x, l] == 0:
                    ind = ind_bitmask_to_matrix(x, l)
                    print("Row ind used", ind)
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
            C_coded.put_block(total, *ind_parity) # fails here
        return 
        
    def peel_col(y,bitmask):
        """Decode column y of C_coded."""
        a = [ind for (ind, val) in enumerate(np.squeeze(bitmask[:, y])) if val == 1]
        to_fill = ind_bitmask_to_matrix(a[0],y)
        print("Filling col singleton", to_fill)  
        
        # Missing block is a data block
        if a[0] != bpp - 1:
            ind = ind_bitmask_to_matrix(bpp-1,y)
            print("parity used", ind)
            total = np.squeeze(C_coded.get_block(*ind))
            for l in range(bpp-1):
                if bitmask[l,y] == 0:
                    ind = ind_bitmask_to_matrix(l,y)
                    print("Col ind used", ind)
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
            C_coded.put_block(total, *ind_parity) # fails here
        return 
    
    """Decoding stage 1: Wait for enough results to come in for decoding."""
    
    # bitmask[i][j] = 1 if block (i, j) has NOT completed
    bitmask = np.ones((bpp, bpp)) 
    print (bitmask.shape)
    
    # For debugging
    recovered_blocks = []           
    
    # Continue until we're in a state that allows us to decode
    while cant_be_decoded(deepcopy(bitmask)):
        all_existing_blocks = C_coded.block_idxs_exist
        print("[decode_gemm] Number of existing blocks in C_coded: ", len(all_existing_blocks))
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
    
    print("DECODING STARTED, BITMASK:")
    print(bitmask)
    
    # bitmask[:-1, :-1] represents the systematic part of the code
    num_missing_blocks = bitmask[:-1, :-1].sum() 
    num_rows = np.shape(bitmask)[0]
    num_cols = np.shape(bitmask)[1]
    while num_missing_blocks > 0:
        print("num_missing_blocks > 0, starting iteration of decoding")
        # Peel along the smaller dimension first
        if num_cols <= num_rows:
            # Peel row singletons
            row_sum = bitmask.sum(axis=1)
            r = np.argwhere(row_sum == 1)
            print("Row singletons", r)
            for x in r:
                print("row done", x[0])
                x = x[0]
                peel_row(x, bitmask)
                bitmask[x, :] = 0 

            # Peel column singletons
            col_sum = bitmask.sum(axis=0)
            d = np.argwhere(col_sum == 1)
            print("Column singletons", d)
            for x in d:
                x = x[0]
                print("column done", x)
                peel_col(x, bitmask)
                bitmask[:, x] = 0
        else: 
            # Peel column singletons
            col_sum = bitmask.sum(axis=0)
            d = np.argwhere(col_sum == 1)
            print("Column singletons", d)
            for x in d:
                x = x[0]
                print("column done", x)
                peel_col(x, bitmask)
                bitmask[:, x] = 0
            
            # Peel row singletons
            row_sum = bitmask.sum(axis=1)
            r = np.argwhere(row_sum == 1)
            print("Row singletons", r)
            for x in r:
                print("row done", x[0])
                x = x[0]
                peel_row(x, bitmask)
                bitmask[x, :] = 0 
        
        # Update and repeat
        num_missing_blocks = bitmask[:-1, :-1].sum()
            
    return 1

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

def pywren_gemm(id, A, B, C, num_col_blocks):
    """
    Computes the id-th block of (C = A * B.T).
    """
    i = id[0]
    j = id[1]
    a = np.zeros((A.shard_sizes[0], B.shard_sizes[0]))
    for x in range(num_col_blocks):
        a += A.get_block(i,x).dot(B.get_block(j,x).T)
    C.put_block(a, i, j)
    return id

def get_block_wrapper(mtx, x):
    a = mtx.get_block(*x)
    return 0

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
    M_coded = matrix.BigMatrix(s3_key, \
                               shape=coded_shape, \
                               shard_sizes=M.shard_sizes, \
                               write_header=True, \
                               parent_fn=coding_fn)
    M_coded.delete()
    
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

def gemm_coded(A, B, blocks_per_parity, s3_key, completion_pct=.7, encode_A=True, encode_B=True, verbose=True, np_A=-1, np_B=-1):
    """
    Compute A * B.T using a product code for redundancy.

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
    
    verbose : bool
        Whether or not to print out progress.
    
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
    # Sanity checks
    if (not encode_A) and np_A == -1:
        raise ValueError("You must provide the number of parity blocks in A if you pre-encoded it.")
    if (not encode_B) and np_B == -1:
        raise ValueError("You must provide the number of parity blocks in B if you pre-encoded it.")
        
    min_encoding_completion_pct = .90
    min_matmul_completion_pct = completion_pct
    
    ### Stage 1: Encoding ###
    start = time.time()
    if encode_A or encode_B:
        if verbose:
            print("gemm_coded: Stage 1 Starting (Encoding)")
            print("gemm_coded: Need >={}% of encoding workers to finish before moving on".format(min_encoding_completion_pct*100))

        # Start the encoding, if requested - TODO add some unique name based on the time
        num_workers = 0
        if encode_A:
            A_coded, futures_encode_A, num_workers_A = start_encode_mtx(A, blocks_per_parity, "A_coded")
            num_workers += num_workers_A
        if encode_B:
            B_coded, futures_encode_B, num_workers_B = start_encode_mtx(B, blocks_per_parity, "B_coded")
            num_workers += num_workers_B

        # Wait for enough encoding workers to finish
        num_done = 0
        while num_done < min_encoding_completion_pct * num_workers:
            fs_A, fs_B = [], []
            if encode_A:
                fs_A, _ = pywren.wait(futures_encode_A, return_when=ANY_COMPLETED)
            if encode_B:
                fs_B, _ = pywren.wait(futures_encode_B, return_when=ANY_COMPLETED)
            num_done = len(fs_A) + len(fs_B)
            if verbose:
                print("gemm_coded: {0} of {1} encoding workers done".format(num_done, num_workers))
    
    # Display final stats
    end = time.time()
    t_enc = end - start
    if not encode_A:
        A_coded = A
    if not encode_B:
        B_coded = B
   
    if verbose:
        fs_A, fs_B = [], []
        if encode_A:
            fs_A, _ = pywren.wait(futures_encode_A, return_when=ALWAYS)
        if encode_B:
            fs_B, _ = pywren.wait(futures_encode_B, return_when=ALWAYS)
        num_done = len(fs_A) + len(fs_B)
        print("gemm_coded: Stage 1 Done, {0} of {1} encoding workers done".format(num_done, num_workers))
        print("gemm_coded: Stage 1 Time = {} sec".format(round(t_enc, 2))) 
        
    # Store futures to return
    fs_enc = []
    if encode_A and encode_B:
        fs_enc = encode_A + encode_B
    elif encode_A:
        fs_enc = encode_A
    elif encode_B:
        fs_enc = encode_B
    
    ### Stage 2: Multiply ###
    if verbose:
        print("gemm_coded: Stage 2 Starting (Multiply)")
        print("gemm_coded: Need >={}% of matmul workers to finish before moving on".format(min_matmul_completion_pct*100))
        
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
    num_row_blocks_C = C_coded.shape[0] // C_coded.shard_sizes[0]
    num_col_blocks_C = C_coded.shape[1] // C_coded.shard_sizes[1]
    num_cols_coded = A_coded.shape[1] // A_coded.shard_sizes[1]  # Inner dimension of the coded multiplication
    block_idx_C = C_coded.block_idxs
    total_matmul_workers = len(block_idx_C)
    
    # Setup multiplication jobs and run
    t_comp_start = time.time()
    np.random.shuffle(block_idx_C) # Avoid bad locality of which blocks straggle via randomization
    pwex = pywren.lambda_executor()

    futures_matmul = pwex.map(lambda x: pywren_gemm(x, A_coded, B_coded, C_coded, num_cols_coded), block_idx_C)
    fs_done_matmul = []
    completed_matmul_workers = 0
    while completed_matmul_workers < min_matmul_completion_pct * total_matmul_workers:
        fs_done_matmul, _ = pywren.wait(futures_matmul, return_when=ANY_COMPLETED)
        completed_matmul_workers = len(fs_done_matmul)
        if verbose:
            time.sleep(1) # prevent stdout flooding 
            print(completed_matmul_workers, "of", total_matmul_workers, "matmul workers done")
    t_comp_end = time.time()
    t_comp = t_comp_end - t_comp_start

    # Display final stats
    if verbose:
        fs_done_matmul, _ = pywren.wait(futures_matmul, return_when=ALWAYS)
        completed_matmul_workers = len(fs_done_matmul)
        print("gemm_coded: Stage 2 Done, {0} of {1} multiply workers done".format(completed_matmul_workers,total_matmul_workers))
        print("gemm_coded: Stage 2 Time = {} sec".format(round(t_comp, 2)))
        
    ### Stage 3: Decoding ###
    if verbose:
        print("gemm_coded: Stage 3 Starting (Decode)")
    
    # Generate decoding indices
    decode_idx = [(i, j) for i in range(num_parity_A) for j in range(num_parity_B)]
    total_decoding_workers = len(decode_idx)
    
    # Setup and run decoding workers
    t_dec_start = time.time()
    print("Arguments to decode_gemm in pwex.map:")
    print("num_row_blocks_C:", num_row_blocks_C)
    print("num_parity_A:", num_parity_A)
    print("decode_idx:", decode_idx)
    #fs_done_decode = map(lambda x: decode_gemm(num_row_blocks_C, num_parity_A, C_coded, x), decode_idx) # try local?
    futures_decode = pwex.map(lambda x: decode_gemm(num_row_blocks_C, num_parity_A, C_coded, x), decode_idx)
    fs_done_decode = []
    completed_decode_workers = 0
    while completed_decode_workers < total_decoding_workers and len(C_coded.block_idxs_not_exist) > 0:
        fs_done_decode, _ = pywren.wait(futures_decode, return_when=ANY_COMPLETED)
        completed_decode_workers = len(fs_done_decode)
        if verbose:
            time.sleep(1) # prevent stdout flooding
            print(completed_decode_workers, "of", total_decoding_workers, "decoding workers done")
    t_dec_end = time.time()
    t_dec = t_dec_end - t_dec_start
    if verbose:
        print("gemm_coded: Stage 3 Done (Decode)")
        print("gemm_coded: Stage 3 Time = {} sec".format(round(t_dec, 2)))
    
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
    
    # Holds regardless of whether or not we encode inside this function since encoding
    # doesn't change the shard sizes
    C_shard_sizes = (A.shard_sizes[0], B.shard_sizes[0])
    
    # Generate worker indices
    get_systematic_part = systematicize(C_coded, blocks_per_parity)
    C = matrix.BigMatrix(s3_key, \
                     shape=(C_num_rows, C_num_cols), \
                     shard_sizes=C_shard_sizes, \
                     parent_fn=get_systematic_part)
    C.delete() # this one is needed
    to_read = C.block_idxs
    total_systematicize_workers = len(to_read)
    
    # Run jobs
    fs_done_systematicize = []
    futures_systematicize = pwex.map(lambda x: get_block_wrapper(C, x), to_read)
    completed_systematicize_workers = 0
    while completed_systematicize_workers < total_systematicize_workers:
        fs_done_systematicize, _ = pywren.wait(futures_systematicize, return_when=ANY_COMPLETED)
        completed_systematicize_workers = len(fs_done_systematicize)
        if verbose:
            time.sleep(1) # prevent stdout flooding
            print(completed_systematicize_workers, "of", total_systematicize_workers, "systematicize workers done")
    if verbose:
        print("gemm_coded: All done")
    
    return C, t_enc, t_comp, t_dec, fs_enc, fs_done_matmul, fs_done_decode, fs_done_systematicize

def gemm_recompute(A, B, thresh, s3_key, verbose=False):
    """    
    Compute A * B.T via speculative execution, aka recomputing.

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
    
    verbose : bool
        Whether or not to print out progress.

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
    if verbose:
        print("gemm_recompute: Initializing results matrix")
        
    ### Initialize result matrix ###
    num_col_blocks = A.shape[1] // A.shard_sizes[1]
    shard_sizes = (A.shard_sizes[0], B.shard_sizes[0])
    C = matrix.BigMatrix(s3_key, shape=(A.shape[0], B.shape[0]), shard_sizes=shard_sizes, \
                         autosqueeze=False, \
                         write_header=True)
    C.delete() # TODO is this really needed??
    if verbose:
        print("gemm_recompute: Results matrix initialized")
        print("gemm_recompute: Stage 1 Starting (Initial Compute)")

    ### Stage 1: Compute thresh percentage of the results ###
    t_comp_start = time.time()
    pwex = pywren.lambda_executor()
    futures = pwex.map(lambda x: pywren_gemm(x, A, B, C, num_col_blocks), C.block_idxs)
    futures_dones = []
    while len(futures_dones) < thresh * len(futures):
        futures_dones, _ = pywren.wait(futures, return_when=ANY_COMPLETED)
        if verbose:
            time.sleep(1) # prevent stdout flooding
            print (len(futures_dones), "done out of", len(futures), "(1st batch)")
    t_comp_end = time.time()
    t_straggle_start = time.time()
    t_comp = t_comp_end - t_comp_start

    if verbose:
        print("gemm_recompute: Stage 1 Done (Initial Compute)")
        print("gemm_recompute: Stage 1 Time = {} sec".format(round(t_comp, 2)))
        print("{0} done out of {1} (1st batch)".format(len(futures_dones), len(futures)))
        print("gemm_recompute: Stage 2 Starting (Recompute)")

    ### Stage 2: Recompute stragglers ###
    futures_stragglers = pwex.map(lambda x: pywren_gemm(x, A, B, C, num_col_blocks), C.block_idxs_not_exist)
    fs_dones_stragglers = []
    
    # Stopping condition can be all of C's blocks existing OR all the original futures existing
    # Long term, would be nice to change this to using some hash table to record the futures' blocks and which
    # ones finish so we can just check that either the 2nd or 1st copy finished, but this proxy works quite well
    while len(C.block_idxs_not_exist) > 0: 
        print("len(C.block_idxs_not_exist)",len(C.block_idxs_not_exist))
        futures_dones, _ = pywren.wait(futures, return_when=ALWAYS)
        fs_dones_stragglers, _ = pywren.wait(futures_stragglers, return_when=ALWAYS)
        if verbose:
            print(len(futures_dones), "done out of", len(futures), "(1st batch)")
            print(len(fs_dones_stragglers), "done out of", len(futures_stragglers), "(2nd batch)")
    t_straggle_end = time.time()
    t_straggle = t_straggle_end - t_straggle_start
    time.sleep(3) # prevent stdout flooding

    futures_dones, _ = pywren.wait(futures, return_when=ALWAYS)
    fs_dones_stragglers, _ = pywren.wait(futures_stragglers, return_when=ALWAYS)
    if verbose:
        print(len(futures_dones), "done out of", len(futures), "(1st batch)")
        print(len(fs_dones_stragglers), "done out of", len(futures_stragglers), "(2nd batch)")
        print("gemm_recompute: Stage 2 Done (Recompute)")
        print("gemm_recompute: Stage 2 Time = {} sec".format(round(t_straggle, 2)))
        print("gemm_recompute done")

    return C, t_comp, t_straggle, futures_dones, fs_dones_stragglers