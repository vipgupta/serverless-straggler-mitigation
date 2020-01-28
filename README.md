# Serverless Straggler Mitigation

Matrix multiplication for serverless cloud computing with built-in straggler mitigation techniques. 

## What is this?

In large scale distributed computing jobs, a small fraction (~2% on AWS Lambda) of workers &mdash; referred to as *stragglers* &mdash; take significantly longer than the median job time to finish. 

We provide an implementation of matrix-matrix multiplication, which is an atomic operation in many high performance computing and machine learning applications (e.g., Kernel Ridge Regression, Alternating Least Squares, SVD), using two different straggler mitigation techniques:
- A locally-recoverable product code, described more in https://arxiv.org/abs/2001.07490.
- Speculative execution, where straggling nodes are recomputed. 

## Interface

The matrix multiplication abstractions are in the `gemm_coded` and `gemm_recompute` functions inside `matrix-matrix/compute.py`. All straggler mitigation is handled behind the scenes, so that these functions can be used as black boxes: pass in your matrices, and get back the product.

## Tests and Examples

We provide a brief Jupyter Notebook containing a suite of test cases, which also serve as examples, inside `matrix-matrix/Tests.ipynb`.
