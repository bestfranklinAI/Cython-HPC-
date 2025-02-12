# Gauss-Seidel and Jacobi Solvers Assignment

## Task Description

This assignment involves the development and optimization of iterative solvers for partial differential equations. The key tasks include:

- Developing a Gauss-Seidel solver using Python constructs (lists, arrays, or NumPy) and a vectorized Jacobi solver.
- Profiling the solvers to identify computational bottlenecks.
- Using Cython to annotate and optimize the most computationally expensive parts of the code.
- Porting the solver to Nvidia GPUs using PyTorch and CuPy, employing vectorized operations for efficiency.
- Measuring and comparing the performance of CPU vs. GPU implementations across different grid sizes.
- Saving the final grid matrix to an HDF5 file for further analysis.

## Assignment Description

This assignment is designed to explore multiple approaches to optimize iterative solvers using various Python tools and libraries. It demonstrates the progression from a basic Python implementation to advanced optimization techniques, including Cython for CPU acceleration and PyTorch/CuPy for GPU acceleration. The assignment emphasizes performance profiling, code optimization, and data storage, providing a comprehensive case study in high-performance numerical computing.