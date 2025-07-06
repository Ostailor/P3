# Advanced Ansatz Benchmarking

This directory collects results from the ADAPT-VQE and k-UpCCGSD studies. Each
subfolder corresponds to an ansatz/optimizer pair and stores convergence data,
optimized parameters and metadata written by `scripts/adapt_vqe.py` and
`scripts/kupccgsd_vqe.py`.

## Summary Table

| Ansatz       | Optimizer | Final Energy | Iterations | Runtime (s) | Depth | Parameters |
|--------------|-----------|--------------|------------|-------------|-------|------------|
| adapt_vqe    | cobyla    | None         | 0          | 5.01937     | None  | N/A        |
| adapt_vqe    | adam      | -857.89235   | 400        | 30.63416    | 3     | N/A        |
| adapt_vqe    | spsa      | None         | 0          | 4.99031     | None  | N/A        |
| kupccgsd_vqe | cobyla    | -864.56048   | 15         | 1323.98259  | 1     | 252        |
| kupccgsd_vqe | adam      | -864.69062   | 114        | 112.42949   | 1     | 252        |
| kupccgsd_vqe | spsa      | -856.59469   | 61         | 172.77958   | 1     | 252        |

## Comparison

- **Lowest energy**: k-UpCCGSD with Adam converged to the lowest energy
  (-864.69 Ha).
- **Fastest runtime**: ADAPT-VQE with Adam finished in about 31 seconds but
  achieved higher energy (-857.89 Ha).
- COBYLA only made progress for k-UpCCGSD (8 iterations) but required the
  longest runtime (~1324 s).
- SPSA failed to improve the ADAPT-VQE ansatz and was slower than Adam for
  k-UpCCGSD.

In practice, ADAPT-VQE is quick to run but less accurate, while k-UpCCGSD is
computationally heavier yet reaches lower energies. Adam provided the best
balance of accuracy and speed across both ansatzes.

## Next Steps

- Explore additional values of `k` for the UpCCGSD ansatz.
- Tune SPSA and COBYLA hyperparameters to check if convergence can be
  improved.
- Integrate the novel optimizer (see `novel_optimizer.py`) into these runs and
  update this summary accordingly.