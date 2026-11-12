# Interpretation and measurement boundaries

The numerical columns are a new, internally matched RTX 5090 / PyTorch 2.7.1+cu128 audit. They replace the old RTX 4090 / PyTorch 2.5.1+cu124 columns as a group. They must not be described as the old measurements with only two missing rows added.

Forward time is solver-only wall time with CUDA synchronization and the frozen diagnostic/counter extraction. It excludes density readout, actor-loss evaluation, backward, and training logging. The frozen audit chooses the lower middle of ten timing repetitions per case, then the table takes the median of nine case values. A cap is a maximum, so cap500 need not be slower than cap100 if both stop early. No timing reruns were selected to enforce monotonic ordering.

The tighter setting's gradient P95 differs visibly from the old table. The raw records show changed FP32 stopping iterations on some identical inputs. For seed 1 near 200k, the old run stopped at 86 iterations with 0.441803% gradient error, while the new run stopped at 83 with 0.0318507%. Across all valid cases, FP64 reference-readout differences remain at most 3.3751e-14 and validity masks match. These observations document the runtime comparison without attributing it to an isolated hardware-only effect. All five new-setting batch gradients, including the larger seed 2 near 200k error, are retained in the P95.

Gradient errors are relative L2 errors of full-batch configured SAC actor-loss gradients, expressed as percentages. They can exceed 100%. Log-density errors are absolute differences in nats. All numerical errors refer to the frozen A25-trained policy input bank, not the final independently trained cap10/cap100 policies. Four batches with invalid FP64 references remain preserved. Their full-batch errors are unavailable rather than counted as zero.

Success uses every final evaluation episode, with 1000 stochastic episodes per trained seed and eval seed 26152. Mean and sample SD use all three training seeds, with no seed replacement or episode filtering. It is not the last training-window reward proxy.
