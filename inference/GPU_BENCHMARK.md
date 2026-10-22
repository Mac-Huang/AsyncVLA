# GPU kernel timing

Run this command from the repository root in the existing AsyncVLA environment:

```bash
python inference/run_asyncvla_gpu.py --mode all --cpu-threads 1 \
  --repeats 100 --warmup 10 --duration 10 \
  --output-dir /workspace/results/asyncvla-gpu-run
```

The output directory must be empty. The command requires the Nsight Systems CLI
(`nsys`). It records CUDA activity and annotates every measured base and edge call.
CPU sampling is disabled.

Images, language tokens, and pose tensors are prepared on the GPU before the
measurement. Every measured call executes the vision encoders again. Base KV
caching remains disabled. The script checks that the prepared-input outputs match
the raw-input outputs exactly before measurement.

Isolated base, isolated edge, and concurrent execution use worker threads and
private CUDA streams. Each worker starts another call as soon as its previous
call and required output processing finish. The duration limits the concurrent
experiment; it does not set a call frequency. Concurrent edge calls consume the
latest completed base tokens.

The primary field, `gpu_kernel_busy_ms`, is the union of actual CUDA kernel
execution intervals for one component call. CUDA correlation IDs and the launch
thread associate each kernel with its model call.

This field excludes CPU preparation, tokenization, host dispatch, transfers,
memcpy/memset activities, synchronization, robot command conversion, and intervals
when none of that call's kernels execute. Kernel intervals still include GPU
memory stalls and resource contention. This field measures elapsed device
activity, not arithmetic instructions or SM occupancy.

`gpu_kernel_sum_ms` adds kernel durations and can double-count concurrent kernels.
`gpu_kernel_span_ms` and `gpu_event_span_ms` include gaps and must not be treated
as GPU busy time. Different components can overlap, so their busy times are not
additive. Tracing can perturb execution.

Outputs:

- `gpu_results.json`: GPU metrics, summaries, and per-call start/finish records.
- `dispatch.json`: wall-clock references and encoder execution counts.
- `gpu_trace.nsys-rep` and `gpu_trace.sqlite`: original trace evidence.
- `run.log` and `command.json`: run output and the exact profiling command.

The existing wall-clock benchmark and `run_asyncvla.py` remain available.
