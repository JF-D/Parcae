# Replay Trace on AWS

## Trace Generation

We can generate trace for replay with the following command using pre-optimized liveput optimization result:
```
python tools/trace_gen.py -n 32 -trace log/opt-res.json -output-trace trace_lh_80_90.txt -start-hour 8.0 -end-hour 9.0
```
We provided some pre-computed results for 32 `p3.2xlarge` instances on AWS in the folder `aws/traces/liveput_res`.

## Replay Trace
1. Set `hostname` file with all the running instances.
2. Run `set_ssh_access.sh` to set up the ssh access to all the instances.
3. Set scripts and model you want to run in L20-24 of `replay_trace.py`. We provide an example using megatron in `examples`.
4. Run with the following command. We need to specify the number of nodes (`32`) and the trace (`trace.txt`) that we want to use.
```
sh replay_trace_run.sh 32 trace.txt
```
