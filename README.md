# Parcae
This is the artifact repository for our NSDI '24 paper "Parcae: Proactive, Liveput-Optimized DNN Training on Preemptible Instances". [[NSDI '24](https://www.usenix.org/conference/nsdi24/presentation/duan)], [[arXiv](https://arxiv.org/abs/2403.14097)]


Parcae is a system that enables cheap, fast, and
scalable DNN training on preemptible instances by proac-
tively adjusting the parallelization strategy of a DNN training
job to adapt to predicted resource changes before instance pre-
emptions and allocations really happen, which significantly
reduces the cost of handling these events. Parcae optimizes
liveput, a novel metric that measures the expected training
throughput of a DNN job under various possible preemp-
tion scenarios. Compared to existing reactive, throughput-
optimized systems, Parcae’s proactive, live-optimized solution
considers both the throughput of a job and its robustness under
preemptions. To optimize liveput, Parcae supports lightweight
instance migration and uses an availability predictor to fore-
cast future preemptions. It then uses a liveput optimizer to
discover an optimal strategy to parallelize DNN training un-
der predicted preemptions. We evaluate Parcae on a variety
of DNNs and preemption traces and show that Parcae outper-
forms existing spot-instance DNN training systems by up to
10×. More importantly, Parcae achieves near-optimal perfor-
mance for training large DNNs under frequent preemptions,
in which case existing approaches cannot make any progress.

## Requirements
- [etcd v3.4.3](https://github.com/etcd-io/etcd/releases/v3.4.3/)
- PyTorch >= 1.8
- CUDA
- [Apex](https://github.com/NVIDIA/apex)

Our tested version is PyTorch 1.11 and CUDA 11.3.

## Installation
The installation of Parcae is the same as installing DeepSpeed. You can also refer to the [DeepSpeed documentation](README_DeepSpeed.md) for detailed instructions.
```
pip install -r requirements.txt
DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 DS_BUILD_UTILS=1 pip install .
```

## Getting Started
Parcae is evaluated by replaying the trace on on-demand instacnes. Try Parcae with the following steps:
- [Liveput Optimizer](liveput/README.md)
- [AWS On-Demand Replay](aws/README.md)


## Acknowledgement

Parcae is built based on [DeepSpeed](https://github.com/microsoft/DeepSpeed). We also learned a lot from [Bamboo](https://github.com/uclasystem/bamboo) (thanks John and Pengzhan) and [TorchElastic](https://github.com/pytorch/elastic).

## Citation

```
@inproceedings{nsdi24parcae,
  author = {Jiangfei Duan and Ziang Song and Xupeng Miao and Xiaoli Xi and Dahua Lin and Harry Xu and Minjia Zhang and Zhihao Jia},
  title = {Parcae: Proactive, {Liveput-Optimized} {DNN} Training on Preemptible Instances},
  booktitle = {21st USENIX Symposium on Networked Systems Design and Implementation (NSDI 24)},
  year = {2024},
  address = {Santa Clara, CA},
  url = {https://www.usenix.org/conference/nsdi24/presentation/duan},
  publisher = {USENIX Association},
  month = apr
}
```
