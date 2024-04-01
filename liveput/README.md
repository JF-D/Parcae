# Liveput Optimizer

## Preemption Sampling

Before run liveput optimization, we need to first run preemption sampler to estimate the migration cost.

```
cd liveput
mkdir log
python main.py --trace traces/p3trace.csv --model gpt-1.5b --train-batch-size 128 --n-sim 10000 --mc-sample --mc-sample-look-aheads 8 10 12 --start-hour 0 --end-hour 12 --restart-cost 30 --update-cache
# We will use profile cost without setting `--restart-cost`
python main.py --trace traces/p3trace.csv --model gpt-1.5b --train-batch-size 128 --n-sim 10000 --mc-sample --mc-sample-look-aheads 8 10 12 --start-hour 0 --end-hour 12 --update-cache
```

The preemption sampling will take a long time. We also provide some pre-computed results for 32 `p3.2xlarge` instances on AWS:
Model|Batch Size|Preemption Sampling Result
---|---|---
GPT-1.5B|128|[link](https://drive.google.com/file/d/11R8BGXmansIoUBtrtw_tl0J6aar5aiMl/view?usp=sharing)

```
mkdir liveput/livestore
cp pre-computed-results.json liveput/livestore/
```

## Liveput Optimization

Afterwards, we can run liveput optimization with the following command. We can change `--look-ahead` value to vary the number of future intervals.

```
cd liveput
mkdir log
python main.py --trace traces/p3trace.csv --model gpt-1.5b --train-batch-size 128 --restart-cost 30 --look-ahead 8
# We will use profile cost without setting `--restart-cost`
python main.py --trace traces/p3trace.csv --model gpt-1.5b --train-batch-size 128 --look-ahead 8
```

#### Sample for all models
Since there are many models and restart cost, you can use the following command to run all the liveput simulations in parallel. Finally, it will generate a table in markdown format. `--liveput-simulation-look-aheads 8 10 12` means we want to do liveput simulation for 8, 10 and 12 look aheads.
```
cd liveput
mkdir log
python main.py --trace traces/p3trace.csv --liveput-simulation --liveput-simulation-look-aheads 8 10 12
```
In function `run_liveput_simulation` of `main.py`, you can modify the following parts to specify the model and restart cost you want to simulate.
```
    models_to_bs = {
        'gpt-1.5b': 256,
        'gpt-2.7b': 256,
        'gpt-7.5b': 512,
        'resnet50': 2048,
    }
    all_restart_costs = [17, 30, 45, 60, 90, 120, 150, 180, 210, 240, 270, 300]
```

## Generate Liveput Optimization Result
After running liveput optimization, we will get 2 outputs: `log/test-xxx.pdf` and `log/test-xxx.json`. The `test-xxx.json` records the optimization result.
We can generate a liveput optimization result with this command:
```
python utils/graph.py --hist log/test-xxx.json --out-trace log/opt-res.json
```
The `log/opt-res.json` is the generated file. Afterwards, we can generate liveput trace events with this file.
