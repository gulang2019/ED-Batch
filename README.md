# ED-Bacth 

This repo is the implementation for [ED-Batch: Efficient Automatic Batching of Dynamic Deep Neural Networks via Finite State Machine](https://arxiv.org/abs/2302.03851). `ED-Batch` is implemented as an runtime extension to DNN framework `dynet` to enable more efficient graph construction, dynamic batching, and memory arrangement. The code is split into two parts: 1. modification to dynet's src code to support state machine guided dynamic batching (dynet/), 2. tools to support static subgraph optimizations (src/). For legacy reason, `ED-Batch` uses the name `OoC` to separate from vanilla dynet. 

## Installation
1. install eigen 
```
hg clone https://bitbucket.org/eigen/eigen -r b2e267d
```

2. build ED-Batch's library 
```bash 
git clone git@github.com:gulang2019/ED-Batch.git 
cd ED-Batch
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. -DBACKEND=cuda -DEIGEN3_INCLUSE_DIR=/path/to/eigen [-DMKL_ROOT=/path/to/mkl] 
make -j4  && make install 
cd ..
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PWD}/lib 
```

3. run test scripts;

- customize the `CUDA_PATH` and `EIGEN_PATH` in `benchmark/Makefile` to your local system

- make & run scripts  
    ```bash
    cd benchmark 
    make 
    ./test_graph tree_lstm GPU 32 128 0 result --dynet-devices GPU:0
    ./test_block lstm GPU 32 128 0 result --dynet-devices GPU:0 --dynet-autobatch 1
    ```

    See scripts in benchmark/sh, or you can generate the test script by 
    ```bash 
    python ./gen_script.py --help 
    ```

## Instruction on code structure 
ED-Batch's runtime extension is implemented in dynet/. See `dynet/ooc-block.h` for static subgraph optimzization. See `dynet/ooc-executor.h` for the runtime driver. See `dynet/ooc-scheduler.h` for dynamic batching algorithms.  Apart from that, the static optimizations for static subgraph is implemented in `src/`. Class `OoC::PatternCache` in `src/OoC.h` implements the utilities to find best batching/memory allocation policy at compile time. See the PQ-tree stuff in `src/pq-trees`, as well as the test script `src/pq-trees/pqtest.cc`.  