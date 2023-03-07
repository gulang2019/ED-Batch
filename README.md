# ED-Bacth 

This repo is the implementation for [ED-Batch: Efficient Automatic Batching of Dynamic Deep Neural Networks via Finite State Machine](https://arxiv.org/abs/2302.03851). `ED-Batch` is implemented as an runtime extension to DNN framework `dynet` to enable more efficient graph construction, dynamic batching, and memory arrangement. The code is split into two parts: 1. modification to dynet's src code to support state machine guided dynamic batching (dynet/), 2. tools to support static subgraph optimizations (src/). For legacy reason, `ED-Batch` uses the name `OoC` to separate from vanilla dynet. 

## Installation
1. requirements
- eigen 3.4.0 
```
wget https://fossies.org/linux/privat/eigen-3.4.0.tar.bz2
tar xjvf eigen-3.4.0.tar.bz2
export EIGEN_BASE_DIR=${PWD}/eigen-3.4.0 
```
- cmake >= 3.12 

- gpu:
    - okay with V100, cuda 11.1, cudnn 8;
    - failed with A100, cuda 11.8, cudnn 8;

2. build ED-Batch's library 
```bash 
git clone git@github.com:gulang2019/ED-Batch.git 
cd ED-Batch
mkdir build && cd build
# build without gpu
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. -DEIGEN3_INCLUSE_DIR=/path/to/eigen [-DMKL_ROOT=/path/to/mkl] 
# build with gpu 
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.. -DBACKEND=cuda -DEIGEN3_INCLUSE_DIR=/path/to/eigen [-DMKL_ROOT=/path/to/mkl] 
make -j4  && make install 
cd ..
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PWD}/lib 
```

3. run test scripts;

- make & run scripts  
    ```bash
    cd benchmark 
    make 
    # gpu test
    ./test_graph tree_lstm GPU 32 128 0 result --dynet-devices GPU:0
    ./test_block lstm GPU 32 128 0 result --dynet-devices GPU:0 --dynet-autobatch 1
    # cpu test
    ./test_graph tree_lstm CPU 32 128 0 result --dynet-devices CPU
    ./test_block lstm CPU 32 128 0 result --dynet-devices CPU --dynet-autobatch 1
    ```

    See scripts in benchmark/sh, or you can generate the test script by 
    ```bash 
    python ./gen_script.py --help 
    ```

## Instruction on code structure 
ED-Batch's runtime extension is implemented in dynet/. See `dynet/ooc-block.h` for static subgraph optimzization. See `dynet/ooc-executor.h` for the runtime driver. See `dynet/ooc-scheduler.h` for dynamic batching algorithms.  Apart from that, the static optimizations for static subgraph is implemented in `src/`. Class `OoC::PatternCache` in `src/OoC.h` implements the utilities to find best batching/memory allocation policy at compile time. See the PQ-tree stuff in `src/pq-trees`, as well as the test script `src/pq-trees/pqtest.cc`.  