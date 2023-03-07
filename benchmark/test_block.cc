#include <vector>
#include <stdexcept>
#include <fstream>
#include <chrono>
#include <iostream>

#include <dynet/training.h>
#include <dynet/expr.h>
#include <dynet/dict.h>
#include <dynet/lstm.h>
#include <dynet/param-init.h>
#include <dynet/globals.h>

#include "models/model.hpp"
#include "models/block.hpp"
#include "utils.h"

using namespace std;
using namespace dynet;
using namespace OoCTest;

string root_dir = "1_1";
string store_dir = "";

const int REPEAT = 100;

void load_graph(
    DataLoader &data_loader,
    std::string workload,
    unsigned embed_size,
    unsigned hidden_size,
    unsigned vocab_size,
    int layers)
{
    global_timer.scope("compile");
    const vector<int> opts = {2,5};
    for (auto i: opts){
        dynet::block_opt_flag = i;
        global_timer.start(workload + "-" + to_string(i));
        if (workload == "tree_lstm_internal"){
            data_loader.add_model(workload + "-" + to_string(i), new TreennInternal(hidden_size, embed_size));
        }
        else if (workload == "tree_lstm_leaf"){
            data_loader.add_model(workload + "-" + to_string(i), new TreennLeaf(hidden_size, embed_size, vocab_size));
        }
        else if (workload == "lstm"){
            data_loader.add_model(workload + "-" + to_string(i), new LSTMCell(hidden_size, embed_size, vocab_size));
        }
        else if (workload == "gru") {
            data_loader.add_model(workload + "-" + to_string(i), new GRUCell(hidden_size));
        }
        else if (workload == "tree_gru_internal") {
            data_loader.add_model(workload + "-" + to_string(i), new NChildGRU(hidden_size, 2));
        }
        else if (workload == "tree_gru_leaf") {
            data_loader.add_model(workload + "-" + to_string(i), new NChildGRU(hidden_size, 1));
        }
        else if (workload == "mvrnn") {
            data_loader.add_model(workload + "-" + to_string(i), new MVCell(hidden_size));
        }
        else{
            throw runtime_error("bad workload");
        }
        global_timer.stop(workload + "-" + to_string(i));
    }
}

void test_block(std::string workload, std::string device, int batch_size, unsigned model_size)
{
    unsigned embed_size = model_size;
    unsigned hidden_size = model_size;
    unsigned vocab_size = 1024;
    int layers = 4;
    DataLoader data_loader;

    load_graph(data_loader, workload, embed_size, hidden_size, vocab_size, layers);

    dynet::autobatch_flag = 1;

    // warm up
    for (int i = 0; i < REPEAT; i++)
    {
        ComputationGraph cg;
        data_loader.reset();
        data_loader.build_graph(cg, {{workload + "-2", batch_size}});
        VariableIndex last_idx = cg.nodes.size() - 1;
        Tensor t = cg.forward(last_idx);
    }
    // validity check
    // for (auto& config: configs){
    //     dynet::autobatch_flag = config.second;
    //     for (auto pre_malloc: {false, true}){
    //         do_pre_malloc = pre_malloc;
    //         data_loader.reset();
    //         ComputationGraph cg;
    //         data_loader.build_graph(cg, {{workload, batch_size}});
    //         VariableIndex last_idx = cg.nodes.size() - 1;
    //         Tensor t = cg.forward(last_idx);
    //         vector<float> result = as_vector(t);
    //         assert_allclose(result, ground_truth);
    //     }
    // }
    

    for (int opt : {2,5})
    {
        dynet::block_opt_flag = opt;
        data_loader.reset();
        string model_tag = workload + "-" + to_string(opt);
        global_timer.scope("opt-" + to_string(opt));
        for (int repeat = 0; repeat < REPEAT; repeat++)
        {
            global_timer.start("total");
            ComputationGraph cg;
            global_timer.start("construction");
            Expression res = data_loader.build_graph(cg, {{model_tag, batch_size}});
            global_timer.stop("construction");
            global_timer.start("forward");
            cg.forward(res);
            global_timer.stop("forward");
            global_timer.stop("total");
        }
    }

    global_timer.show(workload, {"opt-2-block", "opt-5-block"});
    global_timer.save(store_dir + workload + ".csv");

    global_timer.clearall();

    cout << "---------------test_block passed!---------------" << endl;
}

void test_compile_time()
{
    unsigned embed_size = 128;
    unsigned hidden_size = 128;
    unsigned vocab_size = 1024;
    int layers = 4;
    DataLoader data_loader;

    vector<string> workloads = {"tree_lstm_internal", "tree_lstm_leaf", "lstm", "gru", "tree_gru_internal", "tree_gru_leaf", "mvrnn"};
    for (auto workload: workloads)
        load_graph(data_loader, workload, embed_size, hidden_size, vocab_size, layers);

    global_timer.show("compile time", {"compile"});
    global_timer.save(root_dir + "/block/profile.csv");

    global_timer.clearall();

    cout << "---------------profile finished!---------------" << endl;
}

int main(int argc, char **argv)
{

    dynet::initialize(argc, argv);
    std::string workload = string(argv[1]);
    std::string device = string(argv[2]);
    int batch_size = 32;
    unsigned model_size = 256;
    int mode = 0;
    if (argc > 3) batch_size = std::stoi(argv[3]);
    if (argc > 4) model_size = std::stoi(argv[4]);
    if (argc > 5) mode = std::stoi(argv[5]);
    if (argc > 6) root_dir = std::string(argv[6]);
    cout << "running " << workload << " on " << device 
        << ", batch_size=" << batch_size << ", model_size=" << model_size  << endl;
    vector<string> store_paths = {root_dir, "block", to_string(batch_size) + "-" + to_string(model_size), device};
    for (auto s: store_paths) {
        store_dir += s + "/";
        system(("mkdir -p " + store_dir).c_str());
    }
    cout << "save to " << store_dir << endl;

    if (mode == 0)
        test_block(workload, device, batch_size, model_size);
    else if (mode == 1) 
        test_compile_time();
    return 0;
}

/*
block X metric [n_combine, metric, latency] X alg (2,5)
*/
