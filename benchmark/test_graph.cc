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
#include "models/treenn.hpp"
#include "models/lattice.hpp"
#include "models/lstm.hpp"
#include "models/bilstm-tagger.hpp"
#include "models/rnn_inference.hpp"
#include "utils.h"

using namespace std;
using namespace dynet;
using namespace OoCTest;

#define REPEAT (100)

string store_dir = "tmp/";

bool assert_allclose(const std::vector<float>& a, const std::vector<float>& b){
    if (a.size() != b.size()) return false;
    for (int i = 0; i < (int)a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > 1e-2) return false;
    } 
    return true;
}

void load_graph(
    ParameterCollection& model,
    DataLoader& data_loader, 
    std::string workload, 
    int batch_size, 
    unsigned embed_size, 
    unsigned hidden_size,  
    unsigned vocab_size, 
    int layers) {
    vector<int> opts = {0,2,5};
    if (workload == "lattice_lstm"){
        for (auto opt: opts) {
            dynet::block_opt_flag = opt;
            data_loader.add_model(workload + "-" + to_string(opt), new Lattice(model, hidden_size, Lattice::LSTM));
        }
    }
    else if (workload == "lattice_gru"){
        for (auto opt: opts) {
            dynet::block_opt_flag = opt;
            data_loader.add_model(workload + "-" + to_string(opt), new Lattice(model, hidden_size, Lattice::GRU));
        }
    }
    else if(workload == "tree_lstm"){
        for (auto opt: opts) {
            dynet::block_opt_flag = opt;
            data_loader.add_model(workload + "-" + to_string(opt), new Treenn(model, hidden_size, embed_size, Treenn::NORMAL));
        }
    }
    else if(workload == "tree_gru"){
        for (auto opt: opts) {
            dynet::block_opt_flag = opt;
            data_loader.add_model(workload + "-" + to_string(opt), new Treenn(model, hidden_size, embed_size, Treenn::GRU));
        }
    }
    else if(workload == "double_typed_treenn"){
        for (auto opt: opts) {
            dynet::block_opt_flag = opt;
            data_loader.add_model(workload + "-" + to_string(opt), new Treenn(model, hidden_size, embed_size, Treenn::DOUBLETYPE));
        }
    }
    else if(workload == "perfect_treenn"){
        for (auto opt: opts) {
            dynet::block_opt_flag = opt;
            data_loader.add_model(workload + "-" + to_string(opt), new Treenn(model, hidden_size, embed_size, Treenn::PERFECT, 5, 8));
        }
    }
    else if(workload == "gatednn"){
        for (auto opt: opts) {
            dynet::block_opt_flag = opt;
            data_loader.add_model(workload + "-" + to_string(opt), new Treenn(model, hidden_size, embed_size, Treenn::GRID, 10, 15));
        }
    }
    else if(workload == "lstm_1d"){
        for (auto opt: opts) {
            dynet::block_opt_flag = opt;
            data_loader.add_model(workload + "-" + to_string(opt), new NdLSTM(model, embed_size, hidden_size, vocab_size, {{10,50}}));
        }
    }
    else if(workload == "lstm_2d"){
        for (auto opt: opts) {
            dynet::block_opt_flag = opt;
            data_loader.add_model(workload + "-" + to_string(opt), new NdLSTM(model, embed_size, hidden_size, vocab_size, {{3,5}, {20,25}}));
        }
    }
    else if(workload == "lstm_3d"){
        for (auto opt: opts) {
            dynet::block_opt_flag = opt;
            data_loader.add_model(workload + "-" + to_string(opt), new NdLSTM(model, embed_size, hidden_size, vocab_size, {{2,4}, {2,4}, {10,15}}));
        }
    }
    else if(workload == "bilstm_tagger"){
        for (auto opt: opts) {
            dynet::block_opt_flag = opt;
            data_loader.add_model(workload + "-" + to_string(opt), new BilstmTagger(model, layers, embed_size, embed_size, hidden_size, hidden_size));
        }
    }
    else if(workload == "bilstm_tagger_withchar"){
        for (auto opt: opts) {
            dynet::block_opt_flag = opt;
            data_loader.add_model(workload + "-" + to_string(opt), new BilstmTagger(model, layers, embed_size, embed_size, hidden_size, hidden_size, true));
        }
    }
    else if (workload == "dag"){
        for (auto opt: opts) {
            dynet::block_opt_flag = opt;
            data_loader.add_model(workload + "-" + to_string(opt), new RandomDAG(model, hidden_size, vocab_size, 200, 2, 3, 1000, 4, 30));
        }
    }
    else if (workload == "mvrnn"){
        for (auto opt: opts) {
            dynet::block_opt_flag = opt;
            data_loader.add_model(workload + "-" + to_string(opt), new MVRNN(model, hidden_size));
        }
    }
    else if (workload == "lstm_nmt") {
        for (auto opt: opts) {
            data_loader.add_model(workload + "-" + to_string(opt), new LSTMNMT(model, hidden_size, embed_size));
        }
    }
    else {
        cerr << "[ERROR]:" << workload << " not found." << endl;
        throw runtime_error("bad workload");
    }
}

struct Config{
    string alg;
    /* Whether or not using dynet's dynamic batching algorithm
        0 for not use;
        1 for agenda-based algorithm;
        2 for depth-based algorithm. 
    */ 
    int dynet_autobatch; 

    /* Whether or not using ooc's dynamic batching algorithm: 
        0 for not use; 
        1 for using dynet-agenda; 
        2 for RL+ Base state representation; 
        3 for RL + max state representation; 
    */
    int ooc_autobatch; 
    
    /** Whether to enable the block optimization  
     * 0[default]: dynet mem-alloc
     * 1: PQTree mem-alloc 
     * 2: dynet mem-alloc & aot analysis
     * 3: PQTree mem-alloc & aot analysis
     * 4: dynet mem-alloc & aot analysis & pre-mem allocation
     * 5: PQTree mem-alloc & aot analysis & pre-mem allocation
     */ 
    int block_opt_flag;   

    int blocked = 1; // whether subgraph is regarded as a single node in the graph.  
};

void test_end2end(std::string workload, std::string device, int batch_size, int hidden_size){
    int layers = 2;
    unsigned embed_size = hidden_size;
    unsigned vocab_size = 1024;
    DataLoader data_loader;
    ParameterCollection model;
    load_graph(model, data_loader, workload, batch_size, embed_size, hidden_size, vocab_size, layers);

    // train all RL models by one run. 
    vector<float> ground_truth;
    for (int x = 2; x <= 4; x++)
    {
        dynet::autobatch_flag = 0;
        ooc_autobatch_flag = x;
        ComputationGraph cg;
        data_loader.reset();
        data_loader.build_graph(cg, {{workload + "-5", batch_size}});
        VariableIndex last_idx = cg.nodes.size() - 1;
        Tensor t = cg.forward(last_idx);
        ground_truth = as_vector(t);
    }

    global_timer.scope("warm_up");
    for (int i = 0; i < REPEAT; i++)
    {
        dynet::autobatch_flag = 1;
        ooc_autobatch_flag = 0;
        ComputationGraph cg;
        data_loader.build_graph(cg, {{workload + "-5", batch_size}});
        VariableIndex last_idx = cg.nodes.size() - 1;
        global_timer.cumint("lower_bound", cg.dynamic_batching_lowerbound());
        Tensor t = cg.forward(last_idx);
    }

    // algorithm-cpu/gpu parallel-block-opt
    // algorithm {rl_sorted, rl_max_type, rl_normal, dynet_agenda, dynet_depth, typewiseLB}
    // cpu/gpu parallel 
    // paralell-block-opt {0,5} odd PQTree, 0-1 not optimized, 2-3, not optimized param, 4-5  optimized param
    // 
    vector<Config> configs = {
        {"dynet_baseline_agenda", 1, 0, 0, 0}, // an unoptimized version of dynet
        {"dynet_baseline_depth", 2, 0, 0, 0}, // an unoptimized version of dynet
        {"rl_sorted-opt", 0, 3, 5}, 
        {"rl_max_type-opt", 0, 4, 5},
        {"rl_normal-opt", 0, 2, 5}, 
        {"dynet_agenda", 1, 0, 0}, 
        {"dynet_depth", 2, 0, 0}, 
        // {"cpu_gpu_para", 0, 1, 5},
        {"typewiseLB", 8, 0, 0},    
        // {"dynet_agenda-opt", 1, 0, 2}, 
        // {"dynet_depth-opt", 2, 0, 2}, 
        // {"dynet_agenda-opt", 1, 0, 5}, 
        // {"dynet_depth-opt", 2, 0, 5}, 
        // {"rl-backward", 11, 0, 0}, 
        // {"cpu_gpu_para", 0, 1, 0}, 
        // {"rl_normal", 0, 2, 0}, 
        // {"rl_sorted", 0, 3, 0}, 
        // {"rl_max_type", 0, 4, 0},
        };
    
    for (auto& config: configs){
        dynet::autobatch_flag = config.dynet_autobatch;
        dynet::ooc_autobatch_flag = config.ooc_autobatch;
        dynet::block_opt_flag = config.block_opt_flag;
        dynet::blocked = config.blocked;
        global_timer.scope(config.alg);
        data_loader.reset();
        global_timer.start("total");
        for (int r = 0; r < REPEAT; r++){
            global_timer.start("construction");
            ComputationGraph cg;
            data_loader.build_graph(cg, {{workload + "-" + to_string(dynet::block_opt_flag), batch_size}});
            global_timer.stop("construction");
            // cg.show_nodes();
            VariableIndex last_idx = cg.nodes.size() - 1;
            global_timer.start("forward");
            Tensor t = cg.forward(last_idx);
            vector<float> result = as_vector(t);
            global_timer.stop("forward");
        }
        global_timer.stop("total");
        // assert_allclose(result, ground_truth);
    }

    vector<string> scopes = {"warm_up"};
    for (auto& config: configs) scopes.push_back(config.alg);
    global_timer.show("test-" + workload, scopes, REPEAT);
    global_timer.save(store_dir + workload + ".csv");
    global_timer.clearall();

    cout << "---------------test end2end passed!---------------" << endl;
}

void test_compile_time(std::string workload, int batch_size) {
    int layers = 2;
    int hidden_size = 128;
    unsigned embed_size = hidden_size;
    unsigned vocab_size = 1024;
    DataLoader data_loader;
    ParameterCollection model;
    load_graph(model, data_loader, workload, batch_size, embed_size, hidden_size, vocab_size, layers);

    vector<float> ground_truth;
    global_timer.scope("compile");
    {
        global_timer.start("time");
        dynet::autobatch_flag = 0;
        ooc_autobatch_flag = 3;
        ComputationGraph cg;
        data_loader.reset();
        data_loader.build_graph(cg, {{workload + "-5", batch_size}});
        VariableIndex last_idx = cg.nodes.size() - 1;
        Tensor t = cg.forward(last_idx);
        ground_truth = as_vector(t);
        global_timer.stop("time");
    }

    global_timer.scope("dynet"); // the choice made by depth-based algorithm
    {
        ooc_autobatch_flag = 1;
        ComputationGraph cg;
        data_loader.reset();
        data_loader.build_graph(cg, {{workload + "-5", batch_size}});
        VariableIndex last_idx = cg.nodes.size() - 1;
        Tensor t = cg.forward(last_idx);
        ground_truth = as_vector(t);
    }

    global_timer.show("test-" + workload, {"compile", "dynet"});
    global_timer.save(store_dir + workload + ".csv");
    global_timer.clearall();

    cout << "---------------test compile time passed!---------------" << endl;
}

int main(int argc, char **argv)
{
    int batch_size = 32;
    int model_size = 128;
    string device = "none";

    int mode = 0;
     
    dynet::initialize(argc, argv);
    std::string workload = string(argv[1]);
    if (argc > 2)
        device = string(argv[2]);
    if (argc > 3)
        batch_size = std::stoi(argv[3]);
    if (argc > 4) 
        model_size = std::stoi(argv[4]);
    if (argc > 5) 
        mode = std::stoi(argv[5]);
    if (argc > 6)
        store_dir = std::string(argv[6]);
    cout << "running " << workload << " on " << device << 
        ", batch_size " << batch_size << ", model_size " 
        << model_size << ",mode" << mode << ",store_dir" << store_dir << endl;
    system(("mkdir -p " + store_dir).c_str());
    if (mode == 5) { // for compile time test
        store_dir += "/profile/";
        system(("mkdir -p " + store_dir).c_str());
    }
    else {
        store_dir += to_string(batch_size) + "-" + to_string(model_size) + "/";
        system(("mkdir -p " + store_dir).c_str());
        store_dir += device + "/";
        system(("mkdir -p " + store_dir).c_str());
    }


    if (mode == 0)
        test_end2end(workload, device, batch_size, model_size);
    else if(mode == 1) 
        test_compile_time(workload, batch_size);
        
    return 0;
}
