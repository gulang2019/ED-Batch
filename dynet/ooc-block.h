#pragma once

#include "dynet/expr.h"
#include "dynet.h"
#include "utils.h"
#include "globals.h"
#include <iostream>

namespace OoC
{
    struct gather_t{
        size_t dst_offset;
        /**
         * 0: not contiguous, need runtime memory combine;
         * 1: contiguous;
         * 2: ont contiguous, but is ready at compile time;
         */
        int contig;
        std::vector<size_t> src_ids;
        std::vector<size_t> lens;
    }; 
    
    struct BatchInfo
    {
        BatchInfo() {}
        dynet::Tensor nfx;
        std::vector<gather_t> gathers;
        std::vector<dynet::VariableIndex> ids;
        std::vector<int> concat;
        dynet::Node *pseudo_node = nullptr;
        std::vector<const dynet::Tensor *> arg_nfxs;
    };
    struct Block : public dynet::ComputationGraph
    {
        // default constructor
        Block(): dynet::ComputationGraph(nullptr),
        name("block"+std::to_string(names.size())), opt(dynet::block_opt_flag) {
            if (names.count(name) == 0) {
                names[name] = names.size();
            }
            else std::cerr << "[WARNING] define a block with duplicated name " << name << std::endl;
            id = names[name];
        }
        Block(std::string name): dynet::ComputationGraph(nullptr), 
        name(name), opt(dynet::block_opt_flag) {
            if (names.count(name) == 0) {
                names[name] = names.size();
            }
            else std::cerr << "[WARNING] define a block with duplicated name " << name << std::endl;
            id = names[name];
        }
        // get input from an dynet::Expression
        dynet::Expression placeholder(const dynet::Dim &d, std::string name);
        // finish params
        void finish_params() { n_params = nodes.size(); }
        // finish declaration
        void finish_input() { n_input = nodes.size(); }
        // get input from an lookup
        dynet::Expression lookup(dynet::LookupParameter p);
        // pickneglogsoftmax 
        dynet::Expression pickneglogsoftmax(const dynet::Expression&x);
        // register output operation
        void output(const std::vector<dynet::Expression>& exprs);
        // complete definition
        void freeze();
        // batched forward
        void forward(
            const std::vector<const dynet::Tensor *> &xs,
            std::vector<dynet::Tensor*>& ys,
            const std::vector<std::vector<unsigned>> &lookup_indices,
            int batch_size);
        dynet::Expression operator()(
            dynet::ComputationGraph *cg,
            std::unordered_map<std::string, dynet::Expression> expr_inputs,
            std::initializer_list<unsigned> lookup_index);

        std::string as_string(bool verbose = false);


        float* base = nullptr;
        size_t tot_mem;
        std::vector<size_t> offsets;
        std::unordered_map<dynet::SigMap*, int> sig_cache;
        int autobatch_sig(dynet::SigMap& sig);
        void memory_allocate_opt(int batch_size);
        void execute_opt(int bid, int batch_size);
        void combine_tensors_opt(const std::vector<size_t>& src_ids, const std::vector<size_t>& lens, const dynet::Tensor* tout, int batch_size);
        bool aot_analysed = false;
        void aot_analysis(); 

        // the identifier for type
        int id;
        // <nid, bid> pair of output node
        int n_input = -1;
        int n_params = -1;
        struct output_t{
            dynet::Dim dim;
            int nid;
            int bid;
            int idx; // the index in user given indices
        };
        std::vector<output_t> output_nodes;
        std::unordered_map<int, int> output_indices;
        unsigned one_batch_size;
        std::vector<std::pair<int, std::string> > input_nodes;
        // nodes take runtime parameters as input;
        std::vector<std::pair<int, dynet::nt::NodeType> > runtime_nodes;
        std::vector<int> autobatch_concat;

        static std::unordered_map<std::string, int> names;
        std::string name;
        bool freezed = false;

        std::vector<BatchInfo> batches;
        // vector<bid>: the order for memory allocation
        std::vector<int> memory_allocation_order;
        // the unique snode id;
        // the relative memory offset in a batch
        std::vector<dynet::Tensor> nfx_cache;
        std::vector<int> node2batch;

        std::vector<int> node2offset;

        const dynet::Tensor &get_nfx(dynet::VariableIndex i);
        bool guaranteed_contig;
        void memory_allocate(int bid);
        void execute(int bid);
        void reset();
        void combine_tensors(const std::vector<dynet::VariableIndex> &batch_ids,
                             int aid, dynet::Tensor &tout);
        // check for number of memory combine
        int self_check();

        ~Block();

        // memory for shared properties among block
        int memo_block_type = -1;
        int memo_get_type = -1;

        /**
         * 0: dynet batching & mem allocation
         * 1: PQTree batching & mem allocation 
         * 2: dynet batching & mem allocation & aot analysis
         * 3: PQTree batching & mem allocation & aot analysis
         */ 
        int opt; // whether optimization like PQTree is on.
    };
} // namespace OoC