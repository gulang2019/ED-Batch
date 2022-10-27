#pragma once

#include "dynet/expr.h"
#include "dynet.h"
#include "utils.h"
#include <iostream>

namespace OoC
{
    class SuperNode
    {
    public:
        typedef std::function<void(
            const std::vector<dynet::Expression> &input,
            const std::vector<int> &const_params,
            const std::vector<int> &runtime_params,
            std::vector<dynet::Expression> &output)>
            func_t;
        SuperNode(func_t f, const std::string &name) : _func(f), _name(name) {}
        std::vector<dynet::Expression> operator()(
            const std::vector<dynet::Expression> &input,
            const std::vector<int> &const_params,
            const std::vector<int> &runtime_params,
            bool mark_basic_block = false);
        void register_lookup(dynet::LookupParameter *lookup_param, int nid, int param_id)
        {
            lookup_args.push_back({lookup_param, nid, param_id});
        }
        static int new_graph(dynet::ComputationGraph *cg)
        {
            assert(results.empty());
            _cg = cg;
            n_node = 0;
            n_sync = 0;
            return 0;
        }
        static void synchronize(std::string info = "");
        static int n_sync;
        static int n_node;

    private:
        static dynet::ComputationGraph *_cg;
        static std::vector<std::future<void *>> results;
        struct log_t
        {
            int first_time = true;
            dynet::VariableIndex begin, end;
            std::unordered_map<dynet::VariableIndex, int> nid2aid;
            std::vector<int> output_indices;
            int _stid;
        };
        TupleDict<log_t> params_dict;
        func_t _func;
        std::string _name;
        struct lookup_arg_t
        {
            dynet::LookupParameter *p;
            int nid;
            int param_id;
        };
        std::vector<lookup_arg_t> lookup_args;
    };
    struct BatchInfo
    {
        dynet::Tensor nfx;
        std::vector<dynet::VariableIndex> ids;
        std::vector<int> concat;
        dynet::Node *pseudo_node;
        std::vector<const dynet::Tensor *> arg_nfxs;
        ~BatchInfo() {}
    };
    struct Block : public dynet::ComputationGraph
    {
        // default constructor
        Block(): dynet::ComputationGraph(nullptr), id(block_id), 
        name("block"+std::to_string(block_id)) {
            block_id ++;
        }
        Block(std::string name): dynet::ComputationGraph(nullptr), id(block_id++), 
        name(name) {}
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
        void output(std::initializer_list<dynet::Expression> exprs);
        // complete definition
        void freeze();
        // batched forward
        void forward(
            const std::vector<const dynet::Tensor *> &xs,
            dynet::Tensor &fx,
            const std::vector<std::vector<unsigned>> &lookup_indices,
            int batch_size);
        dynet::Expression operator()(
            dynet::ComputationGraph *cg,
            std::unordered_map<std::string, dynet::Expression> expr_inputs,
            std::initializer_list<unsigned> lookup_index);

        std::string as_string(bool verbose = false);

        int id;
        // <nid, bid> pair of output node
        int n_input = -1;
        int n_params = -1;
        std::vector<std::pair<int, int>> output_nodes;
        std::vector<dynet::Dim> output_dims;
        unsigned one_batch_size;
        std::vector<std::pair<int, std::string> > input_nodes;
        // nodes take runtime parameters as input;
        std::vector<std::pair<int, dynet::nt::NodeType> > runtime_nodes;
        std::vector<int> autobatch_concat;

        static int block_id;
        std::string name;
        bool freezed = false;

        std::vector<BatchInfo> batches;
        std::vector<int> memory_allocation_order;
        // the unique snode id;
        // the relative memory offset in a batch
        std::vector<dynet::Tensor> nfx_cache;
        std::vector<int> node2batch;

        std::vector<int> node2offset;

        const dynet::Tensor &get_nfx(dynet::VariableIndex i);
        bool guaranteed_contig;
        void memory_allocate(BatchInfo &batch);
        void execute(BatchInfo &batch);
        void reset();
        void combine_tensors(const std::vector<dynet::VariableIndex> &batch_ids,
                             int aid, dynet::Tensor &tout);
        // check for number of memory combine
        int self_check();

        ~Block();

        // memory for shared properties among block
        int memo_block_type = -1;
        int memo_get_type = -1;
    };
} // namespace OoC