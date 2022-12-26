#ifndef OOC_EXEC_H
#define OOC_EXEC_H

#include "dynet/exec.h"
#include "dynet/ooc-scheduler.h"

using dynet::ComputationGraph;
using dynet::Tensor;
using dynet::VariableIndex;
using dynet::BatchInfo;
using dynet::SigMap;
using std::vector;

namespace OoC{

    class Executor: public dynet::ExecutionEngine {
    public:
        explicit Executor(ComputationGraph& cg):
         dynet::ExecutionEngine(cg){}
        ~Executor();
        const Tensor& forward() override;
        const Tensor& forward(VariableIndex i) override;
        const Tensor& incremental_forward() override;
        const Tensor& incremental_forward(VariableIndex i) override;
        const Tensor& get_value(VariableIndex i) override;
        const Tensor& get_gradient(VariableIndex i) override;
        void backward(bool full) override;
        void backward(VariableIndex i, bool full) override;
        void invalidate() override;
        void invalidate(unsigned i) override;
        
    private:
        void init();
        void memory_allocation(VariableIndex bid);
        void execute(VariableIndex bid);
        static SigMap sigmap;
        vector<Tensor> nfx_cache;
        vector<dynet::BatchInfo> batches;
        AgendaScheduler agenda_scheduler;
        RLScheduler rl_scheduler;
        const Tensor& get_nfx(VariableIndex i);

        vector<size_t> node2offset, node2size;
        vector<VariableIndex> node2batch;

        // the temporary allocated memory
        struct MemoryBlock{
            float* base;
            size_t sz;
            bool avail;
        };
        vector<MemoryBlock> memory_blocks;
        void combine_tensors(const vector<VariableIndex>& batch_ids,
                       int aid, Tensor &tout);
        // for debug
        std::unordered_set<std::pair<int, int>, OoC::hash_pair> mem_transfer_edges;
    };
}

#endif 