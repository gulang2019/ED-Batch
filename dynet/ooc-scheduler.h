#ifndef OOC_SCHEDULER_H
#define OOC_SCHEDULER_H

#include <set>
#include <vector>
#include <unordered_map>
#include <deque>
#include <random>

#include "dynet/dynet.h"
#include "dynet/exec.h"
#include "utils.h"

using dynet::BatchedExecutionEngine;
using dynet::ComputationGraph;
using dynet::SigMap;
using dynet::VariableIndex;
using std::vector;
using std::queue;

#define NITER (1000)

namespace OoC{
    struct node_t
    {
        int type;
        vector<VariableIndex> args;
        bool invalid = true;
        int succ_cnt = 0;
        vector<VariableIndex> typewise_args;
        int typewise_succ_cnt = 0;
    };

    struct type_t
    {
        vector<VariableIndex> frontiers;
        int typewise_frontier_cnt = 0;
        int min_nid = -1;
    };

    namespace RL {
        typedef std::set<int, std::less<int>> state_t;
        struct q_table_t
        {
            std::unordered_map<int, double> q;
            void init(const state_t &state, double init_value = 0.0)
            {
                for (auto s : state)
                    q[s] = init_value;
            }
            void init(const vector<int>& state, double init_value = 0.0){
                for (auto s : state)
                    q[s] = init_value;
            }
            size_t size() { return q.size(); }
            double max()
            {
                double ret = 0;
                for (auto &kv : q)
                    ret = std::max(ret, kv.second);
                return ret;
            }
            int argmax()
            {
                double v = -1e9;
                int arg = 0;
                for (auto &kv : q)
                {
                    if (kv.second > v)
                        v = kv.second, arg = kv.first;
                }
                return arg;
            }
            inline int action() {
                if (_action < 0)
                    _action = argmax();
                return _action;                
            }
            int _action = -1;
        };
    }
}

namespace OoC
{
    class BaseScheduler
    {
    public:
        /**
         * \return 0: done; 1: okay, -1: continue
         */
        virtual int schedule(vector<VariableIndex> &batch, int verbose = 0) = 0;
        virtual void init(ComputationGraph *cg, VariableIndex num_nodes_evaluated, VariableIndex upto) = 0;
        virtual void post_process() {}
        int lower_bound();

    protected:
        vector<int> node2type;
        vector<vector<int>> node2succs;
        static SigMap sigmap;
        void basic_init();

        vector<vector<int>> node2succs_typewise;
        void typewise_init();

        ComputationGraph *cg;
        VariableIndex num_nodes_evaluated, upto;
    };

    class AgendaScheduler : public BaseScheduler
    {
    public:
        int schedule(vector<VariableIndex> &batch, int verbose) override;
        void init(ComputationGraph *cg, VariableIndex num_nodes_evaluated, VariableIndex upto) override;

    private:
        vector<int> node2depth;
        vector<int> type_cnt;
        vector<double> type2depth;
        queue<int> get_batches;
        vector<int> arity;
        vector<vector<VariableIndex>> type2frontiers;
    };

    class RLScheduler : public BaseScheduler
    {
    public:
        int schedule(vector<VariableIndex> &batch, int verbose) override;
        void init(ComputationGraph *cg, VariableIndex num_nodes_evaluated, VariableIndex upto) override;
        void post_process() override;

    private:
        int schedule_impl(vector<VariableIndex>& batch, bool train, int verbose);
        inline int take_action(const vector<int>& state, bool train);
        static std::unordered_map<int, TupleDict<RL::q_table_t> > q_tables;
        vector<vector<VariableIndex> > type2frontiers;
        vector<int> typewise_frontier_cnt;
        vector<int> arity, typewise_arity;
        static std::unordered_set<int> trained;
        void train();
        int inference();
        struct log_t
        {
            vector<int> state;
            int action;
            double reward;
        };
        vector<log_t> logs;
        queue<int> get_batches;
 
        int n_iter = NITER;
        double alpha = 0.4;
        double gamma = 1.0;
        double eps_max = 0.5;
        double eps_min = 0.05;
        double eps_decay = 0.9;
        int td_step = 10;
        double eps;

        // for debug;
        int n_hit, n_batch;
    };
} // namespace OoC

#endif