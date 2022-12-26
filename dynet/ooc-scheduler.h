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
        virtual int schedule(vector<VariableIndex> &batch) = 0;
        virtual void init(ComputationGraph *cg, VariableIndex upto) = 0;
        virtual void post_process() {}

    protected:
        vector<int> node2type;
        vector<vector<int>> node2succs;
        static SigMap sigmap;
        void basic_init();

        vector<vector<int>> node2succs_typewise;
        void typewise_init();

        ComputationGraph *cg;
        VariableIndex upto;
    };

    class AgendaScheduler : public BaseScheduler
    {
    public:
        int schedule(vector<VariableIndex> &batch) override;
        void init(ComputationGraph *cg, VariableIndex upto) override;

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
        int schedule(vector<VariableIndex> &batch) override;
        void init(ComputationGraph *cg, VariableIndex upto) override;
        void post_process() override;

    private:
        int schedule_impl(vector<VariableIndex>& batch, bool train);
        inline int take_action(const vector<int>& state, bool train);
        static std::unordered_map<int, TupleDict<RL::q_table_t> > q_tables;
        vector<vector<VariableIndex> > type2frontiers;
        vector<int> typewise_frontier_cnt;
        vector<int> arity, typewise_arity;
        static std::unordered_set<int> trained;
        void train();
        double inference();
        struct log_t
        {
            vector<int> state;
            int action;
            double reward;
        };
        vector<log_t> logs;
        queue<int> get_batches;
 
        int n_iter = 1000;
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

#define SCALE (20.0)
#define KERNEL_LAUNCH_OVERHEAD (10.0)

    namespace RL
    {

        struct Env
        {
            enum mode_t
            {
                BASIC,
                TYPEWISE
            } mode;

            Env(ComputationGraph &_cg, VariableIndex _upto, int _num_nodes_evaluated, mode_t mode, SigMap &sigmap);
            void set_mode(mode_t _mode) { mode = _mode; }
            bool done();
            void get_state(state_t &state);
            // info vector<[n_combine, n_scatter]>
            void get_memory_info(
                const vector<vector<VariableIndex>> &batches,
                const vector<bool> &pre_allocs,
                vector<std::pair<double, double>> &info);
            state_t incremental_get_state() { return state; }
            int get_largest_type();
            virtual void reset();
            virtual double step(int &type, vector<VariableIndex> &batch) = 0;
            OoC::TupleDict<std::set<int>> best_actions;
            SigMap &sigmap;

        protected:
            void init_basic();
            void init_typewise();
            void reset_basic();
            void reset_typewise();
            VariableIndex upto;
            int num_nodes_evaluated;
            ComputationGraph *cg;
            BatchedExecutionEngine *ee;
            vector<node_t> nodes;
            std::unordered_map<int, type_t> types;
            state_t state;
        };

        struct metric_t
        {
            double ex = 0;
            double ex2 = 0;
            int n = 0;
        };

        struct RLEnv : public Env
        {
            RLEnv(ComputationGraph &_cg, VariableIndex _upto, int _num_nodes_evaluated, mode_t mode, SigMap &sigmap) : Env(_cg, _upto, _num_nodes_evaluated, mode, sigmap) {}

            void reset() override;
            double step(int &type, vector<VariableIndex> &batch) override;

        private:
            double n_node; // number of node commited
            int n_step;    // number of step taken
            vector<metric_t> history;
            double get_reward();
            const double theta = 1.0;
        };

        struct buffer_t
        {
            state_t state;
            int action;
            double reward;
            state_t next_state;
        };

        struct ReplayBuffer
        {
            ReplayBuffer(size_t cap) : cap(cap), head(0), full(false)
            {
                buffer.resize(cap);
            }

            void add(const buffer_t &b)
            {
                buffer[head++] = b;
                if (head >= cap)
                {
                    full = true;
                    head = 0;
                }
            }

            buffer_t &sample()
            {
                if (full)
                    return buffer[random() % cap];
                return buffer[random() % head];
            }

            // return buffer[(head - idx) % cap]
            buffer_t &get(int idx)
            {
                return buffer[(head - idx + cap) % cap];
            }

            size_t cap;
            int head;
            bool full;
            vector<buffer_t> buffer;
        };

        struct DBLearner
        {
            DBLearner(int cap) : buffer(cap) {}

            OoC::TupleDict<RL::q_table_t> q_tables;

            int take_action(const state_t &state, bool train = false);
            void train(Env *env, int iter = 1000);
            int inference(Env *env);
            int typewise_inference(Env *env);

            ReplayBuffer buffer;

            bool trained = false;
            int type_ub;

            // hyper params
            double gamma = 0.90;
            double epsilon = 0.5;
            double epsilon_decay = 0.9;
            double epsilon_lb = 0.05;
            double alpha = 0.4;
            int n_step = 10;
            int n_replay = 0;
        };

        struct PreMallocLearner
        {
            PreMallocLearner(DBLearner &policy) : policy(policy) {}
            bool take_action(int state, bool train = false);
            void train(Env *env, int iter = 1000);
            double inference(Env *env);
            std::unordered_map<int, q_table_t> q_tables;
            DBLearner &policy;

            // hyper params
            double gamma = 1.0;
            double epsilon = 0.3;
            double epsilon_decay = 0.9;
            double epsilon_lb = 0.05;
            double alpha = 0.4;
            int n_step = 10;
            int n_replay = 0;
        };
    } // namespace RL
} // namespace OoC

#endif