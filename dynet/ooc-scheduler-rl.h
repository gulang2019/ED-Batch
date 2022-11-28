#pragma once 
#include <set>
#include <vector>
#include <unordered_map>
#include <deque>
#include <random>

#include "dynet/dynet.h"
#include "utils.h"

namespace dynet{
namespace RL{
    typedef std::set<int, std::less<int> > state_t; 

    struct node_t {
        std::vector<VariableIndex> args;
        int type;
        int succ_cnt = 0;
        bool invalid = true;
    };

    struct type_t{
        std::vector<VariableIndex> frontiers;  
    };
    struct q_table_t {
        std::unordered_map<int, double> q;
        void init(const state_t & state) {for(auto s: state) q[s] = 0.0;}
        size_t size(){return q.size();}
        double max(){
            double ret = 0;
            for (auto &kv: q) ret = std::max(ret, kv.second);
            return ret;
        }
        int argmax(){
            double v = -1e9;
            int arg = 0;
            for (auto &kv: q){
                if (kv.second > v)
                    v = kv.second, arg = kv.first;
            }
            if(arg == 0){
                cout << "error state: ";
                for (auto & kv: q) cout << "(" << kv.first << "," << kv.second << "),";
                cout << endl;
                assert(false);
            }
            return arg;
        }
    };

    struct metric_t{
        double ex = 0;
        double ex2 = 0;
        int n = 0;
    };

    struct Env {
        Env(ComputationGraph& _cg, VariableIndex _upto, int _num_nodes_evaluated);
        
        void reset();
        void get_state(state_t & state);
        double step(int type, std::vector<VariableIndex>& batch);
        
        std::vector<metric_t> history;
        double get_reward(); 
        bool done();
        ComputationGraph* cg;
        VariableIndex upto;
        int num_nodes_evaluated;
        double n_node; // number of node commited 
        int n_step; // number of step taken 
        node_t* nodes = nullptr;
        std::unordered_map<int, type_t> types; 
        ~Env() {delete[] nodes;}
    };

    struct buffer_t {
        state_t state;
        int action;
        double reward;
        state_t next_state;
    };

    struct ReplayBuffer{
        ReplayBuffer(size_t cap):cap(cap), head(0), full(false){
            buffer.resize(cap);
        }
        
        void add(const buffer_t & b) {
            buffer[head++] = b;
            if (head >= cap) {
                full = true; 
                head = 0;
            }
        }

        buffer_t& sample(){
            if (full) return buffer[random() % cap];
            return buffer[random()%head];
        }

        // return buffer[(head - idx) % cap]
        buffer_t& get(int idx) {
            return buffer[(head - idx + cap) % cap];
        }

        size_t cap;
        int head;
        bool full;
        vector<buffer_t> buffer;
    };

    struct QLearner{
        QLearner(int cap):buffer(cap){}

        OoC::TupleDict<q_table_t> q_tables;
        int take_action(const state_t & state, bool train = false);
        void train(Env*env, int iter);
        
        ReplayBuffer buffer;

        bool trained = false; 
        int type_ub;

        // hyper params 
        double gamma = 0.9;
        double epsilon = 0.8;
        double epsilon_decay = 0.99;
        double epsilon_lb = 0.05;
        double alpha = 0.2;
        int n_step = 10;
        int n_relay = 20;
    };
} // namespace RL
} // namespace dynet 