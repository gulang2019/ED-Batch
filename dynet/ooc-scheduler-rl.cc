#include "dynet/exec.h"
#include "dynet/timing.h"
#include "utils.h"
using namespace std;
using namespace OoC;

namespace dynet{
    typedef set<int> state_t; 

    struct node_t {
        const vector<VariableIndex>* args;
        int type;
        int succ_cnt = 0;
    };
    struct type_t{
        vector<VariableIndex> frontiers;  
    };
    struct q_table_t {
        unordered_map<int, double> q;
        void init(const state_t & state) {for(auto s: state) q[s] = 0.0;}
        size_t size(){return q.size();}
        double max(){
            double ret = 0;
            for (auto &kv: q) ret = std::max(ret, kv.second);
            return ret;
        }
        int argmax(){
            double v = -1e6;
            int arg = -1;
            for (auto &kv: q){
                if (kv.second > v)
                    v = kv.second, arg = kv.first;
            }
            return arg;
        }
    };

    struct Env {
        void set_cg(ComputationGraph& _cg, VariableIndex _upto, int _num_nodes_evaluated);
        void reset();
        void get_state(state_t & state);
        double step(int type, vector<VariableIndex>& batch);
        bool done();
        ComputationGraph * cg;
        VariableIndex upto;
        int num_nodes_evaluated;
        vector<node_t> nodes;
        unordered_map<int, type_t> types; 
        ~Env(){cout << "destruct Env;" << endl;}
    };

    struct QLearner{
        TupleDict<q_table_t> q_tables;
        int take_action(const state_t & state, bool train = false);
        void train(Env*env, int iter);
        struct buffer_t {
            state_t state;
            int action;
            double reward;
        };
        deque<buffer_t> buffer;

        int type_ub;

        // hyper params 
        double gamma = 0.9;
        double epsilon = 0.8;
        double epsilon_decay = 0.99;
        double epsilon_lb = 0.05;
        double alpha = 0.2;
        int n_step = 10;
        ~QLearner(){cout << "destruct QLearner;" << endl;}
    };

    bool trained = false;

    void Env::set_cg(ComputationGraph& cg, VariableIndex _upto, int _num_nodes_evaluated){
        upto = _upto; 
        num_nodes_evaluated = _num_nodes_evaluated;
        cout << "clear nodes " << nodes.size() << endl;
        nodes.clear();
        cout << "clear types " << types.size() << endl;
        types.clear();
        int fake_type = 0;
        cout << "set_cg upto" << upto << " " << num_nodes_evaluated << endl;
        for (auto node: cg.nodes) {
            int type = node->autobatch_sig(cg, cg.sigmap);
            type = type==0? --fake_type:type;
            nodes.push_back({&node->args,type,0});
        }
        cout << "finish set cg" << endl;
    }

    void Env::reset(){
        for (auto& node: nodes) node.succ_cnt = 0;
        for (auto& type: types) type.second.frontiers.clear();
        for (int nid = upto; nid >= num_nodes_evaluated; --nid){
            auto & node = nodes[nid];
            for (auto arg: *node.args) nodes[arg].succ_cnt ++;
            if (node.succ_cnt == 0) {
                types[node.type].frontiers.push_back(nid);
            }
        }
    }

    double Env::step(int type, vector<VariableIndex>&batch){
        if (!types[type].frontiers.size()) 
            throw runtime_error("bad policy");
        batch = move(types[type].frontiers);
        assert(types[type].frontiers.size() == 0);
        for (auto & nid: batch){
            auto node = nodes[nid];
            for (auto arg: *node.args){
                if (--nodes[arg].succ_cnt == 0){
                    types[nodes[arg].type].frontiers.push_back(arg);
                }
            }
        }
        return -1;
    }

    void Env::get_state(state_t & state){
        for (auto& kv: types) {
            if (kv.second.frontiers.size())
                state.insert(kv.first);
        }
    }

    void QLearner::train(Env*env, int n_iter){
        cout << "training!" << endl;
        vector<VariableIndex> batch;
        for (int i = 0; i < n_iter; i++){
            env->reset();
            double total_reward = 0;
            while(true) {
                state_t state;
                env->get_state(state);
                if (!state.size()) break;
                if (*state.begin() < 0) {
                    env->step(*state.begin(), batch);
                    continue;
                }
                if (buffer.size() >= n_step){
                    double g = q_tables[state].max();
                    for (int j = n_step - 1; j >= 0; --j){
                        g = g * gamma + buffer[j].reward;
                    }
                    auto & q = q_tables[buffer.front().state].q[buffer.front().action];
                    q += alpha * (g - q);
                    buffer.pop_front();
                }
                int action = take_action(state, true);
                double reward = env->step(action, batch);
                total_reward += reward;
                epsilon *= epsilon > epsilon_lb? 1:epsilon_decay; 
                buffer.push_back({move(state), action, reward});
            }
            double g = 0;
            for (int j = buffer.size() - 1; j >= 0; --j){
                g = g * gamma + buffer[j].reward;
                auto & q = q_tables[buffer[j].state].q[buffer[j].action];
                q += alpha * (g - q);
            }
            buffer.clear();
            if ((i+1)%10==0)
                cout << "iter " << i+1 << ",reward:" << total_reward << endl;
        }
        type_ub = 0;
        for (auto& kv: env->types) type_ub = max(type_ub, kv.first);
    }

    int QLearner::take_action(const state_t & state, bool train){
        assert(state.size());
        if (train){
            if ((random() / (RAND_MAX+0.0)) < epsilon) {
                int i = random() % state.size();
                auto iter = state.begin();
                while(i){
                    --i, ++iter;
                }
                return *iter;
            }
        }
        auto& q_table = q_tables[state];
        if (q_table.size() == 0) {
            q_table.init(state);
        }
        return q_table.argmax();
    }


    void BatchedExecutionEngine::getBatches_rl(
        VariableIndex upto,
        VariableIndex & batch_id
    ){
        static QLearner qlearner;

        Env env;
        cout << "set_cg begins" << endl;
        env.set_cg(cg, upto, num_nodes_evaluated);
        cout << "set_cg finished" << endl;
        if (!trained){
            qlearner.train(&env, 100);
            trained = true;
        }
        
        env.reset();
        vector<vector<VariableIndex> > my_batches;
        while(true){
            state_t state;
            env.get_state(state);
            if (!state.size()) break;
            int action;
            if (*state.begin() < 0) action = *state.begin();
            else if (*state.rbegin() > qlearner.type_ub) action = *state.rbegin();
            else action = qlearner.take_action(state);
            my_batches.push_back({});
            env.step(action, my_batches.back());
        }   

        for (auto iter = my_batches.rbegin(); iter != my_batches.rend(); iter++){
            for (auto id: *iter) {node2batch[id] = batch_id; node2size[id] = cg.nodes[id]->dim.size();}
            batches[batch_id++].ids = move(*iter);
        }
        return;
    }

} // namespace dynet 