#include "dynet/exec.h"
#include "dynet/timing.h"
#include "utils.h"
#include "dynet/nodes-functor.h"
#include "dynet/ooc-scheduler-rl.h"

using namespace std;
using namespace OoC;

namespace dynet{
namespace RL {
    double Env::get_reward(){
        if (history.size() <= n_step)  history.resize(n_step + 1);
        auto & entry = history[n_step];
        entry.n += 1;
        entry.ex = entry.ex + 1.0 / entry.n * (n_node - entry.ex);
        entry.ex2 = entry.ex2 + 1.0 / entry.n * (n_node * n_node - entry.ex2);
        auto reward =  (n_node - entry.ex) / std::sqrt(entry.ex2 - entry.ex * entry.ex + 1e-6);
        return reward;
    }

    Env::Env(ComputationGraph& _cg, VariableIndex _upto, int _num_nodes_evaluated):
    upto(_upto), num_nodes_evaluated(_num_nodes_evaluated), cg(&_cg){
        nodes = new RL::node_t[upto+1];
        assert(nodes!=nullptr);
        int fake_type = 0;
        for (int nid = 0; nid < num_nodes_evaluated; ++ nid){
            nodes[nid].invalid = true;
        }
        for (int nid = num_nodes_evaluated; nid <= upto; nid++) {
            auto _node = _cg.nodes[nid];
            int type = _node->autobatch_sig(_cg, _cg.sigmap);
            type = type==0? --fake_type:type;
            auto & node = nodes[nid];
            node.args = _node->args;
            for (auto& arg: node.args) 
                if (_cg.nodes[arg]->is_get) arg = _cg.nodes[arg]->args[0];
            node.type = type;
            node.succ_cnt = 0;
            node.invalid = _node->is_get;
        }
    }

    void Env::reset(){
        for (int nid = num_nodes_evaluated; nid <= (int)upto; ++nid) nodes[nid].succ_cnt = 0;
        for (auto& type: types) type.second.frontiers.clear();
        for (int nid = upto; nid >= num_nodes_evaluated; --nid){
            auto & node = nodes[nid];
            if (node.invalid) continue;
            for (auto arg: node.args) nodes[arg].succ_cnt ++;
            if (node.succ_cnt == 0) {
                types[node.type].frontiers.push_back(nid);
            }
        }
        n_node = 0;
        n_step = -1;
    }

    double Env::step(int type, vector<VariableIndex>&batch){
        if (!types[type].frontiers.size()) 
            throw runtime_error("bad policy");
        batch = move(types[type].frontiers);
        n_node += batch.size();
        n_step += 1;
        for (auto & nid: batch){
            if (!(nid >= 0 && nid <= upto)){
                cg->show_nodes();
                for (int nid = 0; nid <= upto; nid ++ ){
                    auto & node = nodes[nid];
                    cout << "NID" << nid << ": args";
                    for (auto arg:node.args) cout << arg << ",";
                    cout << "valid: " << !node.invalid;
                    cout << endl;
                }
                assert(false);
            }
            auto node = nodes[nid];
            for (auto arg: node.args){
                if (--nodes[arg].succ_cnt == 0 && !nodes[arg].invalid){
                    types[nodes[arg].type].frontiers.push_back(arg);
                }
            }
        }
        return get_reward(); 
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
            state_t state, next_state;
            env->get_state(state);
            int step = 0;
            while(true) {
                if (!state.size()) break;
                if (step >= n_step){
                    double g = q_tables[state].max();
                    for (int j = 0; j < n_step; ++j){
                        g = g * gamma + buffer.get(j).reward;
                    }
                    auto & q = q_tables[buffer.get(n_step - 1).state].q[buffer.get(n_step -1).action];
                    q += alpha * (g - q);
                }
                int action = take_action(state, true);
                assert(state.count(action));
                double reward = env->step(action, batch);
                epsilon *= epsilon > epsilon_lb? 1:epsilon_decay;
                assert(next_state.size() == 0);
                env->get_state(next_state); 
                buffer.add({move(state), action, reward, next_state});
                assert(state.size() == 0);
                state = move(next_state);
                step ++;
            }

            double g = 0;
            for (int j = 0; j < std::min(step, n_step); ++j){
                auto & entry = buffer.get(j);
                g = g * gamma + entry.reward;
                auto & q = q_tables[entry.state].q[entry.action];
                q += alpha * (g - q);
            }

            // replay 
            for (int i = 0; i < 20; i++){
                buffer_t& entry = buffer.sample();
                auto & q = q_tables[entry.state].q[entry.action];
                q += alpha * ((entry.reward + gamma * q_tables[entry.next_state].max()) - q);
            }

            if (profiling_flag && (i+1)%100==0)
                cout << "iter " << i+1 << ",n_batch:" << env->n_step + 1 << endl;
        }
        type_ub = 0;
        for (auto& kv: env->types) type_ub = std::max(type_ub, kv.first);
    }

    int QLearner::take_action(const state_t & state, bool train){
        assert(state.size());
        if (*state.begin() < 0) {
            auto act = *state.begin();
            q_tables[state].q[act] = 1.0;
            return act;
        }
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
} // namespace RL 
} // namespace dynet

namespace dynet{ 
    void BatchedExecutionEngine::getBatches_rl(
        VariableIndex upto,
        VariableIndex & batch_id
    ){
        global_timer.start("RL::preprocess");
        RL::Env env(cg, upto, num_nodes_evaluated);
        global_timer.stop("RL::preprocess");
        static RL::QLearner qlearner(1000);

        if (!qlearner.trained){
            qlearner.train(&env, 1000);
            qlearner.trained = true;
        }
        
        global_timer.start("RL::batching");
        env.reset();
        vector<vector<VariableIndex> > my_batches;
        while(true){
            RL::state_t state;
            env.get_state(state);
            if (!state.size()) break;
            int action;
            if (*state.rbegin() > qlearner.type_ub) action = *state.rbegin();
            else action = qlearner.take_action(state);
            my_batches.push_back({});
            env.step(action, my_batches.back());
        }   
        global_timer.stop("RL::batching");


        global_timer.start("RL::postprocess");
        for (auto iter = my_batches.rbegin(); iter != my_batches.rend(); iter++){
            auto & my_batch = batches[batch_id];
            for (auto id: *iter) {node2batch[id] = batch_id; }
            my_batch.ids = move(*iter);
            batch_id++;
            auto examplar = cg.nodes[my_batch.ids.front()];
            if (examplar->is_function){
                FunctorNode* node = static_cast<FunctorNode*>(examplar);
                for (int i = 1; i <= node->n_output; ++i) {
                    vector<VariableIndex> get_node_batch(my_batch.ids);
                    for(auto& nid: get_node_batch) {
                        nid += i;
                        node2batch[nid] = batch_id;
                    }
                    batches[batch_id++].ids = move(get_node_batch);
                }
            }
            
        }
        global_timer.stop("RL::postprocess");

        return;
    }

} // namespace dynet 