#include "exec.h"
#include "dynet/timing.h"
#include "dynet/nodes-functor.h"
#include <functional>
#include <queue>
using namespace std;
using namespace OoC;

namespace dynet {
    struct node_t {
        int type;
        vector<VariableIndex> args;
        bool invalid = false;
        int succ_cnt = 0;
        vector<VariableIndex> typewise_args;
        int typewise_succ_cnt = 0;
        int mem_affinity;
    };

    struct node_type_t {
        vector<VariableIndex> frontiers;
        int typewise_frontier_cnt = 0;
        int min_nid = -1;
    };

    void BatchedExecutionEngine::getBatches_typewiseLB(
        VariableIndex upto, 
        VariableIndex& batch_id
    ){
        assert(num_nodes_evaluated == 0);
        global_timer.start("schedule::preparation");
        vector<node_t> nodes(upto+1);
        unordered_map<int, node_type_t> types;
        int fake_type = 0;
        for (VariableIndex nid = 0; nid <= upto; ++nid) {
            auto node = cg.nodes[nid];
            node2size[nid] = node->dim.size();
            int type = node->autobatch_sig(cg, cg.sigmap);
            type = type == 0? --fake_type:type;
            nodes[nid].type = type;
            nodes[nid].args = node->args;
            nodes[nid].mem_affinity = nid;
            nodes[nid].invalid = node->is_get;
            for (auto & arg: nodes[nid].args) 
                if (cg.nodes[arg]->is_get) 
                    arg = cg.nodes[arg]->args[0];
            if (types[type].min_nid < 0)
                types[type].min_nid = nid;
        }
        global_timer.stop("schedule::preparation");

        vector<future<double> > results;
        global_timer.start("schedule::compute G_t");
        int n_thread = 16;
        int per_thread_work = (nodes.size() + n_thread - 1) / n_thread;
        for (int o = 0; o < (int)nodes.size(); o += per_thread_work) {
            results.emplace_back(async([this, o, per_thread_work, &types, &nodes]{
                priority_queue<VariableIndex> Q;
                double node_explored=0.0;
                for (int nid = o; nid < std::min(o+per_thread_work, (int)nodes.size()); nid++){
                    auto& node = nodes[nid];
                    if (node.invalid) continue;
                    if (node.type >= 0 && node.type < cg.types.size() && 
                        !cg.types[node.type].self_reachable)
                        continue;
                    int min_nid = types[node.type].min_nid;
                    for(auto arg: node.args)
                        if (arg >= min_nid)Q.push(arg);
                    while(!Q.empty()){
                        node_explored++;
                        VariableIndex idx = Q.top();
                        while(!Q.empty() && Q.top() == idx) Q.pop();
                        if (nodes[idx].type == node.type){
                            node.typewise_args.push_back(idx);
                        }
                        else {
                            for (auto arg: nodes[idx].args){
                                if(arg >= min_nid) Q.push(arg);
                            }
                        }
                    }
                }
                return node_explored;
            }));
        }
        double ave_explored = 0;
        for(auto&res: results) ave_explored += res.get();
        global_timer.cumint("ave_explored", ave_explored / nodes.size());
        global_timer.stop("schedule::compute G_t");

        for (int nid = nodes.size() - 1; nid >= 0; --nid){
            auto & node = nodes[nid];
            if (node.invalid) continue;
            for (auto arg: nodes[nid].args)
                nodes[arg].succ_cnt++;
            for (auto arg: nodes[nid].typewise_args)
                nodes[arg].typewise_succ_cnt++;
            auto & type = types[nodes[nid].type];
            if (node.succ_cnt == 0) type.frontiers.push_back(nid);
            if (node.typewise_succ_cnt == 0) type.typewise_frontier_cnt += 1;
        }

        global_timer.start("scheule::get batches");
        // memory_affinity.resize(nodes.size());
        // for (int nid = 0; nid < (int)memory_affinity.size(); ++nid) 
        //     memory_affinity[nid] = nid;
        // memory_affinity_tag = nodes.size();
        vector<vector<VariableIndex> > reversed_batches;
        list<int> active_types;
        for (auto &kv: types) 
            if (kv.second.frontiers.size()) active_types.push_back(kv.first);
        int idx = 0;
        while(active_types.size()){
            double best_score = 0;
            list<int>::iterator this_tid = active_types.end();
            for (auto iter = active_types.begin(); iter != active_types.end(); ++iter){
                assert(types[*iter].frontiers.size());
                assert(types[*iter].typewise_frontier_cnt);
                double score = types[*iter].frontiers.size() / (double)types[*iter].typewise_frontier_cnt;
                if (best_score < score){
                    best_score = score;
                    this_tid = iter;
                } 
            }
            assert(this_tid != active_types.end());
            auto & type = types[*this_tid];
            active_types.erase(this_tid);
            type.typewise_frontier_cnt -= type.frontiers.size();

            if (cg.nodes[type.frontiers.front()]->is_function){ // function node
                int nid = type.frontiers.front();
                FunctorNode* node = static_cast<FunctorNode*>(cg.nodes[nid]);
                for (int i = node->n_output; i >= 1; --i) {
                    vector<VariableIndex> get_node_batch(type.frontiers);
                    for(auto& nid: get_node_batch) {
                        nid += i;
                    }
                    reversed_batches.push_back(move(get_node_batch));
                }
            }

            reversed_batches.push_back(move(type.frontiers));
            assert(type.frontiers.size() == 0);
            int idx = 0, aid;
            for (auto id: reversed_batches.back()){
                for (auto arg: nodes[id].args){
                    auto & input_node = nodes[arg];
                    if (--input_node.succ_cnt == 0){
                        if (!types[input_node.type].frontiers.size()){
                            active_types.push_back(input_node.type);
                        }
                        types[input_node.type].frontiers.push_back(arg);
                    }
                }
                for (auto arg: nodes[id].typewise_args){
                    auto & input_node = nodes[arg];
                    if (--input_node.typewise_succ_cnt == 0){
                        types[input_node.type].typewise_frontier_cnt++;
                    }
                }
                idx++;
            }
        }
        global_timer.stop("Scheule::get batches");

        for (auto iter = reversed_batches.rbegin(); iter != reversed_batches.rend(); iter++){
            for (auto id: *iter) node2batch[id] = batch_id;
            batches[batch_id++].ids = move(*iter);
        }


        if (profiling_flag > 2){
            cout << "*****************batch strategy 8***********" << endl;
            for (VariableIndex bid = num_batches_evaluated; bid < batch_id; bid++){
                cout << "BID" << bid << "\t:";
                for (auto id: batches[bid].ids)
                    cout << cg.nodes[id]->as_dummy_string() << ",";
                cout << endl;
            }
            cout << "***************batch strategy end***********" << endl;
        }
        
        return;
    }
} // namespace dynet 