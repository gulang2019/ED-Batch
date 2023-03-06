#include <iostream>
#include <queue>
#include <thread>
#include "OoC.h"
using namespace std;
namespace OoC{

    void StaticScheduler::show(){
        if (profiling_flag > 0){
          fprintf(stdout, "--------------show-----------\n");
          fprintf(stdout, "types: \n");
          for (int tid = 0; tid < _types.size(); tid ++){
            if (_types[tid].frontiers.size() == 0) continue;
            fprintf(stdout, "\t%s: ", type2name[tid].c_str());
            for(auto nid: _types[tid].frontiers) fprintf(stdout, "%d, ", nid);
            fprintf(stdout, "\n"); 
          }
          fprintf(stdout, "----------------------------------\n");
        }
    }
    int StaticScheduler::lower_bound(){
        int ret = 0;
        for (int tid = 0; tid < _types.size(); tid++){
            vector<int> depth(_nodes.size(), 0);
            int max_depth = 0;
            for (int nid = 0; nid < _nodes.size(); nid++){
                bool is_this_type = _nodes[nid]->type == tid;
                depth[nid] = is_this_type;
                for (auto succ: _nodes[nid]->succs) {
                    depth[nid] = max(depth[nid], (is_this_type) + depth[succ]);
                }
                max_depth = max(max_depth, depth[nid]);
            }
            ret += max_depth;
        }
        return ret;
    }

    void TypewiseLBScheduler::init(){
        ooc_timer.start("TypewiseLBScheduler::init");
        // cout << "begin init" << endl;
        nodes.resize(_nodes.size());
        frontier_cnt.resize(_types.size(), 0);
        
        vector<int> first_appear(_types.size(), -1);
        for (int nid = 0; nid < _nodes.size(); nid++){
            auto & v = first_appear[_nodes[nid]->type];
            v = v < 0? nid: v;
        }
        vector<future<double>> results;
        int n_thread = 16;
        int work_per_thread = (_nodes.size() + n_thread - 1) / n_thread;
        // ooc_timer.start("TypewiseLBScheduler::calc_succ");
        for (int i = 0; i < _nodes.size(); i += work_per_thread){
            results.emplace_back(async([this, &first_appear, i, work_per_thread](){
                double explored = 0;
                priority_queue<int> Q;
                for (int nid = i; nid < min(i + work_per_thread, (int)_nodes.size()); nid++){
                    int this_type = _nodes[nid]->type;
                    int min_pos = first_appear[this_type];
                    for (auto succ: _nodes[nid]->succs) 
                        if (succ >= min_pos) Q.push(succ);
                    // ret->push_back({});
                    while (!Q.empty()) {
                        explored ++; 
                        int id = Q.top();
                        while(!Q.empty() && Q.top() == id) Q.pop();
                        if (_nodes[id]->type == this_type) {
                            nodes[nid].succs.push_back(id);
                        }
                        else {
                            for (auto succ: _nodes[id]->succs) {
                                if (succ >= min_pos) Q.push(succ);
                            }
                        }
                    } 
                }
                return explored;
            }));
        }
        
        double mean_explored = 0; 
        for (int t = 0; t < results.size(); t++){
            mean_explored += results[t].get();
        }
        // ooc_timer.stop("TypewiseLBScheduler::calc_succ");

        for (int nid = nodes.size() - 1; nid >= 0; nid--){
            for (auto succ: nodes[nid].succs)
                nodes[succ].inputCnt ++;
            frontier_cnt[_nodes[nid]->type] += nodes[nid].inputCnt == 0;
        }
        mean_explored /= _nodes.size();
        if (profiling_flag > 0)
            fprintf(stdout, "[TypewiseLBScheduler::init]: average explored %f\n", mean_explored);

        if (profiling_flag > 0){
            fprintf(stdout, "---------typewise init--------\n");
            for (int nid = 0; nid < nodes.size(); nid++){
                fprintf(stdout, "[node%d,%s,%d]: ", nid, 
                    OoC::type2name[_nodes[nid]->type].c_str(), nodes[nid].inputCnt);
                for (auto arg: _nodes[nid]->succs) fprintf(stdout, "%d, ", arg);
                fprintf(stdout, "|");
                for (auto arg: nodes[nid].succs) fprintf(stdout, "%d, ", arg);
                fprintf(stdout, "\n");
            }
            fprintf(stdout, "-----typewise init end--------\n");
        }
        ooc_timer.stop("TypewiseLBScheduler::init");
        // cout << "end init" << endl;
    }

    bool TypewiseLBScheduler::get_next_batch(vector<int>& batch){
        ooc_timer.start("get_next_batch");
        int this_tid = -1;
        double max_score = 0;
        for (int tid = 0; tid < _types.size(); tid++){
            int n_frontier = _types[tid].frontiers.size();
            if (!n_frontier) continue;
            double score = (double)(n_frontier) / frontier_cnt[tid];
            if (max_score < score) {
                max_score = score;
                this_tid = tid;
            }
        }
        if (this_tid < 0) {
            if (profiling_flag > 0){
                fprintf(stdout, "[typewise] %d batches, lb %d, random %d times\n", 
                    n_batch, StaticScheduler::lower_bound(), n_rand);
            }
            ooc_timer.stop("get_next_batch");
            return false; 
        }
        if (max_score < 1-1e-6) {
            StaticScheduler::show();
            // fprintf(stdout, "[scheduler]: use random %f\n", max_score);
            n_rand++;
        }
        batch = move(_types[this_tid].frontiers);
        frontier_cnt[this_tid] -= batch.size();
        for (auto& id: batch){
            for (auto&child: _nodes[id]->succs){
                if (--_nodes[child]->inputCnt == 0){
                    _types[_nodes[child]->type].frontiers.push_back(child);
                }
            }
            for (auto&child: nodes[id].succs){
                if (--nodes[child].inputCnt==0){
                    frontier_cnt[_nodes[child]->type] ++;
                }
            }
        }
        n_batch ++;
        ooc_timer.stop("get_next_batch");
        return true;
    }   

    void DynetScheduler::init(){
        cnt.resize(_types.size(), 0);
        depth.resize(_types.size(), 0);
        vector<int> topo_depth(_nodes.size(), 0);
        for (int nid = _nodes.size()-1; nid >= 0; --nid){
            auto node = _nodes[nid];
            for (auto succ: node->succs)
                topo_depth[succ] = max(topo_depth[succ], 1 + topo_depth[nid]);
            depth[node->type] += topo_depth[nid];
            cnt[node->type]++;
        }
    }

    bool DynetScheduler::get_next_batch(vector<int>& batch){
        int this_tid = -1;
        double min_depth = 1e6;
        for (int tid = 0; tid < _types.size(); ++tid){
            if (_types[tid].frontiers.size() == 0) continue;
            if (min_depth > (depth[tid] / cnt[tid])) {
                this_tid = tid;
                min_depth = (depth[tid] / cnt[tid]);
            }
        }
        
        if (this_tid < 0) {
            if (profiling_flag)
                fprintf(stdout, "[dynet]: %d batch, lb %d batch\n", n_batch, StaticScheduler::lower_bound());
            return false;
        }

        batch = move(_types[this_tid].frontiers);
        cnt[this_tid] -= batch.size();
        for (auto& id: batch){
            for (auto&child: _nodes[id]->succs){
                if (--_nodes[child]->inputCnt == 0){
                    _types[_nodes[child]->type].frontiers.push_back(child);
                    depth[_nodes[child]->type] --;
                }
            }
        }

        n_batch ++;
        return true;
    }

    void TFFoldScheduler::init(){
        depth.resize(_nodes.size(), 0);
        exec_order.resize(_nodes.size());
        for (int nid = _nodes.size()-1; nid >= 0; --nid){
            exec_order[nid] = nid;
            auto node = _nodes[nid];
            for (auto succ: node->succs)
                depth[succ] = max(depth[succ], depth[nid]+1);
        }
        
        sort(exec_order.begin(), exec_order.end(), [this](int nid1, int nid2){
            if (depth[nid1] == depth[nid2]) return _nodes[nid1]->type < _nodes[nid2]->type;
            return depth[nid1] < depth[nid2];
        });
    }

    bool TFFoldScheduler::get_next_batch(vector<int>& batch){
        if (idx == exec_order.size()) {
            if (profiling_flag)
                fprintf(stdout, "[tf-fold]: %d batch, lb %d batch\n", n_batch, StaticScheduler::lower_bound());
            return false;    
        }
        size_t old_idx = idx;
        while (idx < exec_order.size() 
        && depth[exec_order[old_idx]] == depth[exec_order[idx]]
        && _nodes[exec_order[old_idx]]->type == _nodes[exec_order[idx]]->type){
            // depth type no change
            idx++;
        }
        batch.clear();
        for (size_t i = old_idx; i < idx; i++){
            batch.push_back(exec_order[i]);
        }
        n_batch++;
        return true;
    }


    void LearnableScheduler::init(){
        scheduler = new QLearningModel();
        scheduler->train(_nodes, _types, 1);
    }

    bool LearnableScheduler::get_next_batch(vector<int>& batch){
        set<int> state;
        for (int tid = 0; tid < _types.size() ; ++tid){
            if (_types[tid].frontiers.size()) 
                state.insert(tid);
        }
        
        if (!state.size()){
            fprintf(stdout, "[RL]: %d batches, lb %d batches\n", 
            n_batch, StaticScheduler::lower_bound());
            return false;
        }
        
        int this_tid = scheduler->get_action(state);

        batch = move(_types[this_tid].frontiers);
        for (auto& id: batch){
            for (auto&child: _nodes[id]->succs){
                if (--_nodes[child]->inputCnt == 0){
                    _types[_nodes[child]->type].frontiers.push_back(child);
                }
            }
        }

        n_batch ++;
        return true;
    }

    bool LearnableScheduler::trained = false;
    
    Scheduler* LearnableScheduler::scheduler = nullptr;
    
} // namespace OoC