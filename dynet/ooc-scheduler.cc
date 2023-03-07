#include <functional>
#include <queue>

#include "dynet/timing.h"
#include "dynet/nodes-functor.h"
#include "dynet/ooc-executor.h"
#include "dynet/ooc-scheduler.h"
#include "dynet/sig.h"

using namespace std;
using namespace OoC;
using namespace dynet;

namespace OoC
{
    SigMap BaseScheduler::sigmap;
    std::unordered_map<int, TupleDict<RL::q_table_t> > RLScheduler::q_tables;
    std::unordered_set<int> RLScheduler::trained;
    void BaseScheduler::basic_init()
    {
        assert(cg != nullptr);
        sigmap.invalidate(nt::reduce);
        node2type.resize(upto + 1);
        node2succs.resize(upto + 1);

        for (int nid = num_nodes_evaluated; nid <= (int)upto; ++nid)
        {
            auto node = cg->nodes[nid];
            node2type[nid] = node->autobatch_sig(*cg, sigmap);
            for (auto arg : node->args)
                node2succs[arg].push_back(nid);
        }
    }

    void BaseScheduler::typewise_init()
    {
        vector<future<double>> results;
        int n_thread = 16;
        int per_thread_work = (upto + n_thread - num_nodes_evaluated) / n_thread;
        vector<int> max_nid(sigmap.size());
        for (int nid = num_nodes_evaluated; nid <= upto; ++nid)
            max_nid[node2type[nid]] = nid;
        node2succs_typewise.resize(upto+1, vector<int>());
        for (int o = num_nodes_evaluated; o <= upto; o += per_thread_work)
        {
            results.emplace_back(async([this, o, per_thread_work, max_nid]
                                       {
            priority_queue<VariableIndex, vector<VariableIndex>, greater<VariableIndex> > Q;
            double node_explored=0.0;
            for (int nid = o; nid < std::min(o+per_thread_work, (int)upto+1); nid++){
            // for (int nid = 0; nid <= upto; nid++){
                auto t = node2type[nid];
                if (t < cg->types.size() && 
                    !cg->types[t].self_reachable)
                    continue;
                for(auto succ: node2succs[nid])
                    if (succ <= max_nid[t])
                        Q.push(succ);
                while(!Q.empty()){
                    node_explored++;
                    VariableIndex idx = Q.top();
                    while(!Q.empty() && Q.top() == idx) Q.pop();
                    if (node2type[idx] == t){
                        node2succs_typewise[nid].push_back(idx);
                    }
                    else {
                        for (auto succ: node2succs[idx]){
                            if(succ <= max_nid[t]) Q.push(succ);
                        }
                    }
                }
            }
            return node_explored; }));
        }
        double ave_explored = 0;
        for (auto &res : results)
            ave_explored += res.get();
        global_timer.cumint("ave_explored", ave_explored);
    }

    int BaseScheduler::lower_bound() {
        if (node2type.size() != (upto + 1)) basic_init();
        int ret = 0;
        vector<int> depth(upto+1);
        for (int nid = num_nodes_evaluated; nid <= (int)upto; ++nid) 
            if (node2type[nid] == 0) ret++;
        for (auto tid = 1; tid < sigmap.size(); ++tid) {
            int max_d = 0;
            for (int nid = num_nodes_evaluated; nid <= (int)upto; ++nid) {
                auto & d = depth[nid] = (node2type[nid] == tid); 
                auto node = cg->nodes[nid];
                for (auto arg: node->args) {
                    d = max(depth[arg] + (node2type[nid] == tid), d);
                }
                max_d = max(d, max_d);
            }
            ret += max_d;
        }
        return ret;
    }

    int AgendaScheduler::schedule(vector<VariableIndex>& batch, int verbose)
    {
        int t = -1;
        double min_depth = 1e9;
        if (!get_batches.empty())
        {
            t = get_batches.front();
            get_batches.pop();
        }
        else if (type2frontiers[0].size())
        {
            t = 0;
        }
        else
        {
            for (int tid = 0; tid < type2frontiers.size(); ++tid)
            {
                if (!type2frontiers[tid].size())
                    continue;
                double d = type2depth[tid] / type_cnt[tid];
                if (min_depth > d)
                {
                    t = tid;
                    min_depth = d;
                }
            }
        }
        if (t < 0)
            return 0;
        
        if (verbose) {
            cout << "[";
            for (auto nid: type2frontiers[t]) cout << nid << ",";
            cout << "]: ";
            cout << "At State(";
            for (int tid = 0; tid < type2frontiers.size(); ++tid) {
                if (!type2frontiers[tid].size())
                    continue;
                cout << "[" << cg->nodes[type2frontiers[tid].front()]->as_dummy_string() << "," 
                    <<  type2depth[tid] << "," << type_cnt[tid] << "],";
            }  
            cout << "), choose to batch " << cg->nodes[type2frontiers[t].front()]->as_dummy_string() << endl;
            global_timer.cumint("batch-"+cg->nodes[type2frontiers[t].front()]->as_dummy_string(), 1);
            cout << endl;
        }

        batch.clear();
        if (t == 0)
        {
            batch.push_back(type2frontiers[0].back());
            type2frontiers[0].pop_back();
        }
        else
        {
            batch = move(type2frontiers[t]);
        }
        assert(batch.size());

        type_cnt[t] -= batch.size();
        for (auto nid : batch)
        {
            type2depth[t] -= node2depth[nid];

            for (auto succ : node2succs[nid])
            {
                if (--arity[succ] == 0)
                {
                    type2frontiers[node2type[succ]].push_back(succ);
                }
            }
        }

        if (cg->nodes[batch.front()]->is_function)
        {
            auto nid = batch.front();
            auto node = static_cast<FunctorNode *>(cg->nodes[nid]);
            for (int i = 0; i < node->n_output; ++i)
            {
                get_batches.push(node2type[nid + 1 + i]);
            }
        }
        return get_batches.size() == 0 ? 1 : -1;
    }

    void AgendaScheduler::init(ComputationGraph *_cg, VariableIndex _num_nodes_evaluated, VariableIndex _upto)
    {
        cg = _cg;
        num_nodes_evaluated = _num_nodes_evaluated;
        upto = _upto;
        basic_init();
        type2frontiers.resize(sigmap.size(), {});
        arity.resize(upto+1, 0);
        for (int nid = num_nodes_evaluated; nid <= (int)upto; ++nid)
        {
            auto node = cg->nodes[nid];
            for (auto arg: node->args) 
                arity[nid] += arg >= num_nodes_evaluated;
            if (!arity[nid])
                type2frontiers[node2type[nid]].push_back(nid);
        }

        node2depth.resize(upto + 1);
        type2depth.resize(sigmap.size());
        type_cnt.resize(sigmap.size());
        for (auto &t : type2depth)
            t = 0;
        for (auto &t : type_cnt)
            t = 0;
        for (int nid = num_nodes_evaluated; nid <= upto; ++nid)
        {
            int &depth = node2depth[nid] = 0;
            auto node = cg->nodes[nid];
            for (auto arg : node->args)
            {
                depth = std::max(depth, 1 + node2depth[arg]);
            }
            auto &type = node2type[nid];
            type2depth[type] += depth;
            type_cnt[type]++;
        }
    }

    void RLScheduler::init(ComputationGraph *_cg, VariableIndex _num_nodes_evaluated, VariableIndex _upto)
    {
        cg = _cg;
        num_nodes_evaluated = _num_nodes_evaluated;
        upto = _upto;
        basic_init();
        train();
        // sigmap.show();

        n_hit = n_batch = 0;
        arity.resize(upto + 1, 0);
        type2frontiers.resize(sigmap.size(), {});
        for (int nid = num_nodes_evaluated; nid <= (int)upto; ++nid)
        {
            auto node = cg->nodes[nid];
            for (auto arg: node->args)
                arity[nid] += arg >= num_nodes_evaluated;
            if (!arity[nid])
                type2frontiers[node2type[nid]].push_back(nid);
        }
    }

    int RLScheduler::inference()
    {
        int loss = 0;
        type2frontiers.resize(sigmap.size(), {});
        for (auto& x: type2frontiers) x.clear();
        for (int nid = num_nodes_evaluated; nid <= upto; ++nid)
        {
            auto node = cg->nodes[nid];
            arity[nid] = 0;
            for (auto arg: node->args)
                arity[nid] += arg >= num_nodes_evaluated;
            if (!arity[nid]){
                type2frontiers[node2type[nid]].push_back(nid);
            }
        }
        while (true)
        {
            vector<VariableIndex> batch;
            int suc = schedule_impl(batch, false, 0);
            if (suc == 0)
                break;
            loss++;
        }
        return loss;
    }

    void RLScheduler::train()
    {
        if (trained.count(ooc_autobatch_flag)) return;
        // cout << "[OoC::RLScheduler]: begin training!" << endl;
        TupleDict<RL::q_table_t> best_q_table;
        auto & q_table = q_tables[ooc_autobatch_flag];
        double best_loss = 1e9;
        typewise_init();
        // cout << "finished typewise init!" << endl; 
        arity.resize(upto + 1, 0);
        typewise_arity.resize(upto + 1);
        typewise_frontier_cnt.resize(sigmap.size());
        type2frontiers.resize(sigmap.size());
        eps = eps_max;

        int lb = lower_bound();
        // cout << "[Scheduler] lower_bound is " << lb << endl;
        int train_iter = -1;
        for (int i = 0; i < n_iter; i++)
        {
            for (auto &x : typewise_arity)
                x = 0;
            for (auto &x : typewise_frontier_cnt)
                x = 0;
            for (auto& x: type2frontiers) x.clear();
            for (int nid = num_nodes_evaluated; nid <= upto; ++nid)
            {
                auto node = cg->nodes[nid];
                for (auto arg: node->args) arity[nid] += arg >= num_nodes_evaluated;
                if (!arity[nid])
                    type2frontiers[node2type[nid]].push_back(nid);
                for (auto succ : node2succs_typewise[nid])
                    typewise_arity[succ]++;
                if (typewise_arity[nid] == 0)
                    typewise_frontier_cnt[node2type[nid]]++;
            }
            logs.clear();
            int n_batch = 0;
            while (true)
            {
                vector<VariableIndex> batch;
                int suc = schedule_impl(batch, true, 0);
                if (suc == 0)
                    break;
                n_batch++;
            }

            for (int i = 0; i < logs.size(); ++i)
            {
                int J = std::min(i + td_step, (int)logs.size());
                double g = (J == logs.size()) ? 0 : q_table[logs[J].state].max();
                for (int j = J - 1; j >= i; --j)
                {
                    g = g * gamma + logs[j].reward;
                }
                auto &q = q_table[logs[i].state].q[logs[i].action];
                q += alpha * (g - q);
            }
            eps = std::max(eps_min, eps_decay * eps);
            if ((i + 1) % 50 == 0)
            {
                int loss = inference();
                if (loss < best_loss)
                {
                    best_loss = loss;
                    best_q_table = q_table;
                }
                // cout << "[OoC::RLScheduler]: " << i+1 <<  ", " << inference() << "batches" << endl;
                if (best_loss < lb * 1.01) { // heuristic stop when have good enough policy
                    train_iter = i+1;
                    break;
                }
            }
        }
        if (train_iter == -1) {
            train_iter = n_iter;
        }
        global_timer.cumint("train_iter", train_iter);

        q_table = best_q_table;
        // cout << "[OoC::RLScheduler]: finished training " << "at step" << train_iter << ", launches" << inference() << "batches" << endl;
        trained.insert(ooc_autobatch_flag);
    }

    int RLScheduler::take_action(const vector<int> &state, bool train)
    {
        auto& q_table = q_tables[ooc_autobatch_flag];
        if (train){
            auto& entry = q_table[state];
            if (entry.q.size() == 0) {
                entry.init(state);
            }
            if ((random() / (RAND_MAX + 0.0)) < eps) {
                return state[random() % state.size()];
            }
            return entry.argmax();
        }
        auto ptr = q_table.get(state);
        n_batch++;
        if (ptr == nullptr)
            return state.back();
        n_hit ++;
        if (trained.count(ooc_autobatch_flag)) return ptr->action();
        return ptr->argmax();
    }

    int RLScheduler::schedule(vector<VariableIndex>& batch, int verbose) {
        return schedule_impl(batch, false, verbose);
    }

    int RLScheduler::schedule_impl(vector<VariableIndex> &batch, bool train, int verbose)
    {
        int t = -1;
        double min_depth = 1e9;
        if (!get_batches.empty())
        {
            t = get_batches.front();
            get_batches.pop();
        }
        else if (type2frontiers[0].size())
        {
            t = 0;
        }
        else
        {
            vector<int> state;
            for (int tid = 1; tid < type2frontiers.size(); ++tid)
            {
                if (type2frontiers[tid].size())
                    state.push_back(tid);
            }
            if (ooc_autobatch_flag == 3) {
                sort(state.begin(), state.end(), [this](int t1, int t2){
                    return type2frontiers[t1].size() > type2frontiers[t2].size();
                });
            } 
            else if (ooc_autobatch_flag == 4) {
                int tmax = 0;
                int max_size = 0;
                for (int tid = 1; tid < type2frontiers.size(); ++tid)
                {
                    if (type2frontiers[tid].size() > max_size){
                        max_size = type2frontiers[tid].size();
                        tmax = tid;
                    }
                }
                if (tmax) state.push_back(tmax);  
            }

            if (state.size()){
                t = take_action(state, train);
                if (train) {
                    logs.push_back({state, t,  -1 + ((type2frontiers[t].size() + 0.0) / typewise_frontier_cnt[t])});
                }
                if (verbose) {
                    cout << "[";
                    for (auto nid: type2frontiers[t]) cout << nid << ",";
                    cout << "]";
                    cout << "At State(";
                    for (auto tid: state) {
                        cout << cg->nodes[type2frontiers[tid].front()]->as_dummy_string() << ",";
                    }
                    cout << "), choose to batch " << cg->nodes[type2frontiers[t].front()]->as_dummy_string() << endl;
                    global_timer.cumint("batch-" + cg->nodes[type2frontiers[t].front()]->as_dummy_string(), 1);
                }
            }
        }
        if (t < 0)
            return 0;
        batch.clear();
        if (t == 0)
        {
            batch.push_back(type2frontiers[0].back());
            type2frontiers[0].pop_back();
        }
        else
        {
            batch = move(type2frontiers[t]);
        }
        assert(batch.size());

        for (auto nid : batch)
        {
            for (auto succ : node2succs[nid])
            {
                if (--arity[succ] == 0)
                    type2frontiers[node2type[succ]].push_back(succ);
            }
        }

        if (train)
        {
            typewise_frontier_cnt[t] -= batch.size();
            for (auto nid : batch)
            {
                for (auto succ : node2succs_typewise[nid])
                {
                    if (--typewise_arity[succ] == 0)
                    {
                        typewise_frontier_cnt[node2type[succ]]++;
                    }
                }
            }
        }

        if (cg->nodes[batch.front()]->is_function)
        {
            auto nid = batch.front();
            auto node = static_cast<FunctorNode *>(cg->nodes[nid]);
            for (int i = 0; i < node->n_output; ++i)
            {
                get_batches.push(node2type[nid + 1 + i]);
            }
        }
        global_timer.stop("schedule-phase2");
        return get_batches.size() == 0 ? 1 : -1;
    }

    void RLScheduler::post_process() {
        // cerr << "[RLScheduler::hit_rate]:" << n_hit << "/" << n_batch << endl;
    }
}