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

        for (int nid = 0; nid <= (int)upto; ++nid)
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
        int per_thread_work = (upto + n_thread) / n_thread;
        vector<int> max_nid(sigmap.size());
        for (int nid = 0; nid <= upto; ++nid)
            max_nid[node2type[nid]] = nid;
        node2succs_typewise.resize(upto+1, vector<int>());
        for (int o = 0; o <= upto; o += per_thread_work)
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

    int AgendaScheduler::schedule(vector<VariableIndex>& batch)
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

    void AgendaScheduler::init(ComputationGraph *_cg, VariableIndex _upto)
    {
        cg = _cg;
        upto = _upto;
        basic_init();
        type2frontiers.resize(sigmap.size(), {});
        arity.resize(upto+1);
        for (int nid = 0; nid <= (int)upto; ++nid)
        {
            auto node = cg->nodes[nid];
            if (!(arity[nid] = node->arity()))
                type2frontiers[node2type[nid]].push_back(nid);
        }

        node2depth.resize(upto + 1);
        type2depth.resize(sigmap.size());
        type_cnt.resize(sigmap.size());
        for (auto &t : type2depth)
            t = 0;
        for (auto &t : type_cnt)
            t = 0;
        for (int nid = 0; nid <= upto; ++nid)
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

    void RLScheduler::init(ComputationGraph *_cg, VariableIndex _upto)
    {
        cg = _cg;
        upto = _upto;
        basic_init();
        train();

        n_hit = n_batch = 0;
        arity.resize(upto + 1);
        type2frontiers.resize(sigmap.size(), {});
        for (int nid = 0; nid <= (int)upto; ++nid)
        {
            auto node = cg->nodes[nid];
            if (!(arity[nid] = node->arity()))
                type2frontiers[node2type[nid]].push_back(nid);
        }
    }

    double RLScheduler::inference()
    {
        double loss = 0;
        type2frontiers.resize(sigmap.size(), {});
        for (auto& x: type2frontiers) x.clear();
        for (int nid = 0; nid <= upto; ++nid)
        {
            if (!(arity[nid] = cg->nodes[nid]->arity())){
                type2frontiers[node2type[nid]].push_back(nid);
            }
        }
        while (true)
        {
            vector<VariableIndex> batch;
            int suc = schedule_impl(batch, false);
            if (suc == 0)
                break;
            loss++;
        }
        return loss;
    }

    void RLScheduler::train()
    {
        if (trained.count(ooc_autobatch_flag)) return;
        cout << "[OoC::RLScheduler]: begin training!" << endl;
        TupleDict<RL::q_table_t> best_q_table;
        auto & q_table = q_tables[ooc_autobatch_flag];
        double best_loss = 1e9;
        typewise_init();
        cout << "finished typewise init!" << endl; 
        arity.resize(upto + 1);
        typewise_arity.resize(upto + 1);
        typewise_frontier_cnt.resize(sigmap.size());
        type2frontiers.resize(sigmap.size());
        eps = eps_max;

        for (int i = 0; i < n_iter; i++)
        {
            for (auto &x : typewise_arity)
                x = 0;
            for (auto &x : typewise_frontier_cnt)
                x = 0;
            for (auto& x: type2frontiers) x.clear();
            for (int nid = 0; nid <= upto; ++nid)
            {
                if ((arity[nid] = cg->nodes[nid]->arity()) == 0)
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
                int suc = schedule_impl(batch, true);
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
            if ((i + 1) % 100 == 0)
            {
                int loss = inference();
                if (loss < best_loss)
                {
                    best_loss = loss;
                    best_q_table = q_table;
                }
                cout << "[OoC::RLScheduler]: " << i+1 <<  ", " << inference() << "batches" << endl;
            }
        }
        q_table = best_q_table;
        cout << "[OoC::RLScheduler]: finished training " << inference() << "batches" << endl;
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

    int RLScheduler::schedule(vector<VariableIndex>& batch) {
        return schedule_impl(batch, false);
    }

    int RLScheduler::schedule_impl(vector<VariableIndex> &batch, bool train)
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
        return get_batches.size() == 0 ? 1 : -1;
    }

    void RLScheduler::post_process() {
        cout << "[RLScheduler::hit_rate]:" << n_hit << "/" << n_batch << endl;
    }
}

namespace dynet
{
    void BatchedExecutionEngine::getBatches_typewiseLB(
        VariableIndex upto,
        VariableIndex &batch_id)
    {
        assert(num_nodes_evaluated == 0);
        global_timer.start("schedule::preparation");
        vector<node_t> nodes(upto + 1);
        unordered_map<int, type_t> types;
        int fake_type = 0;
        for (VariableIndex nid = 0; nid <= upto; ++nid)
        {
            auto node = cg.nodes[nid];
            int type = node->autobatch_sig(cg, cg.sigmap);
            type = type == 0 ? --fake_type : type;
            nodes[nid].type = type;
            nodes[nid].args = node->args;
            nodes[nid].invalid = node->is_get;
            for (auto &arg : nodes[nid].args)
                if (cg.nodes[arg]->is_get)
                    arg = cg.nodes[arg]->args[0];
            if (types[type].min_nid < 0)
                types[type].min_nid = nid;
        }
        global_timer.stop("schedule::preparation");

        vector<future<double>> results;
        global_timer.start("schedule::compute G_t");
        int n_thread = 16;
        int per_thread_work = (upto + n_thread - 1) / n_thread;
        for (int o = 0; o < (int)nodes.size(); o += per_thread_work)
        {
            results.emplace_back(async([this, o, per_thread_work, &types, &nodes]
                                       {
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
                return node_explored; }));
        }
        double ave_explored = 0;
        for (auto &res : results)
            ave_explored += res.get();
        global_timer.cumint("ave_explored", ave_explored / nodes.size());
        global_timer.stop("schedule::compute G_t");

        for (int nid = nodes.size() - 1; nid >= 0; --nid)
        {
            auto &node = nodes[nid];
            if (node.invalid)
                continue;
            for (auto arg : nodes[nid].args)
                nodes[arg].succ_cnt++;
            for (auto arg : nodes[nid].typewise_args)
                nodes[arg].typewise_succ_cnt++;
            auto &type = types[nodes[nid].type];
            if (node.succ_cnt == 0)
                type.frontiers.push_back(nid);
            if (node.typewise_succ_cnt == 0)
                type.typewise_frontier_cnt += 1;
        }

        global_timer.start("scheule::get batches");
        // memory_affinity.resize(nodes.size());
        // for (int nid = 0; nid < (int)memory_affinity.size(); ++nid)
        //     memory_affinity[nid] = nid;
        // memory_affinity_tag = nodes.size();
        vector<vector<VariableIndex>> reversed_batches;
        list<int> active_types;
        for (auto &kv : types)
            if (kv.second.frontiers.size())
                active_types.push_back(kv.first);
        int idx = 0;
        while (active_types.size())
        {
            double best_score = 0;
            list<int>::iterator this_tid = active_types.end();
            for (auto iter = active_types.begin(); iter != active_types.end(); ++iter)
            {
                assert(types[*iter].frontiers.size());
                assert(types[*iter].typewise_frontier_cnt);
                double score = types[*iter].frontiers.size() / (double)types[*iter].typewise_frontier_cnt;
                if (best_score < score)
                {
                    best_score = score;
                    this_tid = iter;
                }
            }
            assert(this_tid != active_types.end());
            auto &type = types[*this_tid];
            active_types.erase(this_tid);
            type.typewise_frontier_cnt -= type.frontiers.size();

            if (cg.nodes[type.frontiers.front()]->is_function)
            { // function node
                int nid = type.frontiers.front();
                FunctorNode *node = static_cast<FunctorNode *>(cg.nodes[nid]);
                for (int i = node->n_output; i >= 1; --i)
                {
                    vector<VariableIndex> get_node_batch(type.frontiers);
                    for (auto &nid : get_node_batch)
                    {
                        nid += i;
                    }
                    reversed_batches.push_back(move(get_node_batch));
                }
            }

            reversed_batches.push_back(move(type.frontiers));
            assert(type.frontiers.size() == 0);
            int idx = 0, aid;
            for (auto id : reversed_batches.back())
            {
                for (auto arg : nodes[id].args)
                {
                    auto &input_node = nodes[arg];
                    if (--input_node.succ_cnt == 0)
                    {
                        if (!types[input_node.type].frontiers.size())
                        {
                            active_types.push_back(input_node.type);
                        }
                        types[input_node.type].frontiers.push_back(arg);
                    }
                }
                for (auto arg : nodes[id].typewise_args)
                {
                    auto &input_node = nodes[arg];
                    if (--input_node.typewise_succ_cnt == 0)
                    {
                        types[input_node.type].typewise_frontier_cnt++;
                    }
                }
                idx++;
            }
        }
        global_timer.stop("Scheule::get batches");

        for (auto iter = reversed_batches.rbegin(); iter != reversed_batches.rend(); iter++)
        {
            for (auto id : *iter)
                node2batch[id] = batch_id;
            batches[batch_id++].ids = move(*iter);
        }

        if (profiling_flag > 2)
        {
            cout << "*****************batch strategy 8***********" << endl;
            for (VariableIndex bid = num_batches_evaluated; bid < batch_id; bid++)
            {
                cout << "BID" << bid << "\t:";
                for (auto id : batches[bid].ids)
                    cout << cg.nodes[id]->as_dummy_string() << ",";
                cout << endl;
            }
            cout << "***************batch strategy end***********" << endl;
        }

        return;
    }
} // namespace dynet