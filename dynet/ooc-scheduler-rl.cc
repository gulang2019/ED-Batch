#include "dynet/exec.h"
#include "dynet/timing.h"
#include "utils.h"
#include "dynet/nodes-functor.h"
#include "dynet/ooc-scheduler.h"

#include <future>
#include <queue>

using namespace std;
using namespace OoC;
using namespace dynet;

namespace OoC
{
    namespace RL
    {
        Env::Env(ComputationGraph &_cg, VariableIndex _upto, int _num_nodes_evaluated, mode_t mode, SigMap &sigmap) : upto(_upto), num_nodes_evaluated(_num_nodes_evaluated), cg(&_cg), mode(mode), sigmap(sigmap)
        {
            init_basic();
            if (mode == TYPEWISE)
                init_typewise();
        }
        void Env::reset()
        {
            reset_basic();
            if (mode == TYPEWISE)
                reset_typewise();
        }

        void Env::init_basic()
        {
            nodes.resize(upto + 1);
            vector<future<void>> results;
            int fake_type = 0;
            for (int nid = num_nodes_evaluated; nid <= upto; ++nid)
            {
                auto _node = cg->nodes[nid];
                if (_node->is_get)
                    continue;
                auto &node = nodes[nid];
                node.invalid = _node->is_get;
                node.args = _node->args;
                for (auto &arg : node.args)
                    if (cg->nodes[arg]->is_get)
                        arg = cg->nodes[arg]->args[0];
                node.type = _node->autobatch_sig(*cg, sigmap);
                if (node.type == 0)
                    node.type = --fake_type;
                node.succ_cnt = 0;
            }
        }

        int Env::get_largest_type()
        {
            auto ret = 0;
            for (auto &kv : types)
            {
                ret = std::max(ret, kv.first);
            }
            return ret;
        }

        void Env::init_typewise()
        {
            vector<future<double>> results;
            global_timer.start("schedule::compute G_t");
            int n_thread = 16;
            int per_thread_work = (upto + n_thread - 1) / n_thread;
            for (int nid = num_nodes_evaluated; nid <= upto; ++nid)
            {
                int t = nodes[nid].type;
                if (types[t].min_nid < 0)
                    types[t].min_nid = nid;
            }
            for (int o = 0; o < (int)nodes.size(); o += per_thread_work)
            {
                results.emplace_back(async([this, o, per_thread_work]
                                           {
                priority_queue<VariableIndex> Q;
                double node_explored=0.0;
                for (int nid = o; nid < std::min(o+per_thread_work, (int)nodes.size()); nid++){
                    auto& node = nodes[nid];
                    if (node.invalid) continue;
                    if (node.type >= 0 && node.type < cg->types.size() && 
                        !cg->types[node.type].self_reachable)
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
        }

        void Env::reset_basic()
        {
            for (int nid = num_nodes_evaluated; nid <= (int)upto; ++nid)
                nodes[nid].succ_cnt = 0;
            for (auto &type : types)
                type.second.frontiers.clear();
            for (int nid = upto; nid >= num_nodes_evaluated; --nid)
            {
                auto &node = nodes[nid];
                if (node.invalid)
                    continue;
                for (auto arg : node.args)
                    nodes[arg].succ_cnt++;
                if (node.succ_cnt == 0)
                {
                    types[node.type].frontiers.push_back(nid);
                }
            }
            state.clear();
            get_state(state);
            // for (int nid = num_nodes_evaluated; nid <= upto; ++nid){
            //     cout << "NID" << nid << ":";
            //     cout << nodes[nid].invalid << ",";
            //     cout << "args:";
            //     for (auto arg: nodes[nid].args) cout << arg << ",";
            //     cout << "succ_cnt:" << nodes[nid].succ_cnt << endl;
            // }
        }

        void Env::reset_typewise()
        {
            for (int nid = num_nodes_evaluated; nid <= (int)upto; ++nid)
                nodes[nid].typewise_succ_cnt = 0;
            for (auto &type : types)
                type.second.typewise_frontier_cnt = 0;
            for (int nid = upto; nid >= num_nodes_evaluated; --nid)
            {
                auto &node = nodes[nid];
                if (node.invalid)
                    continue;
                for (auto arg : node.typewise_args)
                    nodes[arg].typewise_succ_cnt++;
                if (node.typewise_succ_cnt == 0)
                {
                    types[node.type].typewise_frontier_cnt++;
                }
            }
        }

        bool Env::done()
        {
            bool is_done = true;
            for (auto &kv : types)
            {
                is_done = is_done && !kv.second.frontiers.size();
            }
            return is_done;
        }

        void Env::get_state(state_t &state)
        {
            for (auto &kv : types)
            {
                if (kv.second.frontiers.size())
                    state.insert(kv.first);
            }
        }

        void Env::get_memory_info(
            const std::vector<std::vector<VariableIndex>> &batches, // reversed batches
            const std::vector<bool> &pre_allocs,
            std::vector<std::pair<double, double>> &infos)
        {
            infos.resize(batches.size());
            vector<int> memid(nodes.size(), -1);
            int ptr = 0;
            for (auto &info : infos)
                info.first = info.second = 0;
            for (int i = batches.size() - 1; i >= 0; --i)
            {
                auto &batch = batches[i];
                if (pre_allocs[i])
                {
                    auto &node = nodes[batch.front()];
                    for (int aid = 0; aid < node.args.size(); aid++)
                    {
                        for (auto nid : batch)
                        {
                            int arg = nodes[nid].args[aid];
                            if (memid[arg] < 0)
                                memid[arg] = ptr++;
                        }
                    }
                }
            }

            int bid = 0;
            for (auto &batch : batches)
            {
                for (auto nid : batch)
                {
                    if (memid[nid] < 0)
                        memid[nid] = ptr;
                    ptr++;
                }
                bid++;
            }

            bid = 0;
            for (auto &batch : batches)
            {
                // scatter
                {
                    int idx = memid[batch.front()];
                    auto iter = batch.begin();
                    while (iter != batch.end() && idx == memid[*iter])
                        idx++, iter++;
                    if (iter != batch.end())
                        infos[bid].second += batch.size() / (SCALE) + KERNEL_LAUNCH_OVERHEAD;
                }

                // combine
                auto &examplar = nodes[batch.front()];
                for (int aid = 0; aid < examplar.args.size(); aid++)
                {
                    int idx = memid[examplar.args[aid]];
                    auto iter = batch.begin();
                    while (iter != batch.end() && idx == memid[nodes[*iter].args[aid]])
                    {
                        idx++, iter++;
                    }
                    if (iter != batch.end())
                        infos[bid].first += 0; // batch.size() / (SCALE * 2) + KERNEL_LAUNCH_OVERHEAD;
                }
                bid++;
            }
        }
        double RLEnv::get_reward()
        {
            if (history.size() <= n_step)
                history.resize(n_step + 1);
            auto &entry = history[n_step];
            entry.n += 1;
            entry.ex = entry.ex + 1.0 / entry.n * (n_node - entry.ex);
            entry.ex2 = entry.ex2 + 1.0 / entry.n * (n_node * n_node - entry.ex2);
            return (n_node - entry.ex) / std::sqrt(entry.ex2 - entry.ex * entry.ex + 1e-6);
        }

        void RLEnv::reset()
        {
            Env::reset();
            n_node = 0.0;
            n_step = -1;
        }

        double RLEnv::step(int &type, vector<VariableIndex> &batch)
        {
            double best_reward = -1, best_type;
            if (mode == TYPEWISE)
            {
                for (auto &kv : types)
                {
                    if (!kv.second.frontiers.size())
                        continue;
                    double reward = kv.second.frontiers.size() / (kv.second.typewise_frontier_cnt);
                    if (reward > best_reward)
                    {
                        best_reward = reward;
                        best_type = kv.first;
                    }
                }
                if (type == 0)
                {
                    type = best_type;
                }
                state_t state;
                get_state(state);
                best_actions[state].insert(type);
            }

            if (!types[type].frontiers.size())
                throw runtime_error("bad policy");

            double reward = -1;
            batch = move(types[type].frontiers);
            state.erase(type);
            n_node += batch.size();
            n_step += 1;
            if (mode == TYPEWISE)
            {
                auto &this_type = types[type];
                reward += theta * batch.size() / (this_type.typewise_frontier_cnt + 0.1);
                this_type.typewise_frontier_cnt -= batch.size();
            }

            // auto& autobatch_concat = cg->nodes[batch.front()]->autobatch_concat(*cg);
            int narg = nodes[batch.front()].args.size();
            for (int aid = 0; aid < narg; ++aid)
            {
                for (auto &nid : batch)
                {
                    int arg = nodes[nid].args[aid];
                    if (!nodes[arg].invalid && --nodes[arg].succ_cnt == 0)
                    {
                        types[nodes[arg].type].frontiers.push_back(arg);
                        state.insert(nodes[arg].type);
                    }
                }
            }

            if (mode == TYPEWISE)
            {
                for (auto &nid : batch)
                {
                    auto &node = nodes[nid];
                    for (auto arg : node.typewise_args)
                    {
                        if (!nodes[arg].invalid && --nodes[arg].typewise_succ_cnt == 0)
                            types[nodes[arg].type].typewise_frontier_cnt++;
                    }
                }
            }
            reward += (1 - theta) * get_reward();
            return reward;
        }

        void DBLearner::train(Env *env, int n_iter)
        {
            cout << "training!" << endl;
            if (env->mode == Env::TYPEWISE)
                cout << "typewise batch: " << typewise_inference(env) << endl;
            vector<VariableIndex> batch;
            int min_step = 1e9;
            OoC::TupleDict<q_table_t> best_q_table;
            for (int i = 0; i < n_iter; i++)
            {
                env->reset();
                state_t state, next_state;
                env->get_state(state);
                int step = 0;
                while (true)
                {
                    if (!state.size())
                        break;
                    if (step >= n_step)
                    {
                        double g = q_tables[state].max();
                        for (int j = 0; j < n_step; ++j)
                        {
                            g = g * gamma + buffer.get(j).reward;
                        }
                        auto &q = q_tables[buffer.get(n_step - 1).state].q[buffer.get(n_step - 1).action];
                        q += alpha * (g - q);
                    }
                    int action = take_action(state, true);
                    // assert(state.count(action));
                    double reward = env->step(action, batch);
                    assert(next_state.size() == 0);
                    env->get_state(next_state);
                    buffer.add({move(state), action, reward, next_state});
                    assert(state.size() == 0);
                    state = move(next_state);
                    step++;
                }
                epsilon *= epsilon > epsilon_lb ? 1 : epsilon_decay;

                double g = -1;
                for (int j = 0; j < std::min(step, n_step); ++j)
                {
                    auto &entry = buffer.get(j);
                    g = g * gamma + entry.reward;
                    auto &q = q_tables[entry.state].q[entry.action];
                    q += alpha * (g - q);
                }

                if ((i + 1) % 100 == 0)
                {
                    int step = inference(env) + 1;
                    cout << "iter " << i + 1 << ",n_batch:" << step << endl;
                    if (step < min_step)
                    {
                        min_step = step;
                        best_q_table = q_tables;
                    }
                }
            }
            q_tables = best_q_table;
            cout << "Train Finished! #Batch:" << inference(env) + 1 << endl;
            type_ub = env->get_largest_type();
            // function<void(set<int>*s)> func = [](set<int>* s){
            //     for (auto e: *s) cout << e << ",";
            // };
            // env->best_actions.report(func);
        }

        int DBLearner::inference(Env *env)
        {
            env->reset();
            int n_step = 0;
            state_t state;
            env->get_state(state);
            while (state.size())
            {
                n_step++;
                int action;
                action = take_action(state);
                vector<VariableIndex> batch;
                env->step(action, batch);
                state.clear();
                env->get_state(state);
            }
            return n_step;
        }

        int DBLearner::typewise_inference(Env *env)
        {
            if (!env->mode == Env::TYPEWISE)
                return 1e9;
            env->reset();
            state_t state;
            env->get_state(state);
            while (state.size())
            {
                n_step++;
                vector<VariableIndex> batch;
                int action = 0;
                env->step(action, batch);
                state.clear();
                env->get_state(state);
            }
            return n_step;
        }

        int DBLearner::take_action(const state_t &state, bool train)
        {
            assert(state.size());
            if (*state.begin() < 0)
            {
                auto act = *state.begin();
                q_tables[state].q[act] = 1.0;
                return act;
            }
            if (train)
            {
                if ((random() / (RAND_MAX + 0.0)) < epsilon)
                {
                    int i = random() % state.size();
                    auto iter = state.begin();
                    while (i)
                    {
                        --i, ++iter;
                    }
                    return *iter;
                }
                // return 0;
            }
            auto &q_table = q_tables[state];
            if (q_table.size() == 0)
            {
                q_table.init(state);
            }
            return q_table.argmax();
        }

        void PreMallocLearner::train(Env *env, int n_iter)
        {
            cout << "[PreMalloc]: Training!" << endl;
            vector<vector<VariableIndex>> batches;
            vector<bool> pre_allocs;
            vector<int> actions;

            std::unordered_map<int, q_table_t> best_q_tables;
            double min_memtransfer = 1e9;
            for (int i = 0; i < n_iter; ++i)
            {
                env->reset();
                pre_allocs.clear();
                batches.clear();
                actions.clear();
                while (true)
                {
                    state_t state = env->incremental_get_state();
                    if (!state.size())
                        break;
                    int action = policy.take_action(state);
                    actions.push_back(action);
                    pre_allocs.push_back(take_action(action, true));
                    batches.push_back({});
                    env->step(action, batches.back());
                }
                epsilon *= epsilon > epsilon_lb ? 1 : epsilon_decay;
                vector<pair<double, double>> mem_infos;
                env->get_memory_info(batches, pre_allocs, mem_infos);

                for (int bid = 0; bid < (int)batches.size(); bid++)
                {
                    auto &action = actions[bid];
                    auto &reward = mem_infos[bid];
                    auto pre_alloc = pre_allocs[bid];
                    int i = std::min(bid + n_step, (int)batches.size());
                    double g = (i == (int)batches.size()) ? 0 : q_tables[i + 1].max();
                    for (; i > bid; --i)
                    {
                        auto &reward = mem_infos[i - 1];
                        g = g * gamma - (reward.first + reward.second);
                    }
                    auto &q = q_tables[action].q[pre_alloc];
                    q += alpha * (g - q);
                }
                if ((i + 1) % 100 == 0)
                {
                    double n_memtransfer = inference(env);
                    if (n_memtransfer < min_memtransfer)
                    {
                        min_memtransfer = n_memtransfer;
                        best_q_tables = q_tables;
                    }
                    cout << "iter " << i + 1 << ",n_memcpy:" << inference(env) << endl;
                }
            }
            q_tables = best_q_tables;
            cout << "after training: " << inference(env) << endl;
            // for (auto &kv : q_tables)
            // {
            //     cout << "(" << nt::type2name[env->sigmap.sig2type(kv.first)] << "," << kv.second.argmax() << "),";
            //     cout << endl;
            // }
        }

        double PreMallocLearner::inference(Env *env)
        {
            vector<vector<VariableIndex>> batches;
            vector<bool> pre_allocs;
            env->reset();
            while (true)
            {
                state_t state = env->incremental_get_state();
                if (!state.size())
                    break;
                int action = policy.take_action(state);
                pre_allocs.push_back(take_action(action));
                batches.push_back({});
                env->step(action, batches.back());
            }
            epsilon *= epsilon > epsilon_lb ? 1 : epsilon_decay;
            vector<pair<double, double>> mem_infos;
            env->get_memory_info(batches, pre_allocs, mem_infos);
            double n_memtransfer = 0;
            for (auto &x : mem_infos)
                n_memtransfer += x.first + x.second;
            return n_memtransfer;
        }

        bool PreMallocLearner::take_action(int state, bool train)
        {
            if (q_tables.count(state) == 0)
                q_tables[state].init(std::vector<int>({0, 1}), -10);
            if (train && (random() / (RAND_MAX + 0.0) < epsilon))
            {
                return random() >= (RAND_MAX >> 1);
            }
            return q_tables[state].argmax();
        }

    } // namespace RL
} // namespace OoC

namespace dynet
{
    void BatchedExecutionEngine::getBatches_rl(
        VariableIndex upto,
        VariableIndex &batch_id)
    {
        static RL::DBLearner db_learner(1000);
        static RL::PreMallocLearner mem_learner(db_learner);
        global_timer.start("RL::preprocess");
        RL::RLEnv *env;

        if (autobatch_flag == 9 || db_learner.trained)
            env = new RL::RLEnv(cg, upto, num_nodes_evaluated, RL::Env::BASIC, sigmap);
        else
            env = new RL::RLEnv(cg, upto, num_nodes_evaluated, RL::Env::TYPEWISE, sigmap);
        global_timer.stop("RL::preprocess");

        if (!db_learner.trained)
        {
            db_learner.train(env);
            db_learner.trained = true;
            env->set_mode(RL::Env::BASIC);
            mem_learner.train(env);
        }

        global_timer.start("RL::batching");
        env->reset();
        vector<vector<VariableIndex>> my_batches;
        vector<bool> pre_mallocs;
        int n_step = 0, n_miss = 0;
        while (true)
        {
            OoC::RL::state_t state = env->incremental_get_state();
            // env->get_state(state);
            if (!state.size())
                break;
            n_step++;
            int action;
            if (*state.rbegin() > db_learner.type_ub)
            {
                n_miss++;
                action = *state.rbegin();
            }
            else
                action = db_learner.take_action(state);
            pre_mallocs.push_back(mem_learner.take_action(action));
            my_batches.push_back({});
            env->step(action, my_batches.back());
        }
        cerr << "[scheduling] miss_rate:" << n_miss << "/" << n_step << endl;
        global_timer.stop("RL::batching");

        global_timer.start("RL::postprocess");
        auto pre_alloc_iter = pre_mallocs.rbegin();
        for (auto iter = my_batches.rbegin();
             iter != my_batches.rend(); iter++, pre_alloc_iter++)
        {
            auto &my_batch = batches[batch_id];
            for (auto id : *iter)
            {
                node2batch[id] = batch_id;
            }
            my_batch.ids = move(*iter);

            if (dynet::autobatch_flag == 9)
                my_batch.pre_alloc = cg.nodes[my_batch.ids.front()]->is_function;
            else
                my_batch.pre_alloc = *pre_alloc_iter;

            batch_id++;
            auto examplar = cg.nodes[my_batch.ids.front()];
            if (examplar->is_function)
            {
                FunctorNode *node = static_cast<FunctorNode *>(examplar);
                for (int i = 1; i <= node->n_output; ++i)
                {
                    vector<VariableIndex> get_node_batch(my_batch.ids);
                    for (auto &nid : get_node_batch)
                    {
                        nid += i;
                        node2batch[nid] = batch_id;
                    }
                    batches[batch_id++].ids = move(get_node_batch);
                }
            }
        }
        global_timer.stop("RL::postprocess");

        delete env;

        return;
    }

} // namespace dynet