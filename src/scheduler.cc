#include "OoC.h"
#include <tuple>
#include <cstring>
#include <thread>

#define PRECISION (100)
using namespace std;

namespace OoC
{

    int Scheduler::lower_bound(){
        reset();
        vector<int> per_type_max_depth(snodes.size() * stypes.size(), 0);
        vector<int> max_depths(stypes.size(), 0);
        for (int sid = 0; sid < snodes.size(); sid++){
            auto & snode = snodes[sid];
            int this_tid = snode.type;
            for (int tid = 0; tid < stypes.size(); tid++){
                auto & depth = per_type_max_depth[sid * stypes.size() + tid];
                depth = (this_tid == tid);
                for (auto succ: snodes[sid].succs){
                    depth = max(depth, (this_tid == tid) + per_type_max_depth[succ * stypes.size() + tid]);
                }
                max_depths[tid] = max(max_depths[tid], depth);
            }
        }
        int ret = 0;
        for (int i = 0; i < stypes.size(); i++){
            ret += max_depths[i];
        }
        return ret;
    }

    void Scheduler::validate(std::string filename){
        prepare_data_from_file(filename);
        reset();
        n_remain = n_node;
        n_batch = 0;
        int iter = 0;
        while (n_remain){
            set<int> state;
            for (auto &kv: stypes){
                if (kv.second.frontiers.size())
                    state.insert(kv.first);
            }
            int act = get_action(state);
            fprintf(stdout, "[validate::iter%d]: %d act\n", iter++, act);
            commit(act);
        }
        fprintf(stdout, "[scheduler::validation](%s): %d batches\n", filename.c_str(), n_batch);
    }

    void Scheduler::prepare_data_from_file(std::string filename){
        _snodes.clear();
        _stypes.clear();
        ifstream file;
        file.open(filename);
        file >> n_node;
        vector<int> depth(n_node, 0);
        _snodes.resize(n_node);
        max_depth = 0;
        for (int i = 0; i < n_node; i++)
        {
            max_depth = max(max_depth, depth[i]);
            int stid, n_succ;
            file >> stid >> n_succ;
            if (_stypes.count(stid) == 0)
                _stypes[stid] = {};
            auto &stype = _stypes[stid];
            stype.cnt += 1;
            auto &snode = _snodes[i];
            if (snode.inputCnt == 0)
                stype.frontiers.push_back(i);
            snode.type = stid;
            for (int j = 0; j < n_succ; j++)
            {
                int tmp;
                file >> tmp;
                assert(tmp < n_node);
                snode.succs.push_back(tmp);
                _snodes[tmp].inputCnt += 1;
                depth[tmp] = max(depth[tmp], depth[i] + 1);
            }
        }
        file.close();
        max_depth += 1;
    }

    void Scheduler::show_frontier()
    {
        if (verbose <= 0)
            return;
        fprintf(stdout, "-----------show frontier---------------\n");
        
        for (int nid = 0; nid < snodes.size(); nid++){
            fprintf(stdout, "\tn%d, %d: ", nid, snodes[nid].inputCnt);
            for (auto succ: snodes[nid].succs) fprintf(stdout, "%d, ", succ);
            fprintf(stdout, "\n");
        }
        fprintf(stdout, "[frontier] n_node %d n_batch %d n_remain %d: \n", 
           n_node, n_batch, n_remain);
        bool stopped = true;
        for (auto &kv : stypes)
        {
            if (kv.second.frontiers.size() == 0) continue;
            stopped = false;
            fprintf(stdout, "\t%d: ", kv.first);
            for (auto &ele : kv.second.frontiers)
            {
                fprintf(stdout, "%d, ", ele);
            }
            fprintf(stdout, "\n");
        }
        if (stopped) fprintf(stdout, "\t[frontier] no frontier node found!\n");
        fprintf(stdout, "-----------show frontier end---------------\n");
    }

    void Scheduler::visualize(string filename)
    {
        ofstream file;
        file.open(filename);
        int sid = 0;
        file << "digraph G{" << endl;
        function<string(int)> get_name = [&](int id)
        {
            auto ret = "N_" + to_string(id) + "_" + to_string(snodes[id].type);
            return ret;
        };
        for (auto &snode : snodes)
        {
            if (commited.count(sid))
            {
                sid++;
                continue;
            }
            auto this_name = get_name(sid);
            for (auto succ : snode.succs)
            {
                auto that_name = get_name(succ);
                file << this_name << " -> " << that_name << endl;
            }
            sid++;
        }
        file << "}" << endl;
        file.close();
    }

    void Scheduler::train(string filename, int n_trial)
    {
        prepare_data_from_file(filename);
        batch_train(n_trial);
    }

    int Scheduler::train(const std::vector<supernodeInfo*> &snodes, const std::vector<typeInfo> &stypes, int n_trial)
    {
        _snodes.clear();
        _stypes.clear();
        for (auto &snode : snodes)
        {
            _snodes.push_back({});
            auto &_snode = _snodes.back();
            _snode.inputCnt = snode->inputCnt;
            _snode.succs = snode->succs;
            _snode.type = snode->type;
        }
        int n_frontier = 0;
        for (int stid = 0; stid < stypes.size(); stid++){
            _stypes[stid] = stypes[stid];
            n_frontier += stypes[stid].frontiers.size();
        }
        assert(n_frontier);
        n_node = _snodes.size();
        int n_batch = batch_train(n_trial);
        if (verbose > -1){  
            int lb = lower_bound();  
            fprintf(stdout, "[OoC::train]: lower_bound %d, find %d, alpha %f\n", lb, n_batch, n_batch / (0.0 + lb));
            ofstream file;
            file.open("./graph/snode_validation.txt", std::ios::out | std::ios::app);
            file << lb << " " << n_batch << endl;
            file.close();
        }
        return n_batch;
    }

    int Scheduler::train(const std::vector<nodeInfo>& __snodes, const std::unordered_map<int, typeInfo>& __stypes){
        _snodes = __snodes;
        _stypes = __stypes;
        n_node = _snodes.size();
        return train();
    }

    void Scheduler::reset()
    {
        stypes = _stypes;
        snodes = _snodes;
        commited.clear();
    }

    // A, pi(A|S), \Sigma_{a != A} pi(a|S) * Q(S, a),
    int QLearningModel::State::take_action(
        bool inference)
    {
        double p0 = rand() / (RAND_MAX + 0.0);
        if (inference || p0 < 0.1){
            int ret = Q.begin()->first;
            for (auto& kv: Q){
                if (kv.second > Q[ret])
                    ret = kv.first;
            }
            return ret;
        }

        double max_prob = 0;
        double sum = 0;
        for(auto& kv: Q){
            sum += std::exp(kv.second);
            max_prob = max(std::exp(kv.second), max_prob);
        }
        double p = rand() / (RAND_MAX + 0.0) * sum;
        double acc = 0;
        for (auto& kv: Q){
            acc +=  std::exp(kv.second);
            if (acc >= p)
                return kv.first;
        }
        fprintf(stdout, "Q.size() %ld, p %f, acc %f, sum %f\n", Q.size(), p, acc, sum);
        show();
        model->show_frontier();
        assert(false);
        return -1;
    }

    void QLearningModel::State::show()
    {
        fprintf(stdout, "%d: ", model->state2id[this]);
        for (auto& kv : Q)
            fprintf(stdout, "(%d, %f), ", kv.first, kv.second);
    }

    int Scheduler::commit(int tid)
    {
        if (stypes.count(tid) == 0)
        {
            show_frontier();
            fprintf(stdout, "tid %d, stypes.size %ld\n", tid, stypes.size());
            assert(false);
        }
        auto &frontierType = stypes[tid];
        vector<int> snode_batch = move(frontierType.frontiers);
        n_remain -= snode_batch.size();
        frontierType.cnt -= snode_batch.size();
        n_batch++;
        for (auto nid : snode_batch)
        {
            commited.insert(nid);
            auto &snode = snodes[nid];
            for (auto &succ : snode.succs)
            {
                auto &succNode = snodes[succ];
                assert(stypes.count(succNode.type));
                if (--succNode.inputCnt == 0)
                    stypes[succNode.type].frontiers.push_back(succ);
            }
        }
        // fprintf(stdout, "after commit %d\n", tid);
        return snode_batch.size();
    }

    void Scheduler::reversed_commit(int tid, const vector<int> &snode_batch)
    {
        for (auto iter = snode_batch.rbegin(); iter != snode_batch.rend(); iter++)
        {
            auto &snode = snodes[*iter];
            commited.erase(*iter);
            for (auto it = snode.succs.rbegin(); it != snode.succs.rend(); it++)
            {
                auto &succNode = snodes[*it];
                if (succNode.inputCnt++ == 0)
                    stypes[succNode.type].frontiers.pop_back();
            }
        }
        auto &frontierType = stypes[tid];
        assert(frontierType.frontiers.size() == 0);
        n_remain += snode_batch.size();
        n_batch--;
        frontierType.cnt += snode_batch.size();
        frontierType.frontiers = snode_batch;
    }

    QLearningModel::State *QLearningModel::get_curr_state(bool inference)
    {
        set<int> key;

        for (auto &kv : stypes)
        {
            if (kv.second.frontiers.size())
                key.insert(kv.first);
        }

        if (states.count(key) == 0){
            auto & state = states[key];
            assert(state.Q.size() == 0);
            state2id[&state] = n_state++;
            for (auto act : key)
                state.Q[act] = q_init;
            if (inference) {
                fprintf(stdout, "[WARNING]: new state when inference: ");
                state.show();
                fprintf(stdout, "\n");
            }
        }

        return &states[key];
    }

    int QLearningModel::step(bool inference)
    {
        struct Log
        {
            State *state = nullptr;
            int action = -1;
            double reward = 0;
            // double isr = 0; // important sampling ratio;
        };

        list<Log> memo;
        static int n_iter = 0;
        n_iter++;

        n_batch = 0;
        n_remain = n_node;
        State *curr = get_curr_state(inference);
        env.get_reward(n_batch, n_node);
        while (n_remain)
        {
            if (!curr->Q.size()){
                show_frontier();
                assert(false);
            }
            int action =  curr->take_action(inference);
            commit(action);
            double reward = env.get_reward(n_batch, n_remain);
            if (!inference && (int)memo.size() == n_step)
            {
                double G = 0;
                double param = 1;
                auto iter = memo.begin();
                while (iter != memo.end())
                {
                    G += param * iter->reward;
                    iter++, param *= _gamma;
                }
                G += param * (curr->Q[action]);
                auto &log = memo.front();
                log.state->Q[log.action] += alpha * std::exp(-std::sqrt(n_batch)) *  (G - log.state->Q[log.action]);
                memo.pop_front();
            }
            memo.push_back({curr, action, reward});
            // fprintf(stdout, "\tn_batch %d, n_remain %d, state %d, action %d, reward %f\n", n_batch, n_remain, state2id[curr], action, reward);

            curr = get_curr_state(inference);
        }

        if (!inference){
            while (!memo.empty())
            {
                double G = 0;
                double param = 1;
                auto iter = memo.begin();
                while (iter != memo.end())
                {
                    G += param * iter->reward;
                    iter++, param *= _gamma;
                }
                auto &log = memo.front();
                log.state->Q[log.action] += alpha * std::exp(-std::sqrt(n_batch)) *  (G - log.state->Q[log.action]);
                memo.pop_front();
            }
        }

        return n_batch;
    }

    int QLearningModel::train()
    {
        states.clear();
        env.reset();
        n_state = 0;
        reset();
        if (verbose){
            show_frontier();
            visualize("./graph/G.gv");
        }
        for (int iter = 0; iter < n_train_iter; iter++)
        {
            int n_batch = step(false);
            reset();
            // fprintf(stdout, "[q_learning_schedule::train] iter %d: %d batch\n", iter, n_batch);
        }

        int n_batch = step();
        // env.show();
        // show_states();
        
        fprintf(stdout, "[q_learning_schedule::validation]: %d batch\n", n_batch);
        fprintf(stdout, "n_train_iter %d, n_step %d, alpha %f, gamma %f, q_init %f, require_rho %d\n",
                n_train_iter, n_step, alpha, _gamma, q_init, require_rho);
        return n_batch;
    }

    void QLearningModel::show_states(){
        fprintf(stdout, "---------------show states begin---------------\n");
        for (auto ptr : states.data)
        {
            if (ptr) {
                ptr->show();
                fprintf(stdout, "\n");
            }
        }
    }

    int QLearningModel::get_action(const set<int> &state)
    {
        State * ptr;
        if ((ptr = states.get(state)))
            return ptr->take_action(true);
        fprintf(stdout, "policy missed!\n");
        return *state.rbegin();
    }
    
    QLearningModel *QLearningModel::State::model = nullptr;

    int QLearningModel::batch_train(int batch_size){
        if (verbose > -1){
            reset();
            visualize("./graph/G.gv");
        }
        TupleDict<State> buffer;
        int best_n_batch = 1e6;
        for (int i = 0; i < batch_size; i ++){
            int n_batch = train();
            fprintf(stdout, "[batch_train::iter %d]: %d batches\n", i, n_batch);
            // show_states();
            if (n_batch < best_n_batch){
                // fprintf(stdout, "[batch_train::iter %d]: update the states %d ---> %d!\n", i, best_n_batch, n_batch);
                best_n_batch = n_batch;
                buffer = move(states);
            }
        }
        states = move(buffer);
        reset();
        // visualize("./graph/G.gv");
        int n_batch = step(true);
        fprintf(stdout, "[batch_train::validation]: %d batches\n", n_batch);
        // show_states();
        assert(n_batch == best_n_batch);
        return best_n_batch;
    }

    int ParallelQLearningModel::batch_train(int batch_size){
        vector<future<int> > results;
        vector<unique_ptr<Scheduler>> models;
        for (int i = 0; i < batch_size; i++){
            models.push_back(make_unique<QLearningModel>(
                n_train_iter, n_step, alpha, _gamma, q_init, require_rho));
            Scheduler * model = models.back().get();
            results.emplace_back(async([model, this]()->int{
                fprintf(stdout, "[OoC::QLearning]: start training at %x\n", this_thread::get_id());
                return model->train(this->_snodes, this->_stypes);
            }));
        }
        int min_n_batch = 1e6;
        int min_idx = 0;
        for (int i = 0; i < results.size(); i++){
            int n_batch = results[i].get();
            if (min_n_batch > n_batch){
                min_n_batch = n_batch;
                min_idx = i;
            }
        }
        _model = move(models[min_idx]);
        return min_n_batch;
    }

    int ParallelQLearningModel::get_action(const std::set<int> &state){
        return _model->get_action(state);
    }

} // namespace OoC