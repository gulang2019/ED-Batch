#include <functional>
#include <random>
#include <chrono>
#include "batching.h"

using namespace std;
using namespace dynet;
#define LIMIT 3
#define BATCHLIMIT 5
/*
enum NodeType {
      tanh=1, sqrt, abs, erf, square, cube, exp, logsigmoid, loggamma, log, nobackprop, scalegradient, identity, negate, rectify, logistic, softsign, silu, round, ceiling, floor,
      sinh, cosh, asinh, acosh, atanh, sin, cos, tan, asin, acos, atan, plus_const, concat, cmult, csum, sum, squared_distance, softmax, pnls, pickrange, scalar_mult, dropout,
      input, scalar_input, lookup,
      COMPLEX,
      affine, matmul, transpose,
      vanilla_lstm_gates, vanilla_lstm_h, vanilla_lstm_c,
      conv2d
    };
*/
unordered_map<int, string> type2name = {
    {0, "unbatchable"},
    {nt::tanh, "tanh"},
    {nt::sqrt, "sqrt"},
    {nt::abs, "abs"},
    {nt::erf, "erf"},
    {nt::square, "square"},
    {nt::cube, "cube"},
    {nt::exp, "exp"},
    {nt::logsigmoid, "logsigmoid"},
    {nt::loggamma, "loggamma"},
    {nt::log, "log"},
    {nt::nobackprop, "nobackprop"},
    {nt::scalegradient, "scalegradient"},
    {nt::identity, "identity"},
    {nt::negate, "negate"},
    {nt::rectify, "rectify"},
    {nt::logistic, "logistic"},
    {nt::softsign, "softsign"},
    {nt::silu, "silu"},
    {nt::round, "round"},
    {nt::ceiling, "ceiling"},
    {nt::floor, "floor"},
    {nt::sinh, "sinh"},
    {nt::cosh, "cosh"},
    {nt::asinh, "asinh"},
    {nt::acosh, "acosh"},
    {nt::atanh, "atanh"},
    {nt::sin, "sin"},
    {nt::cos, "cos"},
    {nt::tan, "tan"},
    {nt::asin, "asin"},
    {nt::acos, "acos"},
    {nt::atan, "atan"},
    {nt::plus_const, "plus_const"},
    {nt::concat, "concat"},
    {nt::cmult, "cmult"},
    {nt::csum, "csum"},
    {nt::sum, "sum"},
    {nt::squared_distance, "squared_distance"},
    {nt::softmax, "softmax"},
    {nt::pnls, "pnls"},
    {nt::pickrange, "pickrange"},
    {nt::scalar_mult, "scalar_mult"},
    {nt::dropout, "dropout"},
    {nt::input, "input"},
    {nt::scalar_input, "scalar_input"},
    {nt::lookup, "lookup"},
    {nt::COMPLEX, "COMPLEX"},
    {nt::affine, "affine"},
    {nt::matmul, "matmul"},
    {nt::transpose, "transpose"},
    {nt::vanilla_lstm_gates, "vanilla_lstm_gates"},
    {nt::vanilla_lstm_h, "vanilla_lstm_h"},
    {nt::vanilla_lstm_c, "vanilla_lstm_c"},
    {nt::conv2d, "conv2d"}
};


void draw_graph(string filename, vector<vector<int>> &g, vector<int> &node2type, unordered_set<int> &nodes, string graphName)
{
    ofstream file;
    file.open(filename);
    // lookup, logistic, gemm, add, tanh, cmult, concatenate

    file << "digraph " << graphName << " {\n";
    for (size_t node_id = 0; node_id < g.size(); node_id++)
    {
        if (nodes.count(node_id) == 0)
            continue;
        string nodetag = type2name[(op_type)node2type[node_id]] + "_" + std::to_string(node_id);
        for (auto to : g[node_id])
        {
            if (nodes.count(to) == 0)
                continue;
            string totag = type2name[(op_type)node2type[to]] + "_" + std::to_string(to);
            file << "\t" << nodetag << " -> " << totag << ";\n";
        }
    }
    file << "}\n";
    file.close();
}


void draw_graph(string filename, vector<vector<int>> &g, vector<int> &node2type)
{
    ofstream file;
    file.open(filename);
    // lookup, logistic, gemm, add, tanh, cmult, concatenate

    file << "digraph G{\n";
    for (size_t node_id = 0; node_id < g.size(); node_id++)
    {
        string nodetag = type2name[(op_type)node2type[node_id]] + "_" + std::to_string(node_id);
        for (auto to : g[node_id])
        {
            string totag = type2name[(op_type)node2type[to]] + "_" + std::to_string(to);
            file << "\t" << nodetag << " -> " << totag << ";\n";
        }
    }
    file << "}\n";
    file.close();
}

struct hash_pair
{
    template <class T1, class T2>
    size_t operator()(const pair<T1, T2> &p) const
    {
        auto hash1 = hash<T1>{}(p.first);
        auto hash2 = hash<T2>{}(p.second);

        if (hash1 != hash2)
        {
            return hash1 ^ hash2;
        }

        // If hash1 == hash2, their XOR is zero.
        return hash1;
    }
};

DynamicBatching::DynamicBatching(vector<vector<int>>& g_in, vector<int>& node2type_in, bool deleteUnbatachable) : localTimer("DEFAULT"), addTimer("ADD"), node_id((int)g_in.size()), faketype_id(0)
{
    for (int nid = 0; nid < (int)g_in.size(); nid++){
        nodes.insert(nid);
        topo_value[nid] = nid;
    }

    g.resize(g_in.size(), {});
    g_r.resize(g_in.size(), {});
    for (auto node : nodes)
    {
        for (auto to : g_in[node]){
            assert(to > node);
            g[node].push_back(to);
            g_r[to].push_back(node);
        }
    }

    for (auto t : node2type_in)
        node2type.push_back(t);

    for (auto node : node2type)
        type2weight[node2type[node]] = 1;

    draw_graph("G.gv", {&nodes}, "G");

    
    if(deleteUnbatachable){
        unordered_map<int, vector<int> > childrenOfUnbatchable;
        for (size_t nid = 0; nid < g_in.size(); nid++){
            if (node2type[nid] == 0){
                childrenOfUnbatchable[nid] = vector<int>();
                auto & children = childrenOfUnbatchable[nid];
                for (auto from: g_r[nid]){
                    if (childrenOfUnbatchable.count(from)){
                        children.insert(children.end(), childrenOfUnbatchable[from].begin(), childrenOfUnbatchable[from].end());
                    }
                    else children.push_back(from);
                }
                nodes.erase(nid);
            }
            else {
                for (auto from: g_r[nid]){
                    // printf("from %d\n", from);
                    if (node2type[from] == 0){
                        assert(childrenOfUnbatchable.count(from));
                        g_r[nid].insert(g_r[nid].end(), childrenOfUnbatchable[from].begin(), childrenOfUnbatchable[from].end());
                        for (auto child: childrenOfUnbatchable[from]){
                            g[child].push_back(nid);
                        }
                    }
                }
            }
        }
        if (profiling_flag > 1){
            fprintf(stderr, "Unbatchable: %ld, Batchable: %ld\n", childrenOfUnbatchable.size(), nodes.size());
        }
    }
    transitiveReduction(nodes);
    draw_graph("G0.gv", {&nodes}, "G0");
}

void cntHash(vector<string> &values, string &hash)
{
    sort(values.begin(), values.end(), [](string &s1, string &s2)
         {
        if (s1.size() == s2.size()) 
            return s1 < s2;
        else return s1.size() < s2.size(); });
    size_t i, lasti;
    i = lasti = 0;
    while (i < values.size())
    {
        while (i < values.size() && values[lasti] == values[i])
            i++;
        hash += values[lasti] + to_string(i - lasti) + ",";
        lasti = i;
    }
    hash += "|";
}

void DynamicBatching::graphHash(const unordered_set<int> &lower_nodes, const unordered_set<int> &upper_nodes, string &hash)
{
    hash = "";

    vector<string> values;
    for (auto &vec : {lower_nodes, upper_nodes})
    {
        values.clear();
        for (auto node : vec)
        {
            values.push_back(type2name[node2type[node]]);
        }
        cntHash(values, hash);
    }

    values.clear();
    for (auto from : lower_nodes)
    {
        for (auto to : g[from])
        {
            if (upper_nodes.count(to) == 0)
                continue;
            values.push_back(type2name[node2type[from]] + "-" + type2name[node2type[to]]);
        }
    }
    cntHash(values, hash);
}

inline double sample_beta_distribution(double alpha, double beta)
{
    static default_random_engine generator;
    gamma_distribution<double> da(alpha, 1.0);
    gamma_distribution<double> db(beta, 1.0);
    double x = da(generator);
    double y = db(generator);
    return x / (x + y);
}

void DynamicBatching::getContractSubgraphs(std::vector<std::unordered_set<int>> &contract_subgraphs)
{
    addTimer.start("getContractSubgraphs");
    // 1. find isomorphic subgraphs
    unordered_map<int, int> typeCnt;
    for (auto node : nodes)
    {
        int type = node2type[node];
        if (typeCnt.count(type) == 0)
            typeCnt[type] = {};
        typeCnt[type]++;
    }

    localTimer.start("classify nodes");
    vector<unordered_set<int>> subgraphs;
    vector<double> scores;
    vector<bool> visited(node_id, false);
    for (auto from : nodes)
    {
        if (visited[from])
            continue;
        unordered_set<int> subgraph;
        double score = 0;
        unordered_set<int> visited_to;
        function<void(int)> dfs = [&](int node)
        {
            if (nodes.count(node) == 0 || visited[node])
                return;
            visited[node] = true;
            subgraph.insert(node);
            score += 0.5 / typeCnt[node2type[node]];
            for (auto to : g[node])
            {
                if (nodes.count(to) == 0 || visited_to.count(to))
                    continue;
                visited_to.insert(to);
                for (auto newNode : g_r[to])
                {
                    if (nodes.count(node) == 0)
                        continue;
                    score += 0.5 / typeCnt[node2type[to]];
                    dfs(newNode);
                }
            }
            return;
        };
        dfs(from);
        scores.push_back(score / subgraph.size());
        subgraphs.push_back(move(subgraph));
    }
    localTimer.stop("classify nodes");


    // 2. find optimal subgraphs
    localTimer.start("find optimal");
    int maxCnt = -1;
    unordered_map<string, vector<int>> hashBins;
    for (size_t i = 0; i < subgraphs.size(); i++)
    {
        unordered_set<int> upper_nodes;
        for (auto from : subgraphs[i])
        {
            for (auto to : g[from])
            {
                if (nodes.count(to) != 0)
                    upper_nodes.insert(to);
            }
        }
        string hashKey = "";
        graphHash(subgraphs[i], upper_nodes, hashKey);
        // printf("\t %s: %f\n", hashKey.c_str(), scores[i]);
        if (hashBins.count(hashKey) == 0)
            hashBins[hashKey] = {};
        hashBins[hashKey].push_back(i);
    }

    string maxKey;
    double maxScore = 0;
    for (auto &kvec : hashBins)
    {
        double score = 0;
        for (auto &idx : kvec.second)
        {
            score += scores[idx];
        }
        score *= pow((float)kvec.second.size(), 0.2) - (kvec.second.size() == 1) * 1e6;
        if (maxScore < score)
        {
            maxScore = score;
            maxKey = kvec.first;
        }
        maxCnt = max(maxCnt, (int)kvec.second.size());
    }
    localTimer.stop("find optimal");
    
    if (profiling_flag > 1)
        printf("pattern: %s; score: %f; maxCnt: %d\n", maxKey.c_str(), maxScore, maxCnt);
    if (maxCnt == 1)
        return;

    localTimer.start("make return");
    // TODO: fine grained iso check
    vector<int> &opt_subgraph = hashBins[maxKey];
    function<bool(int, int)> isIso = [&](int i, int j)
    {
        // const unordered_set<int> & g1 = subgraphs[opt_subgraph[i]];
        // const unordered_set<int> & g2 = subgraphs[opt_subgraph[j]];
        return true;
    };
    assert(opt_subgraph.size());
    if (profiling_flag > 2)
        printf("opt_subgraph.size(): %ld\n", opt_subgraph.size());
    // class label, cnt
    vector<pair<int, int>> isomap(opt_subgraph.size(), {-1, -1});
    isomap[0] = {0, 1};
    pair<int, int> opt_label = {0, 1};
    for (size_t i = 1; i < opt_subgraph.size(); i++)
    {
        for (size_t j = 0; j < i; j++)
        {
            if (isIso(i, j))
            {
                isomap[i] = isomap[j];
                isomap[i].second++;
                if (isomap[i].second > opt_label.second)
                {
                    opt_label = isomap[i];
                }
                break;
            }
            else isomap[i] = {i, 1};
        }
    }

    fill(visited.begin(), visited.end(), false);
    for (size_t i = 0; i < isomap.size(); i++)
    {
        if (isomap[i].first == opt_label.first)
        {
            unordered_set<int> subgraph = subgraphs[opt_subgraph[i]];
            unordered_set<int> upper_nodes;
            for (auto from : subgraph)
            {
                for (auto to : g[from])
                {
                    if (nodes.count(to) == 0)
                        continue;
                    upper_nodes.insert(to);
                }
            }
            for (auto node : upper_nodes)
                subgraph.insert(node);
            bool unvisited = true;
            for (auto node : subgraph)
            {
                if (visited[node])
                {
                    unvisited = false;
                    break;
                }
                visited[node] = true;
            }
            if (unvisited)
                contract_subgraphs.push_back(move(subgraph));
            else
            {
                if (profiling_flag > 3)
                    fprintf(stderr, "conflict detected\n");
            }
        }
    }
    localTimer.stop("make return");
    addTimer.stop("getContractSubgraphs");

    return;
}

pair<int, int> DynamicBatching::bruteForce(unordered_set<int> &subgraph, vector<int> &best_batch_seq, const search_t & search_type)
{
    if (!subgraph.size()) return {-1,-1};
    addTimer.start("bruteForce");
    localTimer.start("bruteForce");
    unordered_map<int, vector<int>> type2nodes;
    unordered_map<int, int> inputCnt;

    // prepare inputCnt and type2nodes
    for (auto node : subgraph)
    {
        if (type2nodes.count(node2type[node]) == 0)
            type2nodes[node2type[node]] = {};
        inputCnt[node] = 0;
        int &cnt = inputCnt[node];
        const vector<vector<int> > & G = search_type == ROOTPRUNE? g:g_r;
        for (auto arg : G[node])
        {
            if (subgraph.count(arg))
                cnt++;
        }
        if (cnt == 0)
            type2nodes[node2type[node]].push_back(node);
    }

    vector<int> types;
    for (auto &type : type2nodes)
        types.push_back(type.first);

    int min_score = 1e6;
    int min_batch = -1;
    
    vector<int> batch_seq;
    int base = best_batch_seq.size();

    function<void(int, int)> search_all = [&](int score, int idx)
    {
        if (score >= min_score)
            return;
        int cnt = 0;
        for (auto &t2n : type2nodes)
        {
            cnt += t2n.second.size();
        }
        if (cnt == 0)
        {
            if (min_score > score)
            {
                min_score = score;
                min_batch = idx;
                for (int i = 0; i < idx; i++)
                    best_batch_seq[base + i] = batch_seq[i];
            }
            return;
        }

        for (auto &type : types)
        {
            if (type2nodes[type].size() == 0)
                continue;
            vector<int> batched_nodes = type2nodes[type];
            type2nodes[type].clear();
            for (auto &node : batched_nodes)
            {
                for (auto &to : g[node])
                {
                    if (subgraph.count(to) == 0)
                        continue;
                    if (--inputCnt[to] == 0)
                    {
                        type2nodes[node2type[to]].push_back(to);
                    }
                }
            }

            batch_seq[idx] = type;
            // printf("batch %s\n", type2name[type].c_str());
            search_all(score + batched_nodes.size() * type2weight[type], idx + 1);
            // printf("end batch %s\n", type2name[type].c_str());

            for (auto it = batched_nodes.rbegin(); it != batched_nodes.rend(); it++)
            {
                auto node = *it;
                for (auto itt = g[node].rbegin(); itt != g[node].rend(); itt++)
                {
                    auto to = *itt;
                    if (subgraph.count(to) == 0)
                        continue;
                    if (inputCnt[to]++ == 0)
                    {
                        type2nodes[node2type[to]].pop_back();
                    }
                }
            }
            type2nodes[type] = batched_nodes;
        }
    };

    function<void(int &, int &)> search_prune_bottom_to_top = [&] (int & depth, int &n_pruned_node) {
        min_score = 0;
        depth = n_pruned_node = 0;
        while (true){
            int leaf_type_cnt = 0, max_out_edges = 0, leaf_type;  
            for (auto type: types){
                if (!type2nodes[type].size()) continue;
                leaf_type_cnt ++;
                leaf_type = type;
                for (auto node: type2nodes[type]){
                    int cnt = 0;
                    for (auto to: g[node]){
                        if (subgraph.count(to))
                            cnt++;
                    }
                    max_out_edges = max(max_out_edges, cnt);
                }
            }
            if (leaf_type_cnt != 1 || max_out_edges > 1) break;
            best_batch_seq.push_back(leaf_type);
            vector<int> batch = type2nodes[leaf_type];
            type2nodes[leaf_type].clear();
            depth++;
            n_pruned_node += batch.size();
            min_score += batch.size() * type2weight[leaf_type];
            for (auto node: batch) {
                for (auto &to : g[node])
                {
                    if (subgraph.count(to) == 0)
                        continue;
                    if (--inputCnt[to] == 0)
                        type2nodes[node2type[to]].push_back(to);
                }
                subgraph.erase(node);
            }
        }
    };

    function<void(int &, int &, const vector<vector<int> >&)> search_prune = [&] (int & n_layer, int &n_pruned_node, const vector<vector<int> >&G) {
        min_score = 0;
        n_layer = n_pruned_node = 0;
        while (true){
            int type_cnt = 0, max_edges = 0, prune_type;  
            for (auto type: types){
                if (!type2nodes[type].size()) continue;
                type_cnt ++;
                prune_type = type;
                for (auto node: type2nodes[type]){
                    int cnt = 0;
                    for (auto next: G[node]){
                        if (subgraph.count(next))
                            cnt++;
                    }
                    max_edges = max(max_edges, cnt);
                }
            }
            if (type_cnt != 1 || (search_type == LEAFPRUNE && max_edges > 1)) break;
            best_batch_seq.push_back(prune_type);
            vector<int> batch = type2nodes[prune_type];
            type2nodes[prune_type].clear();
            n_layer++;
            n_pruned_node += batch.size();
            min_score += batch.size() * type2weight[prune_type];
            for (auto node: batch) {
                for (auto &next : G[node])
                {
                    if (subgraph.count(next) == 0)
                        continue;
                    if (--inputCnt[next] == 0)
                        type2nodes[node2type[next]].push_back(next);
                }
                subgraph.erase(node);
            }
        }
    };


    if (search_type == ALL){
        best_batch_seq.resize(base + subgraph.size());
        batch_seq.resize(subgraph.size());
        search_all(0, 0);
        assert(min_batch > 0);
        best_batch_seq.resize(base + min_batch);
    }
    else if (search_type == LEAFPRUNE){
        int depth, n_pruned_node;
        search_prune(depth, n_pruned_node, g);
        // search_prune_bottom_to_top(depth, n_pruned_node);
        if (profiling_flag > 3)
            printf("[LEAFPRUNE]: prune %d layer, %d nodes\n", depth, n_pruned_node);
    }else if (search_type == ROOTPRUNE){
        int depth, n_pruned_node;
        search_prune(depth, n_pruned_node, g_r);
        // search_prune_bottom_to_top(depth, n_pruned_node);
        if (profiling_flag > 3)
            printf("[ROOTPRUNE]: prune %d layer, %d nodes\n", depth, n_pruned_node);
    }
    localTimer.stop("bruteForce");
    addTimer.stop("bruteForce");
    return {min_batch, min_score};
}

int DynamicBatching::contract(const vector<unordered_set<int>> &subgraphs)
{
    addTimer.start("contract");
    int edgeCnt = 0;
    unordered_set<int> lower;
    unordered_set<int> upper;
    for (auto node: subgraphs[0]){
       for (auto to: g[node]) {
            if (subgraphs[0].count(to)) {
                lower.insert(node);
                upper.insert(to);
                edgeCnt ++;
            }
       }
    }
    bool isFull = (edgeCnt == lower.size() * upper.size());
    if (profiling_flag > 3)
        fprintf(stderr, "isFull: %d\n", isFull);

    int newType = --faketype_id;
    assert(subgraphs.size());
    type2name[newType] = "vt_" + std::to_string(-newType);
    unordered_set<int> need_red_update_nodes;
    for (auto subgraph : subgraphs)
    {
        int newNode = node_id++;
        nodes.insert(newNode);
        if (upper.size() != 1)
            need_red_update_nodes.insert(newNode);
        topo_value[newNode] = topo_value[*subgraph.begin()];
        node2type.push_back(newType);
        for (auto node : subgraph){
            nodes.erase(node);
        }
        g.push_back(std::vector<int>());
        g_r.push_back(std::vector<int>());
        auto &in_edges = g_r.back();
        auto &out_edges = g.back();
        in_edges.reserve(10);
        out_edges.reserve(10);
        unordered_set<int> froms;
        unordered_set<int> toes;
        for (auto node : subgraph)
        {
            for (auto from : g_r[node])
            {
                if (nodes.count(from) == 0)
                    continue;
                froms.insert(from);
                
            }
            for (auto to : g[node])
            {
                if (nodes.count(to) == 0)
                    continue;
                toes.insert(to);
            }
        }
        for (auto from: froms){
            g[from].push_back(newNode);
            if (lower.size() != 1)
                need_red_update_nodes.insert(from);
            in_edges.push_back(from);
        }
        for (auto to: toes){
            g_r[to].push_back(newNode);
            out_edges.push_back(to);
        }
    }
    
    if (!isFull)
        topo_sort();
    transitiveReduction(need_red_update_nodes);

    addTimer.stop("contract");
    return newType;
}



void DynamicBatching::solve(vector<int> &best_batch_seq)
{
    addTimer.start("solve");
    unordered_map<int, vector<int>> contract_batch_seqs;
    vector<int> best_batch_seq_from_behind;
    vector<int> newTypes;

    int n_iter = 0;
    int score_by_prune = 0;
    while (nodes.size() > LIMIT && n_iter < 20)
    {
        n_iter++;
        vector<unordered_set<int>> contracted_subgraphs;
        if(profiling_flag > 1)
            fprintf(stderr, "begin contraction %d |V| = %ld...\n", n_iter, nodes.size());
        localTimer.start("getContractSubgraph");
        getContractSubgraphs(contracted_subgraphs);
        localTimer.stop("getContractSubgraph");

        if (contracted_subgraphs.size())
        {   
            localTimer.start("contract");
            int newType = contract(contracted_subgraphs);
            localTimer.stop("contract");
            draw_graph("G" + std::to_string(n_iter) + ".gv", {&nodes, &contracted_subgraphs[0]}, "G" + std::to_string(n_iter));
            draw_graph("subgraph" + std::to_string(n_iter) + ".gv", {&contracted_subgraphs[0]}, "subgraph" + std::to_string(n_iter));
            newTypes.push_back(newType);
            contract_batch_seqs[newType] = vector<int>();
            int tmp, prune_score;
            tie(tmp, type2weight[newType]) = bruteForce(contracted_subgraphs[0], contract_batch_seqs[newType]);
            tie(tmp, prune_score) = bruteForce(nodes, best_batch_seq, LEAFPRUNE);
            score_by_prune += prune_score;
            tie(tmp, prune_score) = bruteForce(nodes, best_batch_seq_from_behind, ROOTPRUNE);
            score_by_prune += prune_score;
            if (profiling_flag > 1){
                fprintf(stderr, "batch seq: ");
                for (auto batch : contract_batch_seqs[newType])
                {
                    fprintf(stderr, "%s, ", type2name[batch].c_str());
                }
                fprintf(stderr, "\n");
            }
        }
        else
            break;
        if (profiling_flag > 1)
            localTimer.show();
        localTimer.clear();
    }

    if (profiling_flag > 1)
        fprintf(stderr, "contraction finished with iter %d %ld\n", n_iter, nodes.size());

    // 3. find the optimal policy by backtrack;
    int min_batch, min_score;
    tie(min_batch, min_score) = bruteForce(nodes, best_batch_seq);
    min_score += score_by_prune;

    for (auto iter = best_batch_seq_from_behind.rbegin(); iter != best_batch_seq_from_behind.rend(); iter++){
        best_batch_seq.push_back(*iter);
    }

    // 4. recover the full batch
    for (auto it = newTypes.rbegin(); it != newTypes.rend(); it++)
    {
        int type = *it;
        const vector<int> &batch_seq = contract_batch_seqs[type];
        for (auto b_iter = best_batch_seq.begin(); b_iter != best_batch_seq.end();)
        {
            if (*b_iter == type)
            {
                b_iter = best_batch_seq.erase(b_iter);
                b_iter = best_batch_seq.insert(b_iter, batch_seq.begin(), batch_seq.end());
                b_iter += batch_seq.size();
            }
            else
                b_iter++;
        }
    }
    if (profiling_flag > 2)
        printf("n_batch: %ld, min_score: %d\n", best_batch_seq.size(), min_score);
    if (profiling_flag > 1){
        addTimer.stop("solve");
        addTimer.show();
        printf("[DB::solve] iter %d\n", n_iter);
    }
}

void DynamicBatching::transitiveReduction(unordered_set<int>& redNodes)
{
    addTimer.start("transitiveReduction");
    localTimer.start("transitiveReduction");
    vector<pair<int, int> > erase_edges; 
    for (auto from: redNodes){
        unordered_map<int, int> visited;
        priority_queue<int, vector<int>, function<bool(int a, int b)> > q(
            [&](int a, int b){return topo_value[a] > topo_value[b];}
        );
        unordered_set<int> toes;
        int max_topo = -1;
        for (auto node: g[from]) {
            toes.insert(node);
            max_topo = max(max_topo, topo_value[node]);
        }
        q.push(from);
        while(!q.empty()){
            auto node = q.top();
            q.pop();
            if (toes.count(node) && visited[node] > 1)
                erase_edges.push_back({from, node});
            for (auto to: g[node]){
                if (topo_value[to] > max_topo || nodes.count(to) == 0) continue;
                if (!visited.count(to)){
                    visited[to] = 0;
                    q.push(to);
                } 
                visited[to]++;
            }
        }
    }


    for (auto edge: erase_edges){
        int from, to;
        tie(from, to) = edge;
        // printf("erase %s%d %s%d\n", type2name[node2type[from]].c_str(), from, type2name[node2type[to]].c_str(), to);
        g[from].erase(find(g[from].begin(), g[from].end(), to));
        g_r[to].erase(find(g_r[to].begin(), g_r[to].end(), from));
    }
    localTimer.stop("transitiveReduction");
    addTimer.stop("transitiveReduction");
}


void DynamicBatching::topo_sort(){
    localTimer.start("topo sort");
    addTimer.start("topo sort");
    unordered_set<int> visited;
    int nid = 0;
    function<void(int)> dfs = [&](int node)->void{
        if (topo_value[node] >= 0) return;
        int & value = topo_value[node];
        for (auto from: g_r[node]){
            dfs(from);
        }        
        value = nid++;
    };

    topo_value.clear();
    for (auto node: nodes){
        topo_value[node] = -1;
    }
    for (auto node: nodes){
        dfs(node);
    }
    addTimer.stop("topo sort");
    localTimer.stop("topo sort");
}

void DynamicBatching::draw_graph(string filename, initializer_list<unordered_set<int> *> subgraphs, string graphName){
    if (profiling_flag == 0) return;
    ofstream file;
    file.open(filename);
    // lookup, logistic, gemm, add, tanh, cmult, concatenate

    file << "digraph " << graphName << " {\n";
    int idx = 0;
    for (auto& nodes: subgraphs){
        unordered_map<int, string> cmap;
        unordered_map<int, string> nodecmap;

        file << "subgraph " << graphName + to_string(idx++) << "{\n";
        for (size_t node_id = 0; node_id < g.size(); node_id++)
        {
            if (nodes->count(node_id) == 0)
                continue;
            if (cmap.count(node2type[node_id]) == 0) 
            {
                char tmp[10];
                sprintf(tmp, "#%2x%2x%2x", rand()&0xff, rand()&0xff, rand()&0xff);
                cmap[node2type[node_id]] = string(tmp);
            }
            nodecmap[node_id] = cmap[node2type[node_id]];
            
            string nodetag = type2name[(op_type)node2type[node_id]] + "_" + std::to_string(node_id) + "_" + to_string(topo_value[node_id]);
            for (auto to : g[node_id])
            {
                if (nodes->count(to) == 0)
                    continue;
                string totag = type2name[(op_type)node2type[to]] + "_" + std::to_string(to) + "_" + to_string(topo_value[to]);
                file << "\t" << nodetag << " -> " << totag << ";\n";
            }
        }
        for (auto& kv: nodecmap){
            auto node_id = kv.first;
            string nodetag = type2name[(op_type)node2type[node_id]] + "_" + std::to_string(node_id) + "_" + to_string(topo_value[node_id]);
            file << "\t" << nodetag << "\t[color=\"" << kv.second << "\"];" << endl;
        }
        file << "}\n\n";
    }
    file << "}\n";
    file.close();
}
/* TODO:
1. type2weight;
2. isophomism;
3. stop condition;
4. connect with dynet.
*/