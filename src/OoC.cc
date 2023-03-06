#include <random>
#include "OoC.h"
#include "pqtree.h"
#include "ext-pqtree.h"

using namespace std;
#define LIMIT 3
#define BATCHLIMIT 5

namespace OoC{

Timer ooc_timer("ADD");

int profiling_flag = 0;

unordered_map<int, string> type2name = {
    {OoC::unbatchable, "unbatchable"},
    {OoC::tanh, "tanh"},
    {OoC::sqrt, "sqrt"},
    {OoC::abs, "abs"},
    {OoC::erf, "erf"},
    {OoC::square, "square"},
    {OoC::cube, "cube"},
    {OoC::exp, "exp"},
    {OoC::logsigmoid, "logsigmoid"},
    {OoC::loggamma, "loggamma"},
    {OoC::log, "log"},
    {OoC::nobackprop, "nobackprop"},
    {OoC::scalegradient, "scalegradient"},
    {OoC::identity, "identity"},
    {OoC::negate, "negate"},
    {OoC::rectify, "rectify"},
    {OoC::logistic, "logistic"},
    {OoC::softsign, "softsign"},
    {OoC::silu, "silu"},
    {OoC::round, "round"},
    {OoC::ceiling, "ceiling"},
    {OoC::floor, "floor"},
    {OoC::sinh, "sinh"},
    {OoC::cosh, "cosh"},
    {OoC::asinh, "asinh"},
    {OoC::acosh, "acosh"},
    {OoC::atanh, "atanh"},
    {OoC::sin, "sin"},
    {OoC::cos, "cos"},
    {OoC::tan, "tan"},
    {OoC::asin, "asin"},
    {OoC::acos, "acos"},
    {OoC::atan, "atan"},
    {OoC::plus_const, "plus_const"},
    {OoC::concat, "concat"},
    {OoC::cmult, "cmult"},
    {OoC::csum, "csum"},
    {OoC::sum, "sum"},
    {OoC::squared_distance, "squared_distance"},
    {OoC::softmax, "softmax"},
    {OoC::pnls, "pnls"},
    {OoC::pickrange, "pickrange"},
    {OoC::scalar_mult, "scalar_mult"},
    {OoC::dropout, "dropout"},
    {OoC::input, "input"},
    {OoC::scalar_input, "scalar_input"},
    {OoC::lookup, "lookup"},
    {OoC::select, "select"},
    {OoC::argmax_index, "argmax_index"},
    {OoC::COMPLEX, "COMPLEX"},
    {OoC::affine, "affine"},
    {OoC::matmul, "matmul"},
    {OoC::transpose, "transpose"},
    {OoC::vanilla_lstm_gates, "vanilla_lstm_gates"},
    {OoC::vanilla_lstm_h, "vanilla_lstm_h"},
    {OoC::vanilla_lstm_c, "vanilla_lstm_c"},
    {OoC::conv2d, "conv2d"},
    {OoC::loss, "loss"},
    {OoC::block, "block"},
    {OoC::get, "get"},
    {OoC::select, "select"},
    {OoC::argmax_index, "argmax_index"}
};

DynamicBatching::DynamicBatching(string mode_str, string score_func_str): 
    localTimer("DEFAULT"), 
    addTimer("ADD") {
    if (mode_str == "train"){
        pattern_cache = move(make_unique<PatternCache>(this));
        mode = TRAIN;
    }
    else if (mode_str == "default") mode = NOCACHE;
    else assert(false);
    if (score_func_str == "tfidf") score_func = TFIDF;
    else if (score_func_str == "information_entropy") score_func = INFORMATION_ENTROPY;
    else assert(false);

    assert(mode == NOCACHE || score_func == INFORMATION_ENTROPY);
    n_train_iteration = n_inference_iteration = 0;
}

DynamicBatching::DynamicBatching(vector<vector<int>>& g_in, vector<int>& node2type_in, bool deleteUnbatachable) : 
    faketype_id(0), 
    node_id((int)g_in.size()), 
    n_node_input((int)g_in.size()), 
    isHashed(false),
    localTimer("DEFAULT"), 
    addTimer("ADD"), 
    mode(NOCACHE)
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


    draw_graph("./pics/G.gv", {&nodes}, "G");

    
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
                    // fprintf(stdout,"from %d\n", from);
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
            fprintf(stdout, "Unbatchable: %ld, Batchable: %ld\n", childrenOfUnbatchable.size(), nodes.size());
        }
    }
    transitiveReduction(nodes);
    draw_graph("./pics/G0.gv", {&nodes}, "G0");
}

void DynamicBatching::setGraph(vector<vector<int> >& g_r_in, vector<int> &node2type_in, bool deleteUnBatchable){
    addTimer.start("setGraph");
    g.clear();
    g_r.clear();
    topo_value.clear();
    
    faketype_id = 0;
    node2type.clear();
    type2weight.clear();
    type2BatchSeq.clear();
    type2nop.clear();
    node2nop.clear();
    linearTypes.clear();
    type2father.clear();
    root_types.clear();

    node_id = g_r_in.size();
    n_node_input = g_r_in.size();
    n_unbatchable = 0;
    nodes.clear();
    isHashed = false;
    node_hash.clear();
    node2father.clear();

    boundary_edges.clear();
    addTimer.clear();

    n_train_iteration += mode == TRAIN;
    n_inference_iteration += mode == INFERENCE;

    addTimer.start("init graph");
    for (int nid = 0; nid < (int)g_r_in.size(); nid++){
        nodes.insert(nid);
        topo_value[nid] = nid;
        node2father.push_back(nid);
    }

    g.resize(g_r_in.size(), {});
    g_r = g_r_in;
    if (mode == TRAIN) old_g_r = g_r_in;
    for (auto node : nodes)
    {
        for (auto from : g_r_in[node]){
            assert(from < node);
            g[from].push_back(node);
        }
    }

    for (auto t : node2type_in)
        node2type.push_back(t);

    if (mode == TRAIN) {
        for (int t = op_type::unbatchable; t != op_type::END; t++)
            type2father[t] = t;
    }

    for (auto type: node2type)
        type2weight[type] = 1;
    addTimer.stop("init graph");

    addTimer.start("init distribution");
    if (score_func == INFORMATION_ENTROPY || score_func == TFIDF){
        // init node2father
        unordered_map<int, int> distr;
        distribution_ptr = move(make_unique<Distribution>());
        for (int nid = 0; nid < (int)g_r_in.size(); nid++){
            int type = node2type[nid];
            if (distr.count(type) == 0) distr[type] = 0;
            if (type2nop.count(type) == 0) type2nop[type] = 1;
            distr[type] += 1;
        }
        distribution_ptr->update(distr, 1, true);
        // init distribution
    }
    addTimer.stop("init distribution");

    if (profiling_flag > 0){
        string graphName = "G";
        if (mode == TRAIN) graphName = "T" + to_string(n_train_iteration);
        else if (mode == INFERENCE) graphName = "I"+to_string(n_inference_iteration);
        draw_graph("./pics/"+graphName+".gv", {&nodes}, graphName);
    }

    addTimer.start("delete unbatchable");
    // erase nodes that cannot be batched
    if(deleteUnBatchable){
        unordered_map<int, vector<int> > childrenOfUnbatchable;
        for (size_t nid = 0; nid < g_r_in.size(); nid++){
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
                n_unbatchable += 1;
            }
            else {
                for (auto from: g_r[nid]){
                    // fprintf(stdout,"from %d\n", from);
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
            fprintf(stdout, "Unbatchable: %ld, Batchable: %ld\n", childrenOfUnbatchable.size(), nodes.size());
        }
    }
    addTimer.stop("delete unbatchable");

    hashNodes();
    if (mode == TRAIN || mode == NOCACHE)
        transitiveReduction(nodes, true && (mode != NOCACHE));
    else if (mode == INFERENCE) 
        pattern_cache->transitive_reduction(nodes);

    if (profiling_flag > 0){
        string graphName = "G0";
        if (mode == TRAIN)
            graphName = "T"+to_string(n_train_iteration) + "_0";
        else if (mode == INFERENCE)
            graphName = "I"+to_string(n_inference_iteration) + "_0";
        draw_graph("./pics/" + graphName + ".gv", {&nodes}, graphName);
    }
    addTimer.stop("setGraph");
}

bool DynamicBatching::findOptPatternAndContract(vector<unordered_set<int> > & typical_subgraphs, double T)
{
    addTimer.start("findOptPatternAndContract");
    // 1. find isomorphic subgraphs: tfidf
    unordered_map<int, int> typeCnt;
    for (auto node : nodes)
    {
        int type = node2type[node];
        if (typeCnt.count(type) == 0)
            typeCnt[type] = {};
        typeCnt[type]++;
    }

    if (profiling_flag > 1){
        fprintf(stdout, "[TypeCnt]:\n");
        int idx = 0;
        for (auto kv: typeCnt) {
            fprintf(stdout, "%s:\t%d;", type2name[kv.first].c_str(), kv.second * type2nop[kv.first]);
            if (++idx == 4)
                fprintf(stdout, "\n");
        }
        fprintf(stdout, "\n");
    }

    localTimer.start("classify nodes");
    // the partition of the lower nodes
    vector<unordered_set<int>> subgraphs;
    vector<double> scores;
    vector<bool> visited(node_id, false);
    unordered_map<int, vector<int> > hashBins;
    for (auto from : nodes)
    {
        if (visited[from])
            continue;
        int hashKey = 0;
        unordered_set<int> subgraph;
        double score = 0;
        unordered_set<int> visited_to;
        int n_edge = 0;
        function<void(int)> dfs = [&](int node)
        {
            if (nodes.count(node) == 0 || visited[node])
                return;
            hashKey ^= node2type[node];
            visited[node] = true;
            subgraph.insert(node);
            score += 0.5 / typeCnt[node2type[node]];
            for (auto to : g[node])
            {
                n_edge += nodes.count(to);
                if (nodes.count(to) == 0 || visited_to.count(to))
                    continue;
                hashKey ^= node2type[to];
                hashKey ^= node2type[to] << TO_OFFSET;
                visited_to.insert(to);
                for (auto newNode : g_r[to])
                {
                    if (nodes.count(newNode) == 0)
                        continue;
                    hashKey ^= node2type[newNode] << FROM_OFFSET;
                    score += 0.5 / typeCnt[node2type[to]];
                    dfs(newNode);
                }
            }
            return;
        };
        dfs(from);
        if ((int)(subgraph.size() * visited_to.size()) != n_edge)
            continue;
        scores.push_back(score / subgraph.size());
        for (auto node: visited_to) subgraph.insert(node);
        if (subgraph.size() == 1 ) {
            if (profiling_flag > 2)
                fprintf(stdout,"single node subgraph %s %d\n", type2name[node2type[from]].c_str(), from);
            continue;
        }
        subgraphs.push_back(move(subgraph));
        if (hashBins.count(hashKey) == 0) hashBins[hashKey] = {};
        hashBins[hashKey].push_back(subgraphs.size() - 1);
    }

    // the linear cluster
    unordered_map<int, vector<int> > linearType2indices;
    for (auto node: nodes){
        bool isFrontier = true;
        for (auto from: g_r[node]){
            if (nodes.count(from)) {
                isFrontier = false;
                break;
            }
        } 
        if (isFrontier){
            unordered_set<int> subgraph;
            int type = node2type[node];
            while(true){
                subgraph.insert(node);
                int next = -1, toCnt = 0;
                for (auto to: g[node]) {
                    if (nodes.count(to)){
                        int fromCnt = 0;
                        for (auto from: g_r[to]) {
                            if (nodes.count(from)) fromCnt++;
                        }
                        if (fromCnt > 1) {
                            toCnt = -1;
                            break;
                        }
                        toCnt+=1;
                        if (node2type[to] == type) 
                            next = to; 
                    } 
                }
                if (toCnt != 1 || next == -1)
                    break;
                node = next;
            }
            subgraphs.push_back(subgraph);
            if (profiling_flag > 3) {
                fprintf(stdout, "linear of type %s, %ld\n", type2name[type].c_str(), subgraph.size());
            }
            if (linearType2indices.count(type) == 0) linearType2indices[type] = {};
            linearType2indices[type].push_back(subgraphs.size() - 1); 
            // the first is the longest.
            auto & maxIdx = linearType2indices[type].front();
            if (subgraphs[maxIdx].size() < subgraph.size()) { 
                swap(maxIdx, linearType2indices[type].back());
            }
        }
    }
    localTimer.stop("classify nodes");

    // 2. find optimal subgraphs
    localTimer.start("find optimal");

    // tfidf 
    int maxKey = -1;
    double maxScore = -1e6;
    
    function<void()> tfidf = [&](){
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
        }
    };
    // information entropy 
    // function<void()> infoEntropy = [&](){
        
    // };
    
    if (score_func == TFIDF) {
        tfidf();
    }

    else if (score_func == INFORMATION_ENTROPY){
        for (auto &kvec: hashBins){
            if (kvec.second.size() <= 1) continue;
            unordered_map<int, int> distr;
            bool hasLinear = false;
            auto subgraph = subgraphs[kvec.second[0]];
            for (auto node: subgraph){
                if (type2nop[node2type[node]] < 0){
                    hasLinear = true;
                    break;
                }
            }
            int factor;
            if (hasLinear){
                for (auto idx: kvec.second)
                    getDistribution(subgraphs[idx], distr);
                factor = 1;
            }
            else {
                getDistribution(subgraph, distr);
                factor = kvec.second.size();
                if (profiling_flag > 1){
                    fprintf(stdout, "distr %d: ", factor);
                    for (auto kv: distr) {
                        fprintf(stdout, "%s: %d; ", type2name[kv.first].c_str(), kv.second);
                    }
                    fprintf(stdout, "\n");
                }
            }
            if (profiling_flag > 2) {
                fprintf(stdout,"subgraph.size();%ld\n", subgraph.size());
            }
            // TODO: to solve the overlap problem 
            double score = distribution_ptr->delta_entropy(distr, factor);
             
            if (score > maxScore) {
                maxKey = kvec.first;
                maxScore = score;
            }
        }
    }

    // the linear subgraph   
    double linearScore = 1e-3; // a little bonus to make linear subgraph outperform other contraction that do not change distribution             
    int n_reduced_node = 0;
    for (auto& kv: linearType2indices) {
        int type = kv.first;
        unordered_map<int, int> distr;
        for (auto idx: kv.second){
            getDistribution(subgraphs[idx], distr);
            n_reduced_node += subgraphs[idx].size()-1;
        }
        linearScore += distribution_ptr->delta_entropy(distr, 1);
        if (profiling_flag > 3)
            fprintf(stdout, "linear type: %s, %d\n", type2name[type].c_str(), distr[type]);
    }
    bool isLinear = false;
    if (n_reduced_node && linearScore > maxScore) {
        maxScore = linearScore;
        isLinear = true;
    }

    localTimer.stop("find optimal");
    
    bool update = (maxScore > -1e5) && (rand() / (RAND_MAX + 0.0)) < std::exp(maxScore / T);
    if (profiling_flag > 0) {
        fprintf(stdout, "[getContractSubgraph]: entropy:%f, temperature:%f, sa rate: %f, isLinear %d\n", maxScore, T, std::exp(maxScore / T), isLinear);
    }
    if (!update)
        return false;


    localTimer.start("contract preparation");

    if (isLinear) {
        vector<unordered_set<int> > contract_subgraphs;
        vector<int> typeCnts;
        for (auto& kv: linearType2indices) {
            for (auto &idx : kv.second) 
                contract_subgraphs.push_back(subgraphs[idx]);
            typeCnts.push_back(kv.second.size());  
            typical_subgraphs.push_back(subgraphs[kv.second[0]]);    
        }
        localTimer.stop("contract preparation");
        contract(contract_subgraphs, typeCnts, true);
        addTimer.stop("findOptPatternAndContract");
        return true;
    }

    if (profiling_flag > 1){
        assert(hashBins[maxKey].size());
        auto & subgraph = subgraphs[hashBins[maxKey][0]];
        fprintf(stdout, "pattern: ");
        for (auto node: subgraph) fprintf(stdout,"%s,", type2name[node2type[node]].c_str());
        fprintf(stdout, "; score: %f; \n", maxScore);
    }

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
        fprintf(stdout,"opt_subgraph.size(): %ld\n", opt_subgraph.size());
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
    vector<unordered_set<int> > contract_subgraphs;
    fill(visited.begin(), visited.end(), false);
    for (size_t i = 0; i < isomap.size(); i++)
    {
        if (isomap[i].first == opt_label.first)
        {
            unordered_set<int> subgraph = subgraphs[opt_subgraph[i]];
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
                    fprintf(stdout, "conflict detected\n");
            }
        }
    }
    localTimer.stop("make return");
    
    contract(contract_subgraphs, {(int)contract_subgraphs.size()}, false);
    typical_subgraphs.push_back(contract_subgraphs[0]);

    addTimer.stop("findOptPatternAndContract");

    return true;
}

int DynamicBatching::getDistribution (const unordered_set<int>& subgraph, unordered_map<int, int>& distr){
    addTimer.start("getDistribution");
    int ret = 0;
    for (auto node: subgraph) {
        int type = node2type[node];
        if(distr.count(type) == 0) distr[type] = 0;
        int nop = type2nop[type] > 0? type2nop[type]:node2nop[node];
        distr[type] += nop;
        ret += nop;
    }
    addTimer.stop("getDistribution");
    return ret;
}

int DynamicBatching::mem_cost(const std::vector<std::vector<int> >& batches, const std::unordered_set<int>& subgraph, const int maxIdx, std::list<int> & memIds){
    auto & node2args = old_g_r;
    set<set<int> > subsets;
    vector<vector<vector<int> > > iso_patterns;
    int mem_transfer = 0;
    for (int i = 0; i < maxIdx; i++){
        auto & batch = batches[i];
        assert(batch.size());
        assert(batch[0] >= 0 && batch[0] < node2args.size());
        if (batch.size() > 1)
            subsets.insert(set<int>(batch.begin(), batch.end()));
        iso_patterns.push_back({});
        for (auto nid: batch) assert(node2args[batch[0]].size() == node2args[nid].size());
        for (int aid = 0; aid < node2args[batch[0]].size(); aid++) {
            set<int> subset;
            bool isAbsent = false;
            iso_patterns.back().push_back({});
            for (auto nid: batch) {
                if (subgraph.count(node2args[nid][aid]) == 0) {
                    isAbsent = true, ++mem_transfer;
                    break;
                }
                iso_patterns.back().back().push_back(node2args[nid][aid]);
                subset.insert(node2args[nid][aid]);
            }
            if (isAbsent || subset.size() < iso_patterns.back().back().size()) {
                iso_patterns.back().pop_back();
            }
            if (!isAbsent && subset.size() > 1){
                subsets.insert(move(subset));
            }
        }
        iso_patterns.back().push_back(batch);
        if (batch.size() == 1) iso_patterns.pop_back();
    }

    IsoPQTree tree(subgraph.begin(), subgraph.end());
    bool suc = true;
    if (profiling_flag > -1){
        fprintf(stdout, "mem_cost::subgraph: ");
        for (auto ele: subgraph) fprintf(stdout, "%d, ", ele);
        fprintf(stdout, "\n");
        fprintf(stdout, "batches: ");
        for(int i = 0; i < maxIdx; i++) {
            fprintf(stdout, "{");
            for (auto ele: batches[i]) fprintf(stdout, "%d, ", ele);
            fprintf(stdout, "},");
        }
        fprintf(stdout, "\n");
        fprintf(stdout, "subsets: ");
        for(auto &subset: subsets) {
            fprintf(stdout, "{");
            for (auto ele: subset) fprintf(stdout, "%d, ", ele);
            fprintf(stdout, "},");
        }
        fprintf(stdout, "\n");
        fprintf(stdout, "pattern:");
        for (auto & pattern: iso_patterns){
            fprintf(stdout, "{");
            for(auto &subset: pattern) {
                fprintf(stdout, "{");
                for (auto ele: subset) fprintf(stdout, "%d, ", ele);
                fprintf(stdout, "},");
            }
            fprintf(stdout, "},");
        }
        fprintf(stdout, "\n");
    }
    bool succ = tree.ReduceAll(subsets) && tree.IsoReduceAll(iso_patterns);
    assert(succ);
    memIds = tree.Frontier();
    
    if (profiling_flag > -1){
        fprintf(stdout, "pqtree: %s\n", tree.Print().c_str());
        fprintf(stdout, "memIds: ");
        for (auto ele: memIds) fprintf(stdout, "%d, ", ele);
        fprintf(stdout, "\n");
    }
    
    return mem_transfer;
}

pair<int, int> DynamicBatching::bruteForce(
    unordered_set<int> &subgraph, 
    vector<int> &best_batch_seq, 
    const search_t & search_type, 
    Pattern * pattern){
    if (!subgraph.size()) return {-1,-1};
    addTimer.start("bruteForce");
    localTimer.start("bruteForce");
    unordered_map<int, vector<int>> type2nodes;
    unordered_map<int, int> typeCnt;
    unordered_map<int, int> inputCnt;

    // prepare inputCnt and type2nodes
    for (auto node : subgraph)
    {
        int type = node2type[node];
        if (typeCnt.count(type) == 0) typeCnt[type] = 0;
        typeCnt[type] ++;
        if (type2nodes.count(type) == 0)
            type2nodes[type] = {};
        inputCnt[node] = 0;
        int &cnt = inputCnt[node];
        const vector<vector<int> > & G = search_type == ROOTPRUNE? g:g_r;
        for (auto arg : G[node])
        {
            if (subgraph.count(arg))
                cnt++;
        }
        if (cnt == 0)
            type2nodes[type].push_back(node);
    }

    vector<int> types;
    for (auto &type : type2nodes)
        types.push_back(type.first);

    int min_score = 1e6;
    int min_batch = -1;
    
    vector<int> batch_seq;
    vector<vector<int> > batches(subgraph.size());
    int base = best_batch_seq.size();

    if (pattern){
        pattern->nodes.clear();
        pattern->nodes.insert(pattern->nodes.end(), subgraph.begin(), subgraph.end());
        sort(pattern->nodes.begin(), pattern->nodes.end());
        pattern->types.resize(subgraph.size());
        for (int i = 0; i < subgraph.size(); i++)
            pattern->types[i] = node2type[pattern->nodes[i]];
    }
    
    int n_template_explored = 0;

    function<void(int, int)> search_all = [&](int score, int idx){
        if (score >= min_score)
            return;
        int cnt = 0;
        for (auto &t2n : type2nodes)
            cnt += t2n.second.size();
        if (cnt == 0)
        {
            if (min_score > score)
            {
                n_template_explored ++;
                min_score = score;
                min_batch = idx;
                for (int i = 0; i < idx; i++)
                    best_batch_seq[base + i] = batch_seq[i];
                if (pattern != nullptr) {
                    unordered_map<int, int> nid2bid;
                    for (int bid = 0; bid < idx; bid++){
                        for (auto nid: batches[bid]){
                            nid2bid[nid] = bid;
                        }
                    }
                    list<int> memIds;
                    mem_cost(batches, subgraph, idx, memIds);
                    int min_nid = pattern->nodes[0];
                    pattern->batch_ids.clear();
                    pattern->batch_ids.resize(idx);
                    pattern->mem_allocation_order.clear();
                    auto iter = memIds.begin();
                    while(iter!=memIds.end()){
                        int bid = nid2bid[*iter];
                        auto & batch_id = pattern->batch_ids[bid];
                        pattern->mem_allocation_order.push_back(bid);
                        while(iter!=memIds.end()&&nid2bid[*iter]==bid){
                            batch_id.push_back(*iter - min_nid);
                            iter++;
                        }
                    }
                }
            }
            return;
        }

        for (auto &type : types)
        {
            if (type2nodes[type].size() == 0)
                continue;
            batches[idx] = type2nodes[type];
            type2nodes[type].clear();
            for (auto &node : batches[idx])
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
            // fprintf(stdout,"batch %s\n", type2name[type].c_str());
            batch_seq[idx] = type;
            search_all(score + type2weight[type], idx + 1);
            // fprintf(stdout,"end batch %s\n", type2name[type].c_str());

            for (auto it = batches[idx].rbegin(); it != batches[idx].rend(); it++)
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
            type2nodes[type] = batches[idx];
        }
        return;
    };
    
    // Node level bruteforce; get memory into consideration

    // Rule1: if every string has pattern .*t+, then there is an optimal SCS of .*t+
    function<void(int &, int &, const vector<vector<int> >&)> prune_rule1 = [&] (int & n_layer, int &n_pruned_node, const vector<vector<int> >&G) {
        n_layer = 0;
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
            if (type_cnt != 1) break;
            best_batch_seq.push_back(prune_type);
            vector<int> batch = type2nodes[prune_type];
            typeCnt[prune_type] -= batch.size();
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

    // Rule2: if every string has pattern (^t)t*, then there is an optimal string of pattern (^t)t*
    function<int(int, const vector<vector<int> >&, unordered_map<int, int>&) > topoSort_util = [&](int nid, const vector<vector<int> > & edges, unordered_map<int, int>& local_topo_value) {
        if (local_topo_value.count(nid)) return local_topo_value[nid];
        local_topo_value[nid] = 1;
        int & depth = local_topo_value[nid];
        for (auto to: edges[nid]) {
            if (subgraph.count(to) == 0 || node2type[nid] != node2type[to]) continue;
            depth = max(depth, topoSort_util(to, edges, local_topo_value) + 1);
        }
        return depth;
    };

    function<void(const vector<vector<int> > &, int&)> prune_rule2 = [&](const vector<vector<int> > &edges, int & n_pruned_node){
        while(true) {
            bool foundPrune = false;
            for (auto & kv: type2nodes){
                if (kv.second.size() == 0) continue;
                unordered_map<int, int> local_topo_value;
                int maxDepth = 0;
                for (auto node: kv.second){
                    maxDepth = max(maxDepth, topoSort_util(node, edges, local_topo_value));
                }
                assert(maxDepth);
                const int& typeOfThis = kv.first;  
                if ((int)local_topo_value.size() == typeCnt[typeOfThis]){
                    n_pruned_node += typeCnt[typeOfThis];
                    foundPrune = true;
                    kv.second.clear();
                    for (auto node_depth: local_topo_value){
                        for (auto to: edges[node_depth.first]){
                            int typeOfTo = node2type[node_depth.first];
                            if (subgraph.count(to) == 0 || typeOfTo == typeOfThis) continue;
                            if (--inputCnt[to] == 0) type2nodes[typeOfTo].push_back(to); // may be dangerous
                        }
                        subgraph.erase(node_depth.first);
                    }
                    typeCnt[typeOfThis] = 0;
                    for (int i = 0; i < maxDepth; i++){
                        best_batch_seq.push_back(kv.first);
                    }
                    min_score += type2weight[typeOfThis] *maxDepth;
                }
            }
            if (!foundPrune) break;
        }
    };

    if (search_type == ALL){
        best_batch_seq.resize(base + subgraph.size());
        batch_seq.resize(subgraph.size());
        search_all(0, 0);
        fprintf(stdout, "search::%d candidate explored\n", n_template_explored);
        assert(min_batch > 0);
        best_batch_seq.resize(base + min_batch);
    }
    else if (search_type == LEAFPRUNE){
        min_score = 0;
        int depth, n_pruned_node;
        n_pruned_node = 0;
        prune_rule1(depth, n_pruned_node, g);
        if (profiling_flag > 2)
            fprintf(stdout,"[LEAFPRUNE]: rule1: prune %d layer, %d nodes; ", depth, n_pruned_node);
        n_pruned_node = 0;
        prune_rule2(g, n_pruned_node);
        if (profiling_flag > 2)
            fprintf(stdout,"rule2: prune %d nodes\n", n_pruned_node);
        // search_prune_bottom_to_top(depth, n_pruned_node);
    }else if (search_type == ROOTPRUNE){
        min_score = 0;
        int depth, n_pruned_node;
        n_pruned_node = 0;
        prune_rule1(depth, n_pruned_node, g_r);
        if (profiling_flag > 2)
            fprintf(stdout,"[ROOTPRUNE]: rule1: prune %d layer, %d nodes; ", depth, n_pruned_node);
        n_pruned_node = 0;
        prune_rule2(g_r, n_pruned_node);
        // search_prune_bottom_to_top(depth, n_pruned_node);
        if (profiling_flag > 2)
            fprintf(stdout,"rule2: prune %d nodes\n", n_pruned_node);
    }
    localTimer.stop("bruteForce");
    addTimer.stop("bruteForce");
    return {min_batch, min_score};
}

void DynamicBatching::contract_update_graph(vector<unordered_set<int>> &subgraphs, bool isLinear)
{

    if (profiling_flag > 2) {
        fprintf(stdout,"subgraphs: ");
        for (auto subgraph: subgraphs){
            for (auto node: subgraph){
                fprintf(stdout,"%s ", type2name[node2type[node]].c_str());
            }
            fprintf(stdout,";");
        }
        fprintf(stdout,"\n");
    }
    addTimer.start("contract_update_graph");

    // calculate graph statistics
    int lwcnt, upcnt;
    bool isFull;
    if (isLinear) {
        lwcnt = upcnt = 1;
        isFull = true;
    }
    else 
    {
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
        isFull = (edgeCnt == (int)(lower.size() * upper.size()));
        lwcnt = lower.size();
        upcnt = upper.size();
        if (profiling_flag > 0 && !isFull)
            fprintf(stdout, "[WARNING]: contract not full subgraph\n");
    }

    addTimer.start("update_nodes@contract_update_graph");
    for (size_t i = 0; i < subgraphs.size(); i++){
        int newNode = node_id + i;
        auto & subgraph = subgraphs[i];
        node2father.push_back(newNode);
        for (auto node: subgraph) {
            node2father[node] = newNode;
        }
        nodes.insert(newNode);
        // full graph contraction; topo value can be inherited
        topo_value[newNode] = topo_value[*subgraph.begin()];
        for (auto node : subgraph){
            nodes.erase(node);
        }
    }
    addTimer.stop("update_nodes@contract_update_graph");

    addTimer.start("find new edge@contract_update_graph");
    unordered_set<pair<int, int>, hash_pair> newEdges;
    int cnt = 0;
    for (size_t i = 0; i < subgraphs.size(); i++)
    {
        int newNode = node_id + i;
        auto & subgraph = subgraphs[i];
        for (auto node : subgraph)
        {
            assert(newNode == node2father[node]);
            for (auto from : g_r[node])
            {
                cnt++;
                if (node2father[from] == newNode || nodes.count(node2father[from]) == 0)
                    continue;
                newEdges.insert({node2father[from], newNode});
            }
            for (auto to : g[node])
            {
                cnt++;
                if (node2father[to] == newNode || nodes.count(node2father[to]) == 0 )
                    continue;
                newEdges.insert({newNode, node2father[to]});
            }
        }
    }
    if (profiling_flag > 0){
        fprintf(stdout, "subgraphs.size(): %ld; newEdges.size(): %ld; %d edges visited\n", subgraphs.size(), newEdges.size(), cnt);
    }
    addTimer.stop("find new edge@contract_update_graph");
    
    addTimer.start("insert edge@contract_update_graph");
    g.resize(g.size() + subgraphs.size());
    g_r.resize(g_r.size() + subgraphs.size());
    unordered_set<int> need_red_update_nodes;
    for (auto edge: newEdges){
        assert(nodes.count(edge.first) && nodes.count(edge.second));
        g[edge.first].push_back(edge.second);
        g_r[edge.second].push_back(edge.first);
        bool isFromNewNode = edge.first >= node_id;
        bool isToNewNode = edge.second >= node_id;
        if (!isLinear && ((lwcnt != 1 && isToNewNode)
            ||(upcnt != 1 && isFromNewNode))){
            need_red_update_nodes.insert(edge.first);
        }
    }
    addTimer.stop("insert edge@contract_update_graph");

    transitiveReduction(need_red_update_nodes);

    node_id += subgraphs.size();
    addTimer.stop("contract_update_graph");
    return;
}

int DynamicBatching::contract_update_type(vector<unordered_set<int> >::iterator begin, vector<unordered_set<int> >::iterator end, int base_op_id, bool isLinear) {
    // 1. update distribution, type2nop, node2type, type2name, linearTypes
    addTimer.start("contract_update_type");
    int newType = --faketype_id;
    type2name[newType] = "vt_" + to_string(-newType);
    if (isLinear) linearTypes.insert(newType);
    bool hasLinear = false;
    for (auto node: *begin) 
        hasLinear = hasLinear || linearTypes.count(node2type[node]);
    
    int op_id = base_op_id;
    unordered_map<int, int> delta_distr;
    type2nop[newType] = 0;
    auto & nop = type2nop[newType];
    delta_distr[newType] = 0;
    for (auto iter = begin; iter != end; iter++){
        auto & subgraph = *iter;
        int opCnt = getDistribution(subgraph, delta_distr);
        nop = opCnt;
        if (isLinear || hasLinear) node2nop[op_id] = opCnt;
        delta_distr[newType] -= opCnt;
        node2type.push_back(newType);
        op_id++;
    }
    if (isLinear || hasLinear) nop = -1;
    distribution_ptr->update(delta_distr, -1);
    
    if (profiling_flag > 2){
        fprintf(stdout,"type2nop %s: %d\n", type2name[newType].c_str(), nop);
    }

    // in inference mode the typewise-batch-seq is obtained by cache lookup.
    if (mode == INFERENCE) return newType;


    // 2. update typewise-batch-seq
    if (isLinear) {
        int type = node2type[*begin->begin()];
        type2BatchSeq[newType] = {};
        for (size_t i = 0; i < begin->size(); i++){
            type2BatchSeq[newType].push_back(type);
        }
        type2weight[newType] = type2weight[type] * begin->size();
    }
    else {
        int min_batch;
        tie(min_batch, type2weight[newType]) = bruteForce(*begin, type2BatchSeq[newType]);
    }

    // 3. update rootStypes for training
    if (mode == TRAIN){
        assert(begin!=end);
        type2father[newType] = newType;
        unordered_set<int> erased_types;
        for (auto node: (*begin)){
            type2father[node2type[node]] = newType;
            erased_types.insert(node2type[node]);
        }
        if (profiling_flag > 0){
            fprintf(stdout, "newType %s: ", type2name[newType].c_str());
            for (auto type: erased_types) 
                fprintf(stdout, "%s, ", type2name[type].c_str());
            fprintf(stdout, "\n");
        }
        
    } 

    // 4. return the new type
    addTimer.stop("contract_update_type");
    return newType;
}

vector<int> DynamicBatching::contract(vector<unordered_set<int> > & subgraphs, const vector<int>& typeCnts, bool isLinear){
    int node_base_id = node_id;
    auto begin = subgraphs.begin();
    vector<int> newTypes;
    for (auto cnt: typeCnts){
        assert(((begin - subgraphs.begin()) + cnt) <= (int)subgraphs.size());
        int newType = contract_update_type(begin, begin+cnt, node_base_id, isLinear);
        newTypes.push_back(newType);
        node_base_id += cnt;
        begin = begin + cnt;
    }
    contract_update_graph(subgraphs, isLinear);
    return newTypes;
}

void DynamicBatching::solve(vector<int> &best_batch_seq)
{
    addTimer.start("solve");
    vector<int> best_batch_seq_from_behind;
    double minEntropy = distribution_ptr->getEntropy();
    int min_entropy_iter = 0;
    int min_entropy_type;
    unordered_set<int> opt_nodes = nodes;

    if (mode == INFERENCE){
        cache_hit_rate = pattern_cache->inference();
        double afterInferenceEntropy = distribution_ptr->getEntropy();
        if (profiling_flag > -1){
            fprintf(stdout, "[OoC::INFER]: iter %d, |V|=%d->%ld, hit rate %.3f, before entropy: %f, afther entropy: %f\n", n_inference_iteration, n_node_input, nodes.size(), cache_hit_rate, minEntropy, afterInferenceEntropy);
            // distribution_ptr->show();
            draw_graph("./pics/I"+to_string(n_inference_iteration)+"_1.gv", {&nodes}, "I"+to_string(n_inference_iteration) + "_1");
        }
        if (minEntropy > afterInferenceEntropy) {
            minEntropy = afterInferenceEntropy;
            opt_nodes = nodes;
        }
        addTimer.start("prune");
        bruteForce(nodes, best_batch_seq, LEAFPRUNE);
        bruteForce(nodes, best_batch_seq_from_behind, ROOTPRUNE);
        addTimer.stop("prune");
    }

    int n_iter = 0;
    int score_by_prune = 0;


    while (nodes.size() > LIMIT && n_iter < 100)
    {
        n_iter++;
        vector<unordered_set<int>> typical_subgraphs;
        if(profiling_flag > 1) {
            fprintf(stdout, "[ITER%d] |V| = %ld, entropy: %f.\n", n_iter, nodes.size(), distribution_ptr->getEntropy());
            if (profiling_flag > 1)
                distribution_ptr->show();
        }
        bool foundSolution = findOptPatternAndContract(typical_subgraphs, (n_iter / 100.0));

        if (!foundSolution) break;

        double entropy = distribution_ptr->getEntropy();
        if (entropy < minEntropy) {
            minEntropy = entropy;
            min_entropy_iter = n_iter;
            min_entropy_type = faketype_id;
            opt_nodes = nodes;
        }

        // prunes and visualization;
        if (mode != TRAIN){
            addTimer.start("prune");
            int tmp, prune_score;
            tie(tmp, prune_score) = bruteForce(nodes, best_batch_seq, LEAFPRUNE);
            score_by_prune += prune_score;
            tie(tmp, prune_score) = bruteForce(nodes, best_batch_seq_from_behind, ROOTPRUNE);
            score_by_prune += prune_score;
            addTimer.stop("prune");
        }

        string graphName;
        if(mode == TRAIN) graphName = "T"+to_string(n_train_iteration) + "_" + to_string(n_iter);
        else if (mode == INFERENCE) graphName = "I"+to_string(n_inference_iteration) + "_" + to_string(n_iter+1);
        else if (mode == NOCACHE) graphName = "G" + to_string(n_iter);
        vector<unordered_set<int>*> graph_ptrs = {&nodes};
        for (size_t i = 0; i < typical_subgraphs.size(); i++) 
            graph_ptrs.push_back(&typical_subgraphs[i]);
        draw_graph("./pics/" + graphName + ".gv", graph_ptrs, graphName);
        
        localTimer.clear();
    }

    nodes = opt_nodes;

    if (mode == TRAIN){
        if (profiling_flag > 1){
            fprintf(stdout, "optimal_entropy: %f\n", minEntropy);
        }
        int maxId = -1;
        for (auto node: nodes){
            maxId = max(maxId, node);
            node2father[node] = node;
        }
        node2father.resize(maxId+1);
        backPropogateNode2Father();
        forwardType2Father(min_entropy_type);
        cache_hit_rate = pattern_cache->update(type2BatchSeq);
        if (profiling_flag > -1) {
            fprintf(stdout, "[OoC::TRAIN]: iter %d, hit rate %.5f\n", n_train_iteration, cache_hit_rate);
        }
        draw_boundary_edges();
    }
    addTimer.start("prune");
    int tmp, prune_score;
    tie(tmp, prune_score) = bruteForce(nodes, best_batch_seq, LEAFPRUNE);
    score_by_prune += prune_score;
    tie(tmp, prune_score) = bruteForce(nodes, best_batch_seq_from_behind, ROOTPRUNE);
    score_by_prune += prune_score;
    addTimer.stop("prune");

    if (profiling_flag > 0)
        fprintf(stdout, "contraction finished with iter %d %ld, minimum entropy:%f, min_entropy_iter: %d\n", n_iter, nodes.size(), minEntropy, min_entropy_iter);

    // 3. find the optimal policy by backtrack;
    addTimer.start("bruteForce");
    int min_batch, min_score;
    if (profiling_flag > -1) {
        fprintf(stdout, "[SOLVE] nodes.size() %ld before bruteForce\n", nodes.size());
    }
    tie(min_batch, min_score) = bruteForce(nodes, best_batch_seq);
    min_score += score_by_prune;
    addTimer.stop("bruteForce");

    for (auto iter = best_batch_seq_from_behind.rbegin(); iter != best_batch_seq_from_behind.rend(); iter++){
        best_batch_seq.push_back(*iter);
    }

    // 4. recover the full batch
    for (int type = faketype_id; type < 0; type ++)
    {
        const vector<int> &batch_seq = type2BatchSeq[type];
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
    if (mode != NOCACHE)
        update_mode();
    addTimer.stop("solve");
    if (profiling_flag > 2)
        fprintf(stdout,"n_batch: %ld, min_score: %d\n", best_batch_seq.size(), min_score);
    if (profiling_flag > 0){
        fprintf(stdout,"[DB::solve] iter %d\n", n_iter);
        addTimer.show();
    }
    if (profiling_flag > 1){
        int idx = 0;
        for (auto batch: best_batch_seq) {
            fprintf(stdout, "\t%s", type2name[batch].c_str());
            if ((++idx) % 5 == 0) 
                fprintf(stdout, "\n");
        }
        fprintf(stdout, "\n");
    }
}

void DynamicBatching::transitiveReduction(unordered_set<int>& redNodes, bool update_cache)
{
    addTimer.start("transitiveReduction");
    localTimer.start("transitiveReduction");
    vector<pair<int, int> > erase_edges; 
    for (auto from: redNodes){
        if (g[from].size() <= 1) continue; 
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
        // fprintf(stdout,"erase %s%d %s%d\n", type2name[node2type[from]].c_str(), from, type2name[node2type[to]].c_str(), to);
        g[from].erase(find(g[from].begin(), g[from].end(), to));
        g_r[to].erase(find(g_r[to].begin(), g_r[to].end(), from));
    }

    if (update_cache) {
        pattern_cache->update_tr_edges(erase_edges);
    }
    
    localTimer.stop("transitiveReduction");
    addTimer.stop("transitiveReduction");
}

void DynamicBatching::draw_graph(string filename, initializer_list<unordered_set<int> *> subgraphs, string graphName){
    if (profiling_flag == 0) return;
    ofstream file;
    file.open(filename);

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
            
            string nodetag = type2name[(op_type)node2type[node_id]] + "_" + to_string(node_id) + "_" + to_string(topo_value[node_id]);
            for (auto to : g[node_id])
            {
                if (nodes->count(to) == 0)
                    continue;
                string totag = type2name[(op_type)node2type[to]] + "_" + to_string(to) + "_" + to_string(topo_value[to]);
                file << "\t" << nodetag << " -> " << totag;
                if (isHashed && (int)node_id < n_node_input && to < n_node_input){
                    pair<int, int> hashKey = {node_hash[node_id], node_hash[to]};
                    if (pattern_cache->boundary_edges.count(hashKey)){
                        file << "\t[color=\"red\"]";
                    }
                }
                file << ";\n";
            }

            // for (auto from: g_r[node_id]) {
            //     if (nodes->count(from) == 0) continue;
            //     string fromtag = type2name[(op_type)node2type[from]] + "_" + to_string(from) + "_" + to_string(topo_value[from]);
            //     file << "\t" << nodetag << " -> " << fromtag;
            //     string color = "grey";
            //     if (isHashed && (int)node_id < n_node_input && from < n_node_input){
            //         pair<int, int> hashKey = {node_hash[from], node_hash[node_id]};
            //         if (pattern_cache->boundary_edges.count(hashKey)){
            //             color = "red";
            //         }
            //     }
            //     file << "\t[color=\"" << color << "\"]";
            //     file << ";\n";
            // }
        }
        for (auto& kv: nodecmap){
            auto node_id = kv.first;
            string nodetag = type2name[(op_type)node2type[node_id]] + "_" + to_string(node_id) + "_" + to_string(topo_value[node_id]);
            file << "\t" << nodetag << "\t[color=\"" << kv.second << "\"];" << endl;
        }
        file << "}\n\n";
    }
    file << "}\n";
    file.close();
}

void DynamicBatching::draw_boundary_edges(){
    vector<unordered_set<int>*> boundary_edges_ptr;
    for (auto &kv: boundary_edges){
        boundary_edges_ptr.push_back(&kv.second);
    }
    string graphName = "BoundaryEdges" + to_string(n_train_iteration);
    draw_graph("./pics/" + graphName + ".gv", boundary_edges_ptr, graphName);
}

void DynamicBatching::draw_graph(string filename, vector<unordered_set<int> *>& subgraphs, string graphName, vector<int>* hashKeys){
    if (profiling_flag == 0) return;
    ofstream file;
    file.open(filename);
    // lookup, logistic, gemm, add, tanh, cmult, concatenate

    file << "digraph " << graphName << " {\n";
    int idx = 0;

    unordered_map<int, int> nid2memId;
    unordered_map<int, int> nid2bid;
    function<string(int)> get_name = [&](int nid)->string{
        string ret = type2name[(op_type)node2type[nid]] + "_" + to_string(nid) + "_" + to_string(topo_value[nid]);
        if (nid2memId.count(nid)) 
            ret += "_" + to_string(nid2memId[nid]);
        if (nid2bid.count(nid))
            ret += "_" + to_string(nid2bid[nid]);
        return ret;
    };

    for (auto& nodes: subgraphs){
        unordered_map<int, string> cmap;
        unordered_map<int, string> nodecmap;

        file << "subgraph " << graphName + to_string(idx) << "{\n";
        if (hashKeys!=nullptr){
            auto & pattern = pattern_cache->patterns[(*hashKeys)[idx]];
            // graph [label="The Tale of Two Cities", labelloc=t, fontsize=30];
            file << "\tgraph [label=\"";
            for (auto batch: pattern->batch_seq){
                file << type2name[batch] << " ";
            }
            file << "\"];\n";
            nid2memId.clear();
            nid2bid.clear();
            int min_nid = *nodes->begin();
            for (auto nid: *nodes) min_nid = min(min_nid, nid);
            int idx = 0;
            for (auto bid: pattern->mem_allocation_order){
                for (auto off: pattern->batch_ids[bid]){
                    nid2memId[off+min_nid] = idx++;
                    nid2bid[off+min_nid] = bid;      
                }
            } 
        }
        idx++;

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
            assert(type2name.count((op_type)node2type[node_id]));
            string nodetag = get_name(node_id);
            for (auto to : g[node_id])
            {
                if (nodes->count(to) == 0)
                    continue;
                string totag = get_name(to);
                file << "\t" << nodetag << " -> " << totag;
                if (isHashed && (int)node_id < n_node_input && to < n_node_input){
                    pair<int, int> hashKey = {node_hash[node_id], node_hash[to]};
                    if (pattern_cache->boundary_edges.count(hashKey)){
                        file << "\t[color=\"red\"]";
                    }
                }
                file << ";\n";
            }


            // for (auto from: g_r[node_id]) {
            //     if (nodes->count(from) == 0) continue;
            //     string fromtag = type2name[(op_type)node2type[from]] + "_" + to_string(from) + "_" + to_string(topo_value[from]);
            //     file << "\t" << nodetag << " -> " << fromtag;
            //     string color = "grey";
            //     if (isHashed && (int)node_id < n_node_input && from < n_node_input){
            //         pair<int, int> hashKey = {node_hash[from], node_hash[node_id]};
            //         if (pattern_cache->boundary_edges.count(hashKey)){
            //             color = "red";
            //         }
            //     }
            //     file << "\t[color=\"" << color << "\"]";
            //     file << ";\n";
            // }
        }
        for (auto& kv: nodecmap){
            auto node_id = kv.first;
            string nodetag = get_name(node_id);
            file << "\t" << nodetag << "\t[color=\"" << kv.second << "\"];" << endl;
        }
        file << "}\n\n";
    }
    file << "}\n";
    file.close();
}
// |----|from|to|node| 
void DynamicBatching::hashNodes(){
    addTimer.start("hashNodes");
    if (isHashed) return;
    isHashed = true;
    node_hash.resize(n_node_input);
    for (int nid = 0; nid < n_node_input; nid ++){
        node_hash[nid] = node2type[nid];
        // only use the output information to embedding the node
        for (auto to: g[nid]) {
            node_hash[nid] ^= (node2type[to] << TO_OFFSET);
        }
        // for (auto from: g_r[nid]){
        //     node_hash[nid] ^= (node2type[from] << FROM_OFFSET);
        // }
    }
    addTimer.stop("hashNodes");
}

int DynamicBatching::hashSubgraph(const unordered_set<int> & subgraph, int edgeCondition, unordered_set<pair<int, int>, hash_pair >* outEdges){
    unordered_map<int, int> hashes;
    vector<int> out_nodes;
    vector<int> in_nodes;
    assert(isHashed);

    int ret = 0;
    
    function<bool(int)> boundary_cond1 = [&](int nid){
        return nodes.count(nid) && !subgraph.count(nid);
    };
    function<bool(int)> boundary_cond2 = [&](int nid){
        return nid < n_node_input && node2type[nid] != unbatchable && !subgraph.count(nid);
    };
    function<bool(int)> cond;
    if (edgeCondition == 1) cond = boundary_cond1;
    else if (edgeCondition == 2) cond = boundary_cond2; 
    for (auto node: subgraph){
        bool isOutNode = false, isInNode = false;
        for (auto to: g[node]) {
            if (cond(to)) {
                isOutNode = true;
                out_nodes.push_back(node);
                break;
            }
        }
        for (auto from: g_r[node]){
            if (cond(from)){
                isInNode = true;
                in_nodes.push_back(node);
                break;
            }
        }
        ret += node2type[node]; 
        // if (isInNode && isOutNode) ret ^= node_hash[node] &(THIS_MASK);
        // else if (isInNode) ret ^= node_hash[node] & (TO_MASK|THIS_MASK); 
        // else if (isOutNode) ret ^= node_hash[node] & (FROM_MASK|THIS_MASK);
        // else ret ^= node_hash[node];
    }

    if (outEdges != nullptr){
        for (auto out_node: out_nodes)
            for (auto to: g[out_node]){
                if (to < n_node_input && subgraph.count(to) == 0){
                    pair<int, int> hashKey({node_hash[out_node], node_hash[to]});
                    outEdges->insert(hashKey);
                    boundary_edges[hashKey] = {};
                    boundary_edges[hashKey].insert(out_node);
                    boundary_edges[hashKey].insert(to);
                }
            }
        for (auto in_node: in_nodes)
            for (auto from: g_r[in_node]){
                if (from < n_node_input && subgraph.count(from) == 0){
                    pair<int, int> hashKey({node_hash[from], node_hash[in_node]});
                    outEdges->insert(hashKey);
                    boundary_edges[hashKey] = {};
                    boundary_edges[hashKey].insert(from);
                    boundary_edges[hashKey].insert(in_node);
                }
            }
    }


    return ret;
}

void DynamicBatching::update_mode(){
    switch (mode){
        case TRAIN:
        if (cache_hit_rate > train2inference_thres) mode = INFERENCE;
        break;
        case INFERENCE:
        if (cache_hit_rate < inference2train_thres) mode = TRAIN;
        break;
        default:
        break;
    }
}

void DynamicBatching::backPropogateNode2Father(){
    for (int nid = node2father.size() - 1; nid >= 0; --nid){
        int fatherType = node2type[node2father[nid]];
        node2father[nid] = linearTypes.count(fatherType)? nid: node2father[node2father[nid]];
    }
}

void DynamicBatching::forwardType2Father(int newest_type){
    vector<int> topo_order;

    for (int t = op_type::unbatchable; t != op_type::END; t++)
        topo_order.push_back(t);
    for (int i = -1; i >= newest_type; i--)
        topo_order.push_back(i);
    for (auto type: topo_order) {
        auto father_type = type2father[type];
        if (father_type == type || father_type < newest_type)
            root_types.insert(type);
    }
    if (profiling_flag > 0) {
        fprintf(stdout, "root types: ");
        for (auto type: root_types){
            fprintf(stdout, "%s, ", type2name[type].c_str());
        }
        fprintf(stdout, "\n");
    }
}

double Distribution::delta_entropy(unordered_map<int, int>& updates, int factor){
    double currEntropy = 0;
    double newEntropy = 0;
    int cnt = 0;
    if (profiling_flag > 1){
        fprintf(stdout,"delta_entropy(%ld, %d): ", updates.size(), factor);
        for (auto kv: updates){
            fprintf(stdout,"%s:%d %d;", type2name[kv.first].c_str(), distribution[kv.first], kv.second * factor);
        }
    }
    function<double(int)> xlgx = [](int c)->double{
        return c == 0? 0: c * std::log((double)c);
    };
    for (auto kv: updates){
        assert(distribution.count(kv.first));
        int c0 = distribution[kv.first];
        int c1 = c0 - kv.second * factor;
        currEntropy += - xlgx(c0);
        newEntropy += - xlgx(c1);
        cnt += kv.second * factor;
    }
    newEntropy -= xlgx(cnt);
    double delta_entropy = (currEntropy - newEntropy) / (double)sum;
    if (profiling_flag > 1){
        fprintf(stdout,"entropy: %.2f, %.2f, %.2f", currEntropy, newEntropy, delta_entropy);
        fprintf(stdout,"\n");
    }
    
    return delta_entropy;
}

void Distribution::update(unordered_map<int, int>& updates, int factor, bool setSum){
    if (setSum) sum = 0;
    if (profiling_flag > 3) {
        fprintf(stdout,"update parameters(%d): ", factor);
        for (auto kv: updates){
            fprintf(stdout,"%s: %d;", type2name[kv.first].c_str(), kv.second);
        }
        fprintf(stdout,"\n");
    }
    for (auto kv: updates){
        if (distribution.count(kv.first) == 0){
            distribution[kv.first] = 0;
        }
        distribution[kv.first] += kv.second * factor;
        if (setSum) sum += kv.second * factor;
    }
    
    if (profiling_flag > 2) {
        fprintf(stdout,"update: ");
        for (auto kv: distribution){
            fprintf(stdout,"%s: %d;", type2name[kv.first].c_str(), kv.second);
        }
        fprintf(stdout,"\n");
    }
}

double Distribution::getEntropy(){
    double ret = 0;
    function<double(int)> xlgx = [](int c)->double{
        return c == 0? 0: c * std::log((double)c);
    };
    for (auto kv: distribution) {
        ret += -xlgx(kv.second);
    }
    return ret / (double)sum + std::log(sum);
    return ret;
}

void Distribution::show(){
    int idx = 0;
    for (auto kv: distribution){
        if (kv.second == 0) continue;
        fprintf(stdout, "\t%s:%d;", type2name[kv.first].c_str(), kv.second);
        if ((++idx) % 4 == 0)
            fprintf(stdout, "\n");
    }
    fprintf(stdout, "\n");
}

/* TODO:
1. type2weight;
2. isophomism;
3. stop condition;
4. connect with dynet.
5. graphHash:  we only need to carry batching information. 
*/

} // namespace OoC