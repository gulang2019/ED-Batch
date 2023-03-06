#include "OoC.h"
#include "pq-trees/pqtree.h"
#include "pq-trees/ext-pqtree.h"

using namespace std;

namespace OoC
{
    double PatternCache::update(unordered_map<int, vector<int>> &type2batchSeqs)
    {
        // 0. the nodes of db is hashed
        // 1. for all virtual node in this graph, recover the origin nodes;
        //    store each subgraphs information: 1. the hash;
        // 2. dfs to find the edges at the boundary virtual node pairs;
        assert(db->isHashed);
        db->addTimer.start("update");
        // type, father, subgraph
        unordered_set<int> missed_nodes;
        unordered_map<int, unordered_map<int, unordered_set<int>>> type2subgraphs;
        for (int i = 0; i < db->n_node_input; i++)
        {
            auto father = db->node2father[i];
            auto type = db->node2type[father];
            if (db->root_types.count(type))
            { // the supernodes
                if (type2subgraphs.count(type) == 0)
                    type2subgraphs[type] = {};
                if (type2subgraphs[type].count(father) == 0)
                    type2subgraphs[type][father] = {};
                type2subgraphs[type][father].insert(i);
            }
            else
                missed_nodes.insert(i);
        }
        vector<int> types;
        for (auto &kv : type2subgraphs)
            types.push_back(kv.first);
        sort(types.begin(), types.end(), [&](int t1, int t2)
             { return type2subgraphs[t1].size() > type2subgraphs[t2].size(); });

        vector<int> hashKeys;
        vector<unordered_set<int> *> newSubgraphPtrs;
        double score = 0;
        for (auto &type : types)
        {
            int hashKey;
            auto &subgraphs = type2subgraphs[type];
            unordered_set<pair<int, int>, hash_pair> tmp_boundary_edges;
            for (auto &subgraph : subgraphs)
                hashKey = db->hashSubgraph(subgraph.second, 2, &tmp_boundary_edges);
            auto &subgraph = subgraphs.begin()->second;
            if (patterns.count(hashKey))
            { // cache hit
                score += subgraph.size() * subgraphs.size();
                if (profiling_flag > 0)
                {
                    // check for topology order;
                    for (auto &g : subgraphs)
                        topo_check(hashKey, g.second);
                }
                continue;
            }
            if (!hash_check(tmp_boundary_edges) || !hash_check(tmp_boundary_edges, subgraph))
            {
                fprintf(stdout, "[PatternCache::update::%d]: %s is dropped due to violation\n", db->n_train_iteration, type2name[type].c_str());
                for (auto &subgraph : subgraphs)
                {
                    missed_nodes.insert(subgraph.second.begin(), subgraph.second.end());
                }
                continue;
            }
            fprintf(stdout, "[PatternCache::update::%d]: %s is cached\n", db->n_train_iteration, type2name[type].c_str());

            boundary_edges.insert(tmp_boundary_edges.begin(), tmp_boundary_edges.end());
            unique_ptr<Pattern> pattern(new Pattern());
            db->bruteForce(subgraph, pattern->batch_seq, OoC::DynamicBatching::ALL, pattern.get());
            db->type2BatchSeq[type] = pattern->batch_seq;
            pattern->n_batch = (int)pattern->batch_seq.size();
            pattern->nop = (int)subgraph.size();
            pattern->id = (int)patterns.size();
            for (auto node : subgraph)
            {
                int type = db->node2type[node];
                if (pattern->distr.count(type) == 0)
                    pattern->distr[type] = 0;
                pattern->distr[type] += 1;
            }
            pattern->show();
            patterns[hashKey] = move(pattern);
            // update internel nodes
            update_internal_edges(subgraph);
            newSubgraphPtrs.push_back(&subgraph);
            hashKeys.push_back(hashKey);
        }

        score /= db->n_node_input;

        // WALK AROUND: for rest of the unmatched nodes, store them as the super node
        unordered_set<int> subgraph;
        unordered_set<int> visited;
        function<void(int)> dfs_util = [&](int nid)
        {
            if (visited.count(nid))
                return;
            subgraph.insert(nid);
            visited.insert(nid);
            for (auto to : db->g[nid])
            {
                pair<int, int> hashed_edge = {db->node_hash[nid], db->node_hash[to]};
                if (!boundary_edges.count(hashed_edge) && missed_nodes.count(to))
                    dfs_util(to);
            }
            for (auto from : db->g_r[nid])
            {
                pair<int, int> hashed_edge = {db->node_hash[from], db->node_hash[nid]};
                if (!boundary_edges.count(hashed_edge) && missed_nodes.count(from))
                    dfs_util(from);
            }
        };
        for (auto node : missed_nodes)
        {
            if (!visited.count(node))
            {
                subgraph.clear();
                dfs_util(node);
                unordered_set<pair<int, int>, hash_pair> tmp_boundary_edges;
                int hashKey = db->hashSubgraph(subgraph, 2, &tmp_boundary_edges);
                if (patterns.count(hashKey) || !hash_check(tmp_boundary_edges) || !hash_check(tmp_boundary_edges, subgraph))
                    continue;
                fprintf(stdout, "[PatternCache::update::%d] add subgraph from missed nodes: ", db->n_train_iteration);
                for (auto node : subgraph)
                    fprintf(stdout, "%s, %d; ", type2name[db->node2type[node]].c_str(), node);
                fprintf(stdout, "\n");
                boundary_edges.insert(tmp_boundary_edges.begin(), tmp_boundary_edges.end());
                unique_ptr<Pattern> pattern(new Pattern());
                db->bruteForce(subgraph, pattern->batch_seq);
                pattern->n_batch = (int)pattern->batch_seq.size();
                pattern->nop = (int)subgraph.size();
                pattern->id = (int)patterns.size();
                for (auto node : subgraph)
                {
                    int type = db->node2type[node];
                    if (pattern->distr.count(type) == 0)
                        pattern->distr[type] = 0;
                    pattern->distr[type] += 1;
                }
                pattern->show();
                patterns[hashKey] = move(pattern);
                // update internel nodes
                update_internal_edges(subgraph);
                newSubgraphPtrs.push_back(new unordered_set<int>());
                *newSubgraphPtrs.back() = subgraph;
                hashKeys.push_back(hashKey);
            }
        }

        string graphName = "Subgraph" + to_string(db->n_train_iteration);
        db->draw_graph("./pics/" + graphName + ".gv", newSubgraphPtrs, graphName, &hashKeys);
        db->addTimer.stop("update");
        return score;
    }

    void PatternCache::update_internal_edges(const unordered_set<int> &subgraph)
    {
        for (auto node : subgraph)
        {
            for (auto to : db->g[node])
            {
                if (subgraph.count(to))
                {
                    pair<int, int> hashed_edge = {db->node_hash[node], db->node_hash[to]};
                    internal_edges.insert(hashed_edge);
                }
            }
        }
        for (auto node : subgraph)
        {
            for (auto from : db->g_r[node])
            {
                if (subgraph.count(from))
                {
                    pair<int, int> hashed_edge = {db->node_hash[from], db->node_hash[node]};
                    internal_edges.insert(hashed_edge);
                }
            }
        }
    }

    bool PatternCache::hash_check(unordered_set<pair<int, int>, hash_pair> &edges)
    {
        for (auto edge : edges)
            if (internal_edges.count(edge))
                return false;
        return true;
    }

    bool PatternCache::hash_check(unordered_set<pair<int, int>, hash_pair> &edges, unordered_set<int> &subgraph)
    {
        bool valid = true;
        for (auto node : subgraph)
        {
            for (auto to : db->g[node])
            {
                if (subgraph.count(to))
                {
                    pair<int, int> hashed_edge = {db->node_hash[node], db->node_hash[to]};
                    if (boundary_edges.count(hashed_edge))
                        return false;
                }
            }
            for (auto from : db->g_r[node])
            {
                if (subgraph.count(from))
                {
                    pair<int, int> hashed_edge = {db->node_hash[from], db->node_hash[node]};
                    if (boundary_edges.count(hashed_edge))
                        return false;
                }
            }
        }
        return true;
    }

    void PatternCache::transitive_reduction(const unordered_set<int> &subgraph)
    {
        db->addTimer.start("cached transitive reduction");
        db->hashNodes();
        vector<pair<int, int>> erase_edges;
        for (auto node : subgraph)
        {
            for (auto to : db->g[node])
            {
                pair<int, int> hash_key = {db->node_hash[node], db->node_hash[to]};
                if (tr_edges.count(hash_key))
                    erase_edges.push_back({node, to});
            }
        }
        if (profiling_flag > 0)
            fprintf(stdout, "[PatternCache::transitive_reduction]:found %ld edges to cut\n", erase_edges.size());
        for (auto &edge : erase_edges)
        {
            int from = edge.first, to = edge.second;
            db->g[from].erase(find(db->g[from].begin(), db->g[from].end(), to));
            db->g_r[to].erase(find(db->g_r[to].begin(), db->g_r[to].end(), from));
        }
        db->addTimer.stop("cached transitive reduction");
    }

    double PatternCache::inference()
    {
        db->addTimer.start("inference");
        vector<bool> visited(db->n_node_input, false);
        vector<pair<int, int>> newEdges;
        unordered_set<int> subgraph;
        int hashKey = 0;

        function<void(int)> dfs = [&](int nid)
        {
            if (visited[nid])
                return;
            visited[nid] = true;
            subgraph.insert(nid);
            int inCnt = 0, outCnt = 0;
            for (auto from : db->g_r[nid])
            {
                pair<int, int> hashed_edge = {db->node_hash[from], db->node_hash[nid]};
                if (db->node2type[from] == unbatchable || tr_edges.count(hashed_edge))
                    continue;
                if (boundary_edges.count(hashed_edge))
                {
                    newEdges.push_back({from, nid});
                    inCnt++;
                }
                else
                {
                    dfs(from);
                }
            }
            // assert(inCnt == 0 || inCnt == (int)db->g_r[nid].size());
            for (auto to : db->g[nid])
            {
                pair<int, int> hashed_edge = {db->node_hash[nid], db->node_hash[to]};
                if (db->node2type[to] == unbatchable || tr_edges.count(hashed_edge))
                    continue;
                if (boundary_edges.count(hashed_edge))
                {
                    outCnt++;
                }
                else
                {
                    dfs(to);
                }
            }
            // assert(outCnt == 0 || outCnt == (int)db->g[nid].size());
            hashKey += db->node2type[nid];
            // if (inCnt && outCnt) hashKey ^= db->node_hash[nid] & (THIS_MASK);
            // else if (inCnt) hashKey ^= db->node_hash[nid] & (TO_MASK | THIS_MASK);
            // else if (outCnt) hashKey ^= db->node_hash[nid] & (FROM_MASK | THIS_MASK);
            // else hashKey ^= db->node_hash[nid];
        };

        unordered_map<int, vector<unordered_set<int>>> hashKey2Subgraphs;
        vector<unordered_set<int>> allSubgraphs;
        vector<unordered_set<int>> missedSubgraphs;
        unordered_map<int, int> hashKey2newType;
        unordered_map<int, int> hashKey2hitCnt;

        double score = 0;
        db->addTimer.start("pattern matching@inference");
        for (int nid = 0; nid < db->n_node_input; nid++)
        {
            bool isUnbatchable = db->node2type[nid] == unbatchable;
            score += isUnbatchable;
            if (!isUnbatchable && !visited[nid])
            {
                subgraph.clear();
                newEdges.clear();
                hashKey = 0;
                dfs(nid);
                if (patterns.count(hashKey))
                {
                    // do contraction on-the-fly
                    // 1. update type
                    score += subgraph.size();
                    if (hashKey2newType.count(hashKey) == 0)
                    {
                        // update type2weight, type2batchseq, type2nop by cache look up
                        hashKey2newType[hashKey] = --db->faketype_id;
                        hashKey2hitCnt[hashKey] = 0;
                        const int &newType = hashKey2newType[hashKey];
                        auto &pattern = patterns[hashKey];
                        db->type2weight[newType] = pattern->n_batch;
                        db->type2nop[newType] = pattern->nop;
                        db->type2BatchSeq[newType] = pattern->batch_seq;
                    }
                    // 2. update nodes: nodes, topo_value, node2type, node2father
                    hashKey2hitCnt[hashKey]++;
                    int type = hashKey2newType[hashKey];
                    int newNode = db->node_id++;
                    db->node2type.push_back(type);
                    db->nodes.insert(newNode);
                    db->node2father.push_back(newNode);
                    db->topo_value[newNode] = db->topo_value[*subgraph.begin()];
                    for (auto node : subgraph)
                    {
                        db->nodes.erase(node);
                        db->node2father[node] = newNode;
                    }
                    db->g.push_back({});
                    db->g_r.push_back({});
                }
                else
                    missedSubgraphs.push_back(move(subgraph));
                // update edges: the updated edges are boundary edges, so at least one side is the newNode
                unordered_set<pair<int, int>, hash_pair> inserted_edge;
                for (auto &edge : newEdges)
                {
                    pair<int, int> newEdge = {db->node2father[edge.first], db->node2father[edge.second]};
                    if (profiling_flag > 0 && newEdge.first == newEdge.second)
                    {
                        fprintf(stdout, "[WARNING::INFERENCE] (%s, %s)\n", type2name[db->node2type[edge.first]].c_str(), type2name[db->node2type[edge.second]].c_str());
                    }
                    if (inserted_edge.count(newEdge) == 0)
                    {
                        inserted_edge.insert(newEdge);
                        db->g[newEdge.first].push_back(newEdge.second);
                        db->g_r[newEdge.second].push_back(newEdge.first);
                    }
                }
                // assert(hashKey == db->hashSubgraph(subgraph, 1, nullptr));
                // if (profiling_flag > 0) allSubgraphs.push_back(subgraph);
                // if (patterns.count(hashKey)){ // cacheHit
                //     if (hashKey2Subgraphs.count(hashKey) == 0) {
                //         hashKey2Subgraphs[hashKey] = {};
                //     }
                //     hashKey2Subgraphs[hashKey].push_back(subgraph);
                //     score += subgraph.size();
                // }
                // else missedSubgraphs.push_back(move(subgraph));
            }
        }
        // update distribution_ptr
        for (auto &kv : hashKey2hitCnt)
        {
            // kv.first: hashKey; kv.second: occurance
            auto &pattern = patterns[kv.first];
            for (auto &type2cnt : pattern->distr)
            {
                db->distribution_ptr->distribution[type2cnt.first] -= type2cnt.second * kv.second;
            }
            db->distribution_ptr->distribution[hashKey2newType[kv.first]] = pattern->nop * kv.second;
        }
        db->addTimer.stop("pattern matching@inference");
        score /= db->n_node_input;

        // db->addTimer.start("contract@inference");
        // vector<unordered_set<int> > subgraphs;
        // vector<int> typeCnts;
        // for (auto & kv: hashKey2Subgraphs) {
        //     subgraphs.insert(subgraphs.end(), kv.second.begin(), kv.second.end());
        //     typeCnts.push_back(kv.second.size());
        // }
        // vector<int> newTypes = db->contract(subgraphs, typeCnts, false);
        // db->addTimer.stop("contract@inference");
        // db->addTimer.start("cache lookup@inference");
        // int idx = 0;
        // for (auto &kv: hashKey2Subgraphs){
        //     int newType = newTypes[idx++];
        //     db->type2BatchSeq[newType] = patterns[kv.first]->batch_seq;
        // }
        // db->addTimer.stop("cache lookup@inference");

        if (profiling_flag > 0)
        {
            string graphname = "SI" + to_string(db->n_inference_iteration);
            vector<unordered_set<int> *> tmp;
            for (auto &kv : hashKey2Subgraphs)
                tmp.push_back(&kv.second[0]);
            db->draw_graph("./pics/" + graphname + ".gv", tmp, graphname);

            graphname = "AI" + to_string(db->n_inference_iteration);
            tmp.clear();
            for (auto &subgraph : allSubgraphs)
                tmp.push_back(&subgraph);
            db->draw_graph("./pics/" + graphname + ".gv", tmp, graphname);

            graphname = "MissedSubgraph" + to_string(db->n_inference_iteration + db->n_train_iteration);
            tmp.clear();
            for (auto &subgraph : missedSubgraphs)
                tmp.push_back(&subgraph);
            db->draw_graph("./pics/" + graphname + ".gv", tmp, graphname);
        }

        db->addTimer.stop("inference");
        return score;
    }

    double PatternCache::update_tr_edges(const vector<pair<int, int>> &new_tr_edges)
    {
        db->addTimer.start("update_tr_edges");
        db->hashNodes();
        double score = 0;
        unordered_set<pair<int, int>, hash_pair> new_edges;
        for (auto edge : new_tr_edges)
        {
            pair<int, int> hashed_edge = {db->node_hash[edge.first], db->node_hash[edge.second]};
            if (tr_edges.count(hashed_edge))
            {
                if (new_edges.count(hashed_edge) == 0)
                    score++;
            }
            else
            {
                new_edges.insert(hashed_edge);
                tr_edges.insert(hashed_edge);
            }
        }
        score /= new_tr_edges.size();
        if (profiling_flag > 0)
        {
            bool noRedundancy = true;
            for (auto node : db->nodes)
            {
                for (auto to : db->g[node])
                {
                    pair<int, int> hashed_edge = {db->node_hash[node], db->node_hash[to]};
                    if (tr_edges.count(hashed_edge))
                        noRedundancy = false;
                }
            }
            fprintf(stdout, "[PatternCache::update_tr_edges]: hit rate %f; %ld; no redundant edges: %d;", score, tr_edges.size(), noRedundancy);
            // for (auto edge: tr_edges){
            //     fprintf(stdout, "(%s, %s)", type2name[db->node_hash[edge.first]&THIS_MASK].c_str(), type2name[db->node_hash[edge.second]&THIS_MASK].c_str());
            // }
            fprintf(stdout, "\n");
        }
        db->addTimer.stop("update_tr_edges");
        return score;
    }

    bool PatternCache::topo_check(int hashKey, std::vector<int> &subgraph)
    {
        assert(patterns.count(hashKey));
        auto &pattern = patterns[hashKey];
        if (subgraph.size() != pattern->nodes.size())
            return false;
        sort(subgraph.begin(), subgraph.end());
        for (int i = 0; i < subgraph.size(); i++)
        {
            if (subgraph[i] - subgraph[0] != pattern->nodes[i])
                return false;
            if (db->node2type[subgraph[i]] == pattern->types[i])
                return false;
        }
        return true;
    }
    bool PatternCache::topo_check(int hashKey, const std::unordered_set<int> &subgraph)
    {
        std::vector<int> subgraph_vec(subgraph.begin(), subgraph.end());
        return topo_check(hashKey, subgraph_vec);
    }
    void Pattern::show()
    {
        fprintf(stdout, "id: %d, n_batch: %d, nop: %d\n", id, n_batch, nop);
        fprintf(stdout, "memory layout: ");
        for (auto bid : mem_allocation_order)
        {
            fprintf(stdout, "[%d (", bid);
            for (auto nid : batch_ids[bid])
            {
                fprintf(stdout, "%s_%d, ", type2name[types[nid]].c_str(), nid);
            }
            fprintf(stdout, ")]");
        }
        fprintf(stdout, "\n");
    }

    void PatternCache::get_batch_ids_ooc(const std::vector<std::vector<int>> &node2args, const std::vector<int> &node2type, Pattern *pattern)
    {
        vector<vector<int>> node2succs(node2args.size());
        vector<int> input_cnt;
        vector<int> types;
        unordered_map<int, int> type_cnt;

        unordered_map<int, vector<int>> type2frontiers;
        for (int nid = 0; nid < (int)node2args.size(); nid++)
        {
            int internal_input_cnt = 0;
            for (auto arg : node2args[nid])
                if (arg >= 0)
                {
                    node2succs[arg].push_back(nid);
                    ++internal_input_cnt;
                }
            int tid = node2type[nid];
            if (type2frontiers.count(tid) == 0)
            {
                type2frontiers[tid] = {};
                types.push_back(tid);
            }
            if (internal_input_cnt == 0)
            {
                type2frontiers[tid].push_back(nid);
            }

            input_cnt.push_back(internal_input_cnt);
            if (type_cnt.count(node2type[nid]) == 0) type_cnt[node2type[nid]] = 0;
            type_cnt[node2type[nid]] += 1;
        }

        vector<int> node2depth(node2args.size(), 0);        
        for (auto tid: types){
            vector<int> depth(node2args.size(), 0);
            for (int nid = (int)node2args.size() - 1; nid >= 0; --nid){
                int is_the_type = (node2type[nid] == tid);
                depth[nid] = is_the_type;
                for (auto succ: node2succs[nid]){
                    assert(succ > nid);
                    depth[nid] = max(depth[nid], is_the_type + depth[succ]);
                }
                if (is_the_type) node2depth[nid] = depth[nid];
            }
        }

        vector<vector<int>> batches(node2args.size());

        if (profiling_flag > 0)
        {
            int nid = 0;
            for (auto &args : node2args)
            {
                fprintf(stdout, "%s_%d[%d]: ", type2name[node2type[nid]].c_str(), nid, input_cnt[nid]);
                for (auto arg : args)
                {
                    if (arg >= 0)
                        fprintf(stdout, "%s_%d, ", type2name[node2type[arg]].c_str(), arg);
                    else
                        fprintf(stdout, "%d, ", arg);
                }
                fprintf(stdout, "||");
                for (auto arg : node2succs[nid])
                {
                    if (arg >= 0)
                        fprintf(stdout, "%s_%d, ", type2name[node2type[arg]].c_str(), arg);
                    else
                        fprintf(stdout, "%d, ", arg);
                }
                fprintf(stdout, "\n");
                nid++;
            }
        }
        if (profiling_flag > 0)
        {
            fprintf(stdout, "type2frontiers:\n");
            for (auto &kv : type2frontiers)
            {
                fprintf(stdout, "\t%s: ", type2name[kv.first].c_str());
                for (auto nid : kv.second)
                    fprintf(stdout, "%s_%d: ", type2name[node2type[nid]].c_str(), nid);
                fprintf(stdout, "\n");
            }
        }

        function<bool(int, list<int> &)> memory_allocation = [&](int idx, list<int> &mem_ids)
        {
            set<int> S;
            for (int i = 0; i < (int)node2args.size(); i++)
                S.insert(i);
            vector<vector<vector<int>>> patterns;
            if (profiling_flag > 1)
            {
                fprintf(stdout, "[memory_allocation] batches: ");
                for (int bid = 0; bid < idx; bid++)
                {
                    auto &subset = batches[bid];
                    fprintf(stdout, "(");
                    for (auto ele : subset)
                        fprintf(stdout, "%d, ", ele);
                    fprintf(stdout, ")");
                }
                fprintf(stdout, "\n");
            }
            set<set<int>> consecutive_constraints;
            for (int bid = 0; bid < idx; bid++)
            {
                auto &batch = batches[bid];
                if (batch.size() <= 1)
                    continue;
                consecutive_constraints.insert(set<int>(batch.begin(), batch.end()));
                if (node2args[batch.front()].size() < 1)
                    continue;
                patterns.push_back({});
                auto &pattern = patterns.back();
                pattern.push_back(batch);
                for (int aid = 0; aid < node2args[batch.front()].size(); aid++)
                {
                    pattern.push_back({});
                    bool no_external_input = true;
                    for (auto nid : batch)
                    {
                        auto arg = node2args[nid][aid];
                        no_external_input = no_external_input && (arg >= 0);
                        pattern.back().push_back(arg);
                    }
                    if (!no_external_input)
                        pattern.pop_back();
                }
            }
            IsoPQTree tree(node2args.size());

            if (profiling_flag > 0)
            {
                fprintf(stdout, "consecutive_constraints: ");
                for (auto &subset : consecutive_constraints)
                {
                    fprintf(stdout, "(");
                    for (auto ele : subset)
                        fprintf(stdout, "%d, ", ele);
                    fprintf(stdout, ")");
                }
                int pid = 0;
                fprintf(stdout, "\n");
                fprintf(stdout, "patterns:\n");
                for (auto &pattern : patterns)
                {
                    fprintf(stdout, "\t%d", pid++);
                    for (auto &subset : pattern)
                    {
                        fprintf(stdout, "(");
                        for (auto ele : subset)
                            fprintf(stdout, "%d, ", ele);
                        fprintf(stdout, "), ");
                    }
                    fprintf(stdout, "\n");
                }
            }
            
            if (!tree.ReduceAll(consecutive_constraints) || !tree.IsoReduceAll(patterns))
            {
                if (profiling_flag > -1) cout << "isoReduce failed..." << endl;
                return false;
            }
            mem_ids = tree.Frontier();
            if (profiling_flag > 0)
            {
                cout << "tree: " << tree.Print() << endl;
                cout << "frontiers: ";
                for (auto mid: mem_ids) cout << mid <<",";
                cout << endl;
            }
            return true;
        };

        // brute force search for the best batching policy;
        // use 
        // TupleDict<int> memo;
        vector<bool> visited(node2args.size(), false);
        function<void(int)> dfs = [&](int idx)
        {
            vector<int> state;
            int batch_lower_bound = 0;
            unordered_map<int, int> per_type_max_depth;
            for (int nid = 0; nid < visited.size(); nid++){
                if (visited[nid]) continue;
                int type = node2type[nid];
                if (per_type_max_depth.count(type) == 0)
                    per_type_max_depth[type] = 0;
                per_type_max_depth[type] = max(per_type_max_depth[type], node2depth[nid]);
            }
            for (auto & kv: per_type_max_depth) batch_lower_bound += kv.second;

            for (auto &tid : types)
            {
                auto &frontier = type2frontiers[tid];
                state.insert(state.end(), frontier.begin(), frontier.end());
            }
            sort(state.begin(), state.end());
            // fprintf(stdout, "dfs: ");
            // for (auto &ele : state)
            //     fprintf(stdout, "%d, ", ele);
            // fprintf(stdout, ": (%d, %d)\n", idx + batch_lower_bound, pattern->n_batch);
            if (idx + batch_lower_bound >= pattern->n_batch) return; 
            
            vector<int> candidate_types;
            for (auto &tid : types)
            {
                if (type2frontiers[tid].size()){
                    if (type_cnt[tid] == 1) {
                        candidate_types = {tid};
                        break;
                    }
                    candidate_types.push_back(tid);
                }
            }            

            for (auto &tid : candidate_types)
            {
                batches[idx] = move(type2frontiers[tid]);
                for (auto nid : batches[idx])
                {
                    for (auto succ : node2succs[nid])
                    {
                        if (--input_cnt[succ] == 0)
                        {
                            type2frontiers[node2type[succ]].push_back(succ);
                        }
                    }
                }
                for (auto nid: batches[idx]) visited[nid] = true;
                type_cnt[tid] -= batches[idx].size();
                dfs(idx + 1);
                type_cnt[tid] += batches[idx].size();
                for (auto nid: batches[idx]) visited[nid] = false;

                for (auto iter = batches[idx].rbegin(); iter != batches[idx].rend(); iter++)
                {
                    int nid = *iter;
                    for (auto succ_iter = node2succs[nid].rbegin(); succ_iter != node2succs[nid].rend(); succ_iter++)
                    {
                        int succ = *succ_iter;
                        if (input_cnt[succ]++ == 0)
                        {
                            type2frontiers[node2type[succ]].pop_back();
                        }
                    }
                }
                type2frontiers[tid] = move(batches[idx]);
            }
            if (!candidate_types.size() && idx < pattern->n_batch)
            {
                list<int> mem_ids;
                if(!memory_allocation(idx, mem_ids)){
                    cerr << "[pattern_cache::WARNING]: memory allocation incomplete!" << endl;
                }
                if (OoC::profiling_flag){
                    cout << "[pattern_cache] found #batches: " << idx << endl;
                }
                pattern->n_batch = idx;
                unordered_map<int, int> nid2bid;
                for (int bid = 0; bid < idx; bid++)
                {
                    for (auto &nid : batches[bid])
                        nid2bid[nid] = bid;
                }
                pattern->mem_allocation_order.clear();
                pattern->batch_ids.clear();
                pattern->batch_ids.resize(idx);
                auto iter = mem_ids.begin();
                while (iter != mem_ids.end())
                {
                    int bid = nid2bid[*iter];
                    pattern->mem_allocation_order.push_back(bid);
                    while (iter != mem_ids.end() && nid2bid[*iter] == bid)
                    {
                        pattern->batch_ids[bid].push_back(*iter);
                        iter++;
                    }
                }
            }
        };

        pattern->n_batch = 1e6;
        dfs(0);
        if (profiling_flag > 1)
            fprintf(stdout, "get_ids finished!\n");
    }

    
    Pattern *PatternCache::add_pattern(int key, const vector<vector<int>> &node2args, const vector<int> &node2types,
    std::string alg)
    {
        if (patterns.count(key)){
            fprintf(stdout, "[ADD PATTERN]: add existed pattern %d\n", key);
            return patterns[key].get();
        }
        if (profiling_flag > 0)
        {
            int nid = 0;
            fprintf(stdout, "add_pattern: \n");
            for (auto &args : node2args)
            {
                fprintf(stdout, "\t%s_%d: ", type2name[node2types[nid]].c_str(), nid);
                for (auto &arg : args)
                {
                    if (arg >= 0)
                        fprintf(stdout, "%s_%d, ", type2name[node2types[arg]].c_str(), arg);
                    else
                        fprintf(stdout, "%d, ", arg);
                }
                fprintf(stdout, "\n");
                nid++;
            }

            ofstream file;
            file.open("./pics/BB" + to_string(key) + ".gv");
            file << "digraph BB" + to_string(key) + "{\n";
            file << "\tnode [style=filled]\n";
            unordered_map<int, string> tid2color;
            function<string(int)> get_name = [&](int nid)
            {
                return type2name[node2types[nid]] + "_" + to_string(nid);
            };

            nid = 0;
            for (auto &args : node2args)
            {
                if (tid2color.count(node2types[nid]) == 0) {
                    char tmp[10];
                    sprintf(tmp, "#%2x%2x%2x", rand() & 0xff, rand() & 0xff, rand() & 0xff);
                    tid2color[node2types[nid]] = string(tmp);
                }
                auto this_node_name = get_name(nid);
                file << this_node_name << "\t[color=\"" << tid2color[node2types[nid]] << "\"]" << endl;
                for (auto arg : args)
                {
                    if (arg >= 0)
                    {
                        auto that_node_name = get_name(arg);
                        file << that_node_name << "->" << this_node_name << endl;
                    }
                }
                nid += 1;
            }
            file << "}\n";
            file.close();
        }
        unique_ptr<Pattern> pattern = make_unique<Pattern>();
        pattern->nop = node2args.size();
        pattern->types = node2types;
        get_batch_ids(node2args, pattern->types, pattern.get(), alg);
        if (profiling_flag > 0)
            pattern->show();
        if (profiling_flag > 1)
        {
            ofstream file;
            file.open("./pics/BB" + to_string(key) + ".gv");
            file << "digraph BB" + to_string(key) + "{\n";
            file << "\tnode [style=filled]\n";
            unordered_map<int, int> nid2bid;
            unordered_map<int, int> nid2mid;
            unordered_map<int, string> bid2color;
            int mid = 0;
            for (auto bid : pattern->mem_allocation_order)
            {
                for (auto nid : pattern->batch_ids[bid])
                {
                    nid2bid[nid] = bid;
                    nid2mid[nid] = mid++;
                }
                char tmp[10];
                sprintf(tmp, "#%2x%2x%2x", rand() & 0xff, rand() & 0xff, rand() & 0xff);
                bid2color[bid] = string(tmp);
            }
            function<string(int)> get_name = [&](int nid)
            {
                return type2name[node2types[nid]] + "_" + to_string(nid) + "_" + to_string(nid2bid[nid]) + "_" + to_string(nid2mid[nid]);
            };
            int nid = 0;

            for (auto &args : node2args)
            {
                auto this_node_name = get_name(nid);
                file << this_node_name << "\t[color=\"" << bid2color[nid2bid[nid]] << "\"]" << endl;
                for (auto arg : args)
                {
                    if (arg >= 0)
                    {
                        auto that_node_name = get_name(arg);
                        file << that_node_name << "->" << this_node_name << endl;
                    }
                }
                nid += 1;
            }
            file << "}\n";
            file.close();
        }
        patterns[key] = move(pattern);
        return patterns[key].get();
    }

    Pattern PatternCache::get_pattern(const vector<vector<int>> &node2args, const vector<int> &node2types,
    std::string alg) {
        Pattern pattern;
        pattern.nop = node2args.size();
        pattern.types = node2types;
        get_batch_ids(node2args, pattern.types, &pattern, alg);
        
        return move(pattern);
    }

    void PatternCache::get_batch_ids_dynet(const vector<vector<int>> &node2args, const vector<int>& node2types, Pattern *pattern){
        vector<int> node2depth(node2args.size(), 0);
        vector<vector<int> > node2succs(node2args.size());
        vector<int> pred_cnt(node2args.size(), 0);
        struct type_t {
            int cnt = 0;
            int total_depth = 0;
            vector<int> frontiers;
        };
        unordered_map<int, type_t> types;
        for (int nid = 0; nid < (int) node2args.size(); nid++){
            auto & depth = node2depth[nid];
            depth = 0;
            for (auto arg: node2args[nid]) {
                if (arg < 0) continue;
                depth = std::max(depth, node2depth[arg] + 1);
                node2succs[arg].push_back(nid);
                pred_cnt[nid] ++;
            }
            int tid = node2types[nid];
            auto & type = types[tid];
            if (pred_cnt[nid] == 0) type.frontiers.push_back(nid);
            type.cnt++;
            type.total_depth+=depth;
        }
    

        while (true){
            double min_depth = 1e9;
            int tid = 0;
            for (auto & kv: types){
                if (!kv.second.frontiers.size()) continue;
                if ((kv.second.total_depth / (0.0 + kv.second.cnt)) < min_depth){
                    tid = kv.first;
                    min_depth = kv.second.total_depth / (0.0 + kv.second.cnt);
                } 
            }    
            if(tid == 0) break;
            auto & type = types[tid];
            type.cnt -= type.frontiers.size();
            for (auto nid: type.frontiers) type.total_depth -= node2depth[nid];
            pattern->batch_ids.push_back(move(type.frontiers));
            for (auto nid: pattern->batch_ids.back()){
                for (auto nnid: node2succs[nid]){
                    if (--pred_cnt[nnid] == 0) 
                        types[node2types[nnid]].frontiers.push_back(nnid);
                }
            }
        }
        pattern->n_batch = pattern->batch_ids.size();
        for(int bid = 0; bid < pattern->n_batch; bid++) pattern->mem_allocation_order.push_back(bid);
    }

    void PatternCache::get_batch_ids(const std::vector<std::vector<int>> &node2args, 
    std::vector<int>& node2types, Pattern *pattern, string alg){
        if (alg == "ooc"){
            get_batch_ids_ooc(node2args, node2types, pattern);
        }
        else if (alg == "dynet"){
            get_batch_ids_dynet(node2args, node2types, pattern);
        }
        else {
            throw((string("not supported alg:") + alg).c_str());
        }
    }

} // namespace OoC