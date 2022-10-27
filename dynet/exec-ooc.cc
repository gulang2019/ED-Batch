#include "exec.h"
#include "dynet/timing.h"
#include <functional>
using namespace std;
using namespace OoC;

namespace dynet{
struct nodeInfo {
    vector<VariableIndex> preds;
    vector<int> succs;
    int hash = 0;
    int type;
    bool visited = false;
    int supernode_id = -1;
    int bid = -1;
    int mid;
    int offset;
};

struct supernodeInfo{
    // unordered_set<int> preds;
    vector<int> succs;
    int type;
    int inputCnt;
    int dirtyInputCnt;
    int bid;
    int offset;
};

struct typeInfo{
    int weight;
    int cnt = 0;
    int pureNodeCnt = 0;
    vector<int> frontiers;
    vector<int> dims;
};
// output: batch_id, node2size, batches[...].ids
void BatchedExecutionEngine::getBatches(VariableIndex upto, VariableIndex & batch_id){
    // 1. construct the reverse computation graph and node_hash
    vector<nodeInfo> nodes(upto+1);
    fprintf(stderr, "n_node: %d\n", upto);
    NamedTimer localTimer;
    
    int sig;
    int old_num_nodes_evaluated = num_nodes_evaluated;
    localTimer.start("graph constrcution");

    while (num_nodes_evaluated <= upto){
        Node * node = cg.nodes[num_nodes_evaluated];
        sig = node->autobatch_sig(cg, sigmap);
        if (sig) break;
        num_nodes_evaluated++;
    }
    for (VariableIndex j = num_nodes_evaluated; j <= upto; ++j){
        Node * node = cg.nodes[j];
        sig = node->autobatch_sig(cg, sigmap);
        node2size[j] = node->dim.size();
        nodes[j].type = nodes[j].hash = sig;
        nodes[j].preds = node->args;
        for (auto arg: node->args){
            nodes[arg].succs.push_back(j);
            nodes[arg].hash ^= (sig << TO_OFFSET);
        }
    }
    localTimer.stop("graph constrcution");

    // 2. construct the super node graph out of the node information
    localTimer.start("super graph constrcution");
    vector<int> subgraph;
    unordered_set<int> preds;
    int hashKey;
    function <void (int)> dfs = [&](int nid){
        // fprintf(stdout, "visit %d\n", nid);
        if (nodes[nid].visited) return;
        subgraph.push_back(nid);
        nodes[nid].visited = true;
        for (auto from : nodes[nid].preds){
            if (nodes[from].supernode_id >= 0){ // input node
                preds.insert(nodes[from].supernode_id);
                // A flag marks the input
                nodes[nid].bid = -2; 
            }
            else if (from >= num_nodes_evaluated) dfs(from);
        }
        if (!nodes[nid].preds.size()) nodes[nid].bid = -2;
        for (auto iter = nodes[nid].succs.begin(); iter != nodes[nid].succs.end();){
            auto to = *iter;
            pair<int, int> hashed_edge = {nodes[nid].hash, nodes[to].hash};
            if (db.pattern_cache->tr_edges.count(hashed_edge)){
              iter = nodes[nid].succs.erase(iter);
              nodes[to].preds.erase(find(nodes[to].preds.begin(), nodes[to].preds.end(), nid));
              continue;
            }
            else iter++;
            if (db.pattern_cache->boundary_edges.count(hashed_edge)) continue;
            dfs(to);
        }
        hashKey += nodes[nid].type;
    };

    vector<supernodeInfo> snodes;
    vector<typeInfo> types;
    unordered_map<int, int> hashKey2typeId; 

    for (VariableIndex nid = num_nodes_evaluated; nid <= upto; ++nid){
        if (!nodes[nid].visited) {
            subgraph.clear();
            preds.clear();
            hashKey = 0;
            localTimer.start("dfs");
            dfs(nid);
            sort(subgraph.begin(), subgraph.end());
            
            localTimer.stop("dfs");
            if (db.pattern_cache->patterns.count(hashKey)){
                auto & pattern = db.pattern_cache->patterns[hashKey];
                const int supernode_id = (int)snodes.size();
                bool isNewType = false;
                if (hashKey2typeId.count(hashKey) == 0){
                    isNewType = true;
                    hashKey2typeId[hashKey] = types.size();
                    types.push_back({});
                    types.back().weight = pattern->n_batch;
                }
                const int& stid = hashKey2typeId[hashKey];
                auto & stype = types[stid];
                vector<int> dims(stype.weight, 0);
                int maxBid = 0;
                for (auto & nid: subgraph){ // this is a topo order
                    auto & node = nodes[nid];
                    const auto & type = node.type;
                    auto & bid = node.bid;
                    node.supernode_id = supernode_id;
                    
                    if (bid == -2) bid = 0; // the input node
                    else { 
                        bid = 0;
                        for (auto from: node.preds) {
                            if (nodes[from].supernode_id != supernode_id)
                                break;
                            bid = max(bid, 1 + nodes[from].bid);
                        }
                    }
                    while (pattern->batch_seq[bid] != type) bid++;
                    node.offset = dims[bid];
                    dims[bid] ++;
                    maxBid = max(bid, maxBid);
                }
                
                if (maxBid != pattern->n_batch - 1){
                  fprintf(stderr, "[ERROR]: maxBid %d, pattern->weight %d\n", maxBid, pattern->n_batch);
                  for (auto node: subgraph){
                    fprintf(stderr, "[%s, %d, %d]\t:", type2name[nodes[node].type].c_str(), nodes[node].bid, node);
                    for (auto from: nodes[node].preds) {
                        // if (nodes[from].supernode_id != supernode_id)
                        //     break;
                        fprintf(stderr, "(%s %d),", type2name[nodes[from].type].c_str(), nodes[from].bid );
                    }
                    fprintf(stderr, "\n");
                  }
                  assert(false);
                }
                

                if (isNewType) stype.dims = dims;
                snodes.push_back({});
                auto & snode = snodes.back();
                snode.inputCnt = preds.size();
                
                snode.type = stid;
                stype.cnt += 1;
                if (preds.size() == 0) 
                    stype.frontiers.push_back(supernode_id);
                
                for (auto pred: preds) {
                    snode.dirtyInputCnt += snodes[pred].type != stid;
                    snodes[pred].succs.push_back(supernode_id);
                }
                stype.pureNodeCnt += snode.dirtyInputCnt == 0;
            }
            else {
                // shouldn't cache miss
                int idx = 0;
                fprintf(stdout, "[WARNING]: unmatched pattern\n");
                for (auto node: subgraph) {
                    fprintf(stdout, "(%s, %d) ", type2name[nodes[node].type].c_str(), node);
                    if ((++idx) % 4 == 0) fprintf(stdout, "\n");
                }
                fprintf(stdout, "\n");
                assert(false);
                return;
            }
        }
    }
    localTimer.stop("super graph constrcution");

    // 3. do the scheduling 
    localTimer.start("super graph scheduling");
    // (1) prune rule1, rule2 ==> (dfs)
    int batchId = batch_id + num_nodes_evaluated - old_num_nodes_evaluated;
    bool hasUpdate = true;
    int useHeuristic = 0;
    function <void(int)> update = [&](int tid){
        hasUpdate = true;
        auto & frontierType = types[tid];
        vector<int> snode_batch = frontierType.frontiers;
        frontierType.cnt -= snode_batch.size();
        frontierType.frontiers.clear();
        int offset = 0;
        for (auto nid: snode_batch){
            auto & snode = snodes[nid];
            snode.bid = batchId;
            snode.offset = offset++;
            frontierType.pureNodeCnt -= (snode.dirtyInputCnt == 0);
            for (auto& succ: snode.succs){
                auto & succNode = snodes[succ];
                if (--succNode.inputCnt == 0) {
                    types[succNode.type].frontiers.push_back(succ);
                }
                if (succNode.type != snode.type && (--succNode.dirtyInputCnt == 0)) {
                    types[succNode.type].pureNodeCnt++;
                }
            }
        }   
        for (int bid = batchId; bid < batchId + frontierType.weight; bid++){
            auto & node_batch = batches[bid];
            node_batch.dim = snode_batch.size();
            node_batch.ids.resize(frontierType.dims[bid - batchId] * snode_batch.size());
        }
        batchId += frontierType.weight;
    }; 

    while (hasUpdate){
        while (hasUpdate){
            hasUpdate = false;
            int frontierTypeCnt = 0, frontierTypeIdx;
            for (int idx = 0; idx < (int)types.size(); idx++){
                auto & type = types[idx];
                if (type.frontiers.size()) {
                    ++ frontierTypeCnt;
                    frontierTypeIdx = idx;
                }
            }
            if (frontierTypeCnt == 1)
                update(frontierTypeIdx);

            for (int idx = 0; idx < (int) types.size(); idx++){
                auto & type = types[idx];
                if (type.pureNodeCnt == type.cnt){
                    while(type.cnt)
                        update(idx);
                }
            }
        }

        for (int idx = 0; idx < (int) types.size(); idx++){
            auto & type = types[idx];
            if (type.frontiers.size()){
                ++useHeuristic;
                update(idx);
                break;
            }
        }
    }
    if(OoC::profiling_flag > -1){
        fprintf(stdout, "[getBatch] use heuristic %d times\n", useHeuristic);
    }
    localTimer.stop("super graph scheduling");
    // 4. set the batch seq by scan
    localTimer.start("result generation");
    batches.resize(batchId);
    for (int i = old_num_nodes_evaluated; i < num_nodes_evaluated; i++){
        int bid = batch_id + i - old_num_nodes_evaluated;
        batches[bid].ids.resize(1, i);
        node2batch[i] = bid;
    }
    for (int nid = num_nodes_evaluated; nid <= upto; nid++){
        auto & node = nodes[nid];
        auto & snode = snodes[node.supernode_id];
        int bid = node.bid + snode.bid;
        auto & batch = batches[bid];

        batch.ids[node.offset * batch.dim + snode.offset] = nid;
        node2batch[nid] = bid;
    }
    localTimer.stop("result generation");
    localTimer.show();
    for (int bid = batch_id; bid < batchId; bid ++){
        fprintf(stderr, "batch %d: ",  bid);
        for (auto nid: batches[bid].ids) {
            auto & node = nodes[nid];
            if (node.supernode_id == -1) continue;
            auto & snode = snodes[node.supernode_id];
            auto & stype = types[snode.type];
            assert(node.bid < stype.dims.size());
            fprintf(stderr, "(%s,%d,%d,%d,%d,%d), ", 
                type2name[nodes[nid].type].c_str(), 
                nid, 
                node.offset, 
                node.supernode_id, 
                snode.type, 
                stype.dims[node.bid]);
        }
        fprintf(stderr, "\n");
    }
    batch_id = batchId;
}

} // namespace dynet