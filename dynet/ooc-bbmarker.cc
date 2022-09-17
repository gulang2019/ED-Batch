#include "dynet.h"
#include "timing.h"

using namespace dynet;
using namespace OoC;
using namespace std;

namespace dynet {
    
    SigMap ComputationGraph::sigmap; // static of computation graph
    vector<typeInfo> ComputationGraph::stypes; // static of computation graph
    TupleDict<int> ComputationGraph::stype_dict; // static of computation graph
    PatternCache ComputationGraph::pattern_cache; // static of computation graph

    void ComputationGraph::mark_basic_block(){
        if (autobatch_flag != 7) return;
        if (profiling_flag) timer.start("bbmark");
        vector<int> node2type;
        unordered_set<int> preds;
        int n_unbatchable = 0;
        int sid = snodes.size();
        for (int nid = n_marked_node; nid < nodes.size(); nid++){
            auto node = nodes[nid];
            int sig = node->autobatch_sig(*this, sigmap);
            if (sig == 0){
                n_unbatchable++;
                assert(!node->args.size());
                unbatchable_ops.push_back(nid);
                nid2sid.push_back(-1);
                continue;
            }
            for (auto arg: node->args){
                int arg_sid = nid2sid[arg];
                if (arg_sid >= 0 && arg < n_marked_node) 
                    preds.insert(arg_sid);
            }
            node2type.push_back(sig);
            nid2sid.push_back(sid);
        }
        if (n_unbatchable == (nodes.size() - n_marked_node)) {
            n_marked_node = nodes.size();
            if (profiling_flag) timer.stop("bbmark");
            return;
        }
        assert(n_unbatchable == 0);
        // if (n_unbatchable){
        //     for (int nid = n_marked_node; nid < nodes.size(); nid++){
        //         int sig = nodes[nid]->autobatch_sig(*this, sigmap);
        //         fprintf(stderr, "%d, %d, %s: %s\n", nid, sig, type2name[sigmap.sig2type(sig)].c_str(), nodes[nid]->as_dummy_string().c_str());
        //     }
        // }
        // assert(n_unbatchable == 0);
        
        snodes.push_back({});
        auto & snode = snodes.back();
        snode.inputCnt = preds.size();
        snode.min_nid = n_marked_node;
        for (auto pred: preds) 
            snodes[pred].succs.push_back(sid);

        if (!stype_dict.count(node2type)){
            int stid = stypes.size();
            stype_dict[node2type] = stid;
            stypes.push_back({});
            vector<vector<int> > node2args;
            for (int nid = n_marked_node; nid < nodes.size(); nid++)
            {
                auto node = nodes[nid];
                node2args.push_back({});
                for (auto arg : node->args)
                    node2args.back().push_back(arg - n_marked_node);
            }
            fprintf(stdout, "add pattern %d, %ld\n", stid, node2args.size());
            stypes.back().pattern = pattern_cache.add_pattern(stid, node2args, node2type);
            assert(stypes.back().pattern);
            fprintf(stdout, "add pattern finished!\n");
        }
        int stid = stype_dict[node2type];
        snode.type = stid;
        auto & stype = stypes[stid];
        if (stid >= n_stored_stypes) {
            n_new_ops += stype.pattern->nop;
            if (n_new_ops > 1)
                schedule_mode = TRAIN; 
        }
        if (!preds.size()) stype.frontiers.push_back(sid);
        stype.cnt++;

        n_marked_node = nodes.size();
        if (profiling_flag) timer.stop("bbmark");
    }

} // namespace dynet


/*

computation graph: mark_basic_block
    cg.snodes, 
stypes, 
study from computation graph:
computation graph executor static scheduler


*/