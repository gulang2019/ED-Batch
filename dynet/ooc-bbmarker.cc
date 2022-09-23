#include "dynet.h"
#include "timing.h"

using namespace dynet;
using namespace OoC;
using namespace std;

namespace dynet
{

    SigMap ComputationGraph::sigmap;                // static of computation graph
    vector<OoC::typeInfo> ComputationGraph::stypes; // static of computation graph
    Trie<OoC::BBInfo *> ComputationGraph::head;     // static of computation graph
    PatternCache ComputationGraph::pattern_cache;   // static of computation graph

    int ComputationGraph::mark_basic_block(int stid)
    {
        if (autobatch_flag != 7)
            return -1;
        if (profiling_flag)
            timer.start("bbmark");
        BBInfo *data;
        int sid = snodes.size();
        if (stid < 0)
        {
            int n_unbatchable = 0;
            Trie<BBInfo*> * curr = &head;
            for (int nid = n_marked_node; nid < nodes.size(); nid++)
            {
                auto node = nodes[nid];
                int sig = node->autobatch_sig(*this, sigmap);
                if (sig == 0)
                {
                    n_unbatchable++;
                    assert(!node->args.size());
                    unbatchable_ops.push_back(nid);
                    nid2sid.push_back(-1);
                    continue;
                }
                if (curr->next.count(sig) == 0)
                    curr->next[sig] = new Trie<BBInfo *>();
                curr = curr->next[sig];
                nid2sid.push_back(sid);
            }
            if (n_unbatchable == (nodes.size() - n_marked_node))
            {
                n_marked_node = nodes.size();
                if (profiling_flag)
                    timer.stop("bbmark");
                return -1;
            }
            if (n_unbatchable)
            {
                for (int nid = n_marked_node; nid < nodes.size(); nid++)
                {
                    fprintf(stdout, "(%s, %d), ", OoC::type2name[sigmap.sig2type(nodes[nid]->autobatch_sig(*this, sigmap))].c_str(), nid);
                }
                fprintf(stdout, "\n");
            }
            assert(n_unbatchable == 0);

            if (!curr->isLeaf)
            {
                curr->data = new BBInfo();
                curr->isLeaf = true;
                int stid = stypes.size();
                curr->data->stid = stid;
                stypes.push_back({});
                vector<int> node2type;
                vector<vector<int>> node2args;
                for (int nid = n_marked_node; nid < nodes.size(); nid++)
                {
                    auto node = nodes[nid];
                    int sig = node->autobatch_sig(*this, sigmap);
                    node2type.push_back(sig);
                    node2args.push_back({});
                    int aid = 0;
                    for (auto arg : node->args)
                    {
                        node2args.back().push_back(arg - n_marked_node);
                        int arg_sid = nid2sid[arg];
                        if (arg_sid >= 0 && arg < n_marked_node)
                        {
                            curr->data->pred_pos.push_back({nid - n_marked_node, aid});
                        }
                        aid++;
                    }
                }
                fprintf(stdout, "add pattern %d, %ld\n", stid, node2args.size());
                stypes.back().pattern = pattern_cache.add_pattern(stid, node2args, node2type);
                assert(stypes.back().pattern);
                stypes.back().bbinfo = curr->data;
                if (profiling_flag > 1)
                {
                    fprintf(stdout, "pattern: ");
                    for (int nid = n_marked_node; nid < nodes.size(); nid++)
                    {
                        fprintf(stdout, "%s, ", type2name[sigmap.sig2type(node2type[nid - n_marked_node])].c_str());
                    }
                    fprintf(stdout, "\n");
                }
                fprintf(stdout, "add pattern finished!\n");
            }
            data = curr->data;
        }
        else {
            data = stypes[stid].bbinfo;
            assert(data && stid == data->stid);
        }

        snodes.push_back({});
        auto &snode = snodes.back();
        snode.min_nid = n_marked_node;

        // results.emplace_back(thread_pool.enqueue([&]{
        for (auto &kv : data->pred_pos)
        {
            int nid = nodes[n_marked_node + kv.first]->args[kv.second];
            int arg_sid = nid2sid[nid];
            // snodes[arg_sid].inputCnt++;
            snode.succs.push_back(arg_sid);
        }
        // }));

        stid = data->stid;
        snode.type = stid;
        auto &stype = stypes[stid];
        if (stid >= n_stored_stypes)
        {
            n_new_ops += stype.pattern->nop;
            if (n_new_ops > 1)
                schedule_mode = TRAIN;
        }
        // if (!preds.size()) stype.frontiers.push_back(sid);
        stype.cnt++;

        n_marked_node = nodes.size();
        if (profiling_flag)
            timer.stop("bbmark");
        return sid;
    }

    void ComputationGraph::mark_sum()
    {
        if (autobatch_flag != 7)
            return;
        assert(n_marked_node == nodes.size() - 1);
        auto node = nodes[n_marked_node];
        int sig = node->autobatch_sig(*this, sigmap);
        assert(sigmap.sig2type(sig) == op_type::sum);
        int sid = snodes.size();
        snodes.push_back({});
        nid2sid.push_back(sid);
        auto &snode = snodes.back();
        snode.min_nid = n_marked_node;
        Trie<BBInfo *> *curr = &head;
        if (curr->next.count(sig) == 0)
        {
            curr->next[sig] = new Trie<BBInfo *>();
        }
        curr = curr->next[sig];
        if (!curr->isLeaf)
        {
            curr->data = new BBInfo();
            curr->isLeaf = true;
            int stid = stypes.size();
            curr->data->stid = stid;
            stypes.push_back({});
            stypes.back().pattern = pattern_cache.add_pattern(stid, {{}}, {sig});
            fprintf(stdout, "add pattern: sum %d, \n", stypes.back().pattern->nop);
            stypes.back().pattern->show();
        }
        assert(curr->data);
        int stid = curr->data->stid;
        snode.type = stid;
        stypes[stid].cnt++;
        for (auto arg : node->args)
        {
            int arg_sid = nid2sid[arg];
            assert(arg_sid >= 0);
            // snodes[arg_sid].inputCnt++;
            snode.succs.push_back(arg_sid);
        }
        n_marked_node = nodes.size();
    }

    void ComputationGraph::export_snode_graph(string filename)
    {
        ofstream file;
        file.open(filename);
        file << snodes.size() << endl;
        int sid = 0;
        for (auto &snode : snodes)
        {
            file << snode.type << " " << snode.succs.size() << " ";
            for (auto succ : snode.succs)
                file << succ << " ";
            file << endl;
        }
        file.close();
    }

} // namespace dynet