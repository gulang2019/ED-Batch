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
            Trie<BBInfo *> *curr = &head;
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
                curr->data->nop = nodes.size() - n_marked_node;
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
        else
        {
            data = stypes[stid].bbinfo;
            assert(data && stid == data->stid);
        }

        snodes.push_back({});
        auto &snode = snodes.back();
        snode.min_nid = n_marked_node;
        snode.type = data->stid;
        stypes[data->stid].cnt++;

        n_marked_node = nodes.size();
        if (profiling_flag)
            timer.stop("bbmark");
        return sid;
    }

    void ComputationGraph::mark_sum_and_finish()
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
            curr->next[sig] = new Trie<BBInfo *>();

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
            snode.succs.push_back(arg_sid);
        }
        n_marked_node = nodes.size();

        global_timer.start("synchronize snode");
        vector<future<int>> results;
        int n_thread = std::min(thread::hardware_concurrency(), ((unsigned)snodes.size() + 99) / 100);
        int n_work = (snodes.size() + n_thread - 2) / n_thread;
        for (int thread_id = 0; thread_id < n_thread; thread_id++)
        {
            results.emplace_back(async([thread_id, this, n_work]
                                       {
            int n_ops = 0;
            for (int sid = thread_id * n_work; sid < std::min((thread_id+1)*n_work, (int)this->snodes.size()-1); sid++){
                auto & snode = this->snodes[sid];
                int stid = snode.type;
                OoC::BBInfo * data = stypes[stid].bbinfo;
                for (auto &kv : data->pred_pos){
                    int nid = this->nodes[snode.min_nid + kv.first]->args[kv.second];
                    int arg_sid = this->nid2sid[nid];
                    snode.succs.push_back(arg_sid);
                }
                n_ops += stid >= this->n_stored_stypes? data->nop:0;
            }
            return n_ops; }));
        }
        int n_new_op = 0;
        for (int thread_id = 0; thread_id < n_thread; thread_id++){
            n_new_op += results[thread_id].get();
        }
        if (n_new_op > 1)
            schedule_mode = TRAIN;

        int frontier_type_cnt = 0;
        for (int sid = snodes.size()-1; sid >= 0; sid--){  
            auto & snode = snodes[sid]; 
            if (snode.inputCnt == 0){
                stypes[snode.type].frontiers.push_back(sid);
                frontier_type_cnt += 1;
            }
            for (auto succ: snode.succs) snodes[succ].inputCnt++;   
        }
        global_timer.stop("synchronize snode");
        fprintf(stdout, "[OoC::forward]: frontier_type_cnt: %d\n", frontier_type_cnt);
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