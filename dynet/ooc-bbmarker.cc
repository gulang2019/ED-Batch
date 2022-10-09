#include "dynet.h"
#include "dynet/ooc-computation_graph.h"
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

    int ComputationGraph::mark_basic_block(bool sync)
    {
        if (autobatch_flag != 7 || (int)nodes.size() == n_marked_node){
            return -1;
        }
        if (profiling_flag)
            timer.start("bbmark");

        if (sync)
            SuperNode::synchronize("sync_from_markbb");
        int sid = snodes.size();
        Trie<BBInfo *> *curr = &head;
        for (int nid = n_marked_node; nid < nodes.size(); nid++)
        {
            auto node = nodes[nid];
            int sig = node->autobatch_sig(*this, sigmap);
            if (curr->next.count(sig) == 0)
                curr->next[sig] = new Trie<BBInfo *>();
            curr = curr->next[sig];
            nid2sid.push_back(sid);
        }

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
            int fake_type = 0;
            for (int nid = n_marked_node; nid < nodes.size(); nid++)
            {
                auto node = nodes[nid];
                int sig = node->autobatch_sig(*this, sigmap);
                node2type.push_back(sig ? sig : --fake_type); // for unbatchable ops, give it a unique tid;
                node2args.push_back({});
                bool is_input_node = false;
                for (auto arg : node->args)
                {
                    node2args.back().push_back(arg - n_marked_node);
                    is_input_node = is_input_node || (arg < n_marked_node);
                }
                if (is_input_node)
                    curr->data->input_nodes.push_back(nid - n_marked_node);
            }
            fprintf(stdout, "add pattern %d, %ld\n", stid, node2args.size());
            if (profiling_flag > 1){
                for (int nid = n_marked_node; nid < nodes.size(); nid++){
                    auto node = nodes[nid];
                    fprintf(stdout, "[node %d %s]: %s\n", nid,
                     type2name[sigmap.sig2type(node->autobatch_sig(*this, sigmap))].c_str(),
                     node->as_dummy_string().c_str());
                }   
            }
            if (node2args.size() > 100) 
                throw 0;
            stypes.back().pattern = pattern_cache.add_pattern(stid, node2args, node2type);
            assert(stypes.back().pattern);
            stypes.back().bbinfo = curr->data;
            fprintf(stdout, "add pattern finished!\n");
        }
        BBInfo* data = curr->data;

        snodes.push_back(new supernodeInfo);
        auto snode = snodes.back();
        snode->min_nid = n_marked_node;
        snode->type = data->stid;
        stypes[data->stid].cnt++;

        if (profiling_flag > 1)
            log.push_back({SEQ_CONSTRUCT, "seq_construct", n_marked_node, nodes.size(), sid, data->stid});
        n_marked_node = nodes.size();
        if (profiling_flag)
            timer.stop("bbmark");
        return data->stid;
    }

    void ComputationGraph::construct_snode_graph()
    {
        if (autobatch_flag != 7)
            return;
        OoC::SuperNode::synchronize("sync_from_snode_construct");
        if (profiling_flag > 1) 
            for (auto node: nodes)  assert(node != nullptr);
        global_timer.start("synchronize snode");
        vector<future<int> > results;
        int n_thread = std::min(thread::hardware_concurrency(), ((unsigned)snodes.size() + 99) / 100);
        int n_work = (snodes.size() + n_thread - 1) / n_thread;
        for (int thread_id = 0; thread_id < n_thread; thread_id++)
        {
            results.emplace_back(async([thread_id, this, n_work]
                                       {
            int n_ops = 0;
            for (int sid = thread_id * n_work; sid < std::min((thread_id+1)*n_work, (int)this->snodes.size()); sid++){
                auto snode = this->snodes[sid];
                int stid = snode->type;
                OoC::BBInfo * data = stypes[stid].bbinfo;
                for (auto i_node_offset : data->input_nodes){
                    auto node = this->nodes[snode->min_nid + i_node_offset];
                    for (auto arg: node->args){
                        if ((int)arg < snode->min_nid) {
                            snode->succs.push_back(this->nid2sid[arg]);
                        }
                    }
                }
                n_ops += stid >= this->n_stored_stypes? data->nop:0;
            }
            return n_ops; }));
        }
        int n_new_op = 0;
        for (int thread_id = 0; thread_id < n_thread; thread_id++)
        {
            n_new_op += results[thread_id].get();
        }
        if (n_new_op > 1)
            schedule_mode = TRAIN;

        int frontier_type_cnt = 0;
        for (int sid = snodes.size() - 1; sid >= 0; sid--)
        {
            auto &snode = snodes[sid];
            if (snode->inputCnt == 0)
            {
                stypes[snode->type].frontiers.push_back(sid);
                stypes[snode->type].cnt++;
                frontier_type_cnt += 1;
            }
            for (auto succ : snode->succs)
                snodes[succ]->inputCnt++;
        }
        global_timer.stop("synchronize snode");
        fprintf(stdout, "[ComputationGraph::construct_snode_graph]: frontier_type_cnt: %d\n", frontier_type_cnt);
        show_log();
    }

    void ComputationGraph::export_snode_graph(string filename)
    {
        ofstream file;
        file.open(filename);
        file << snodes.size() << endl;
        for (auto &snode : snodes)
        {
            file << snode->type << " " << snode->succs.size() << " ";
            for (auto succ : snode->succs)
                file << succ << " ";
            file << endl;
        }
        file.close();
    }

    void ComputationGraph::show_log(){
        if (profiling_flag < 2) return;
        fprintf(stdout, "------------------show log---------------------\n");
        for (auto& item: log){
            if (item.type == SYNCHRONIZE){
                fprintf(stdout, "[%s] nodes %d, result.size() %d\n", 
                    item.info.c_str(), item.begin, item.end);
            }
            else if (item.type == PARA_CONSTRUCT){
                fprintf(stdout, "[%s] begin %d, end %d, sid %d, stid %d\n",  
                    item.info.c_str(), item.begin,item.end, item.sid, item.stid);
            }
            else if (item.type == SEQ_CONSTRUCT){
                fprintf(stdout, "[%s] begin %d, end %d, sid %d, stid %d\n",  
                    item.info.c_str(), item.begin,item.end, item.sid, item.stid);
            }
        }
        fprintf(stdout, "sync happens %d time, %d of %ld is wrapped\n", 
            SuperNode::n_sync, SuperNode::n_node, nodes.size());
        fprintf(stdout, "----------------show log end--------------------\n");
    }

    void ComputationGraph::show_nodes(){
        if (profiling_flag < 2) return;
        int nid = 0;
        for (auto node: nodes){
            fprintf(stdout, "[node %d]: %s\n", nid, node->as_dummy_string().c_str());
            nid++;
        }
    }

} // namespace dynet

// A1. delay the commit of lookup ; but sync every mark_basic_block;
// A2. make placeholder for the commited but unbatched ops;