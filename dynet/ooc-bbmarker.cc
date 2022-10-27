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

        global_timer.start("mark_basic_block");

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

        global_timer.stop("mark_basic_block");
        n_marked_node = nodes.size();
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
        global_timer.start("snode launch");
        for (int sid = 0; sid < snodes.size(); sid++){
            auto snode = this->snodes[sid];
            int stid = snode->type;
            OoC::BBInfo * data = stypes[stid].bbinfo;
            for (auto i_node_offset : data->input_nodes){
                auto node = this->nodes[snode->min_nid + i_node_offset];
                for (auto arg: node->args){
                    if ((int)arg < snode->min_nid) {
                        int succ_id = this->nid2sid[arg];
                        if (find(snode->succs.begin(), snode->succs.end(), succ_id) == snode->succs.end()) {
                            snode->succs.push_back(succ_id);
                        }
                    }
                }
            }
        }
        global_timer.stop("snode launch");

        global_timer.start("collect frontiers");
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
        global_timer.stop("collect frontiers");
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

    void ComputationGraph::export_graph(string filename){
        ofstream file;
        file.open(filename);
        file << nodes.size() << endl;
        for (auto &node: nodes){
            file << node->autobatch_sig(*this,sigmap) << " " << node->args.size();
            for (auto arg: node->args) 
                file << " " << arg;
            file << endl;
        }
        file.close();
    }

    void ComputationGraph::export_dot_graph(string filename){
        ofstream file;
        file.open(filename);
        file << "digraph {" << endl;
        file << "subgraph G0 {\n";
        {
            auto get_name = [this](int nid) {
                return type2name[sigmap.sig2type(nodes[nid]->autobatch_sig
                    (*this, sigmap))] + "_" + to_string(nid);
            };
            unordered_map<int, string> colormap;
            auto get_color = [&colormap](int type) {
                if (colormap.count(type) == 0){
                    char tmp[20];
                    sprintf(tmp, "#%6x", rand()%0xffffff);
                    colormap[type] = string(tmp);
                } 
                return colormap[type];
            };
            for (int nid = 0; nid < nodes.size(); nid++){
                string name = get_name(nid);
                string color = get_color(nodes[nid]->autobatch_sig(*this, sigmap));
                file << name << " [color=\"" << color << "\"];" << endl; 
                for (auto arg: nodes[nid]->args){
                    file << get_name(arg) << " -> " << name << ";" << endl;
                }
            }
        }
        file << "}\n";
        file << "subgraph G1{\n";
        {
            auto get_name = [this](int nid) {
                return "S" + to_string(nid);
            };
            unordered_map<int, string> colormap;
            auto get_color = [&colormap](int type) {
                if (colormap.count(type) == 0){
                    char tmp[20];
                    sprintf(tmp, "#%6x", rand()%0xffffff);
                    colormap[type] = string(tmp);
                } 
                return colormap[type];
            };
            for (int nid = 0; nid < snodes.size(); nid++){
                string name = get_name(nid);
                string color = get_color(snodes[nid]->type);
                file << name << " [color=\"" << color << "\"];" << endl; 
                for (auto arg: snodes[nid]->succs){
                    file << get_name(arg) << " -> " << name << ";" << endl;
                }
            }
        }
        file << "}\n";
        file << "}\n";

        file.close();
    }

    void ComputationGraph::print_unbatchables(){
        fprintf(stdout, "-------------unbatchables--------------\n");
        for (auto node: nodes){
            auto type = node->autobatch_sig(*this, sigmap);
            if (!type) {
                fprintf(stdout, "%s\n", node->as_dummy_string().c_str());
            }
        }
        fprintf(stdout, "---------------------------------------\n");
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
        // if (profiling_flag < 2) return;
        int nid = 0;
        for (auto node: nodes){
            vector<string>args;
            for (auto arg: node->args) args.push_back(to_string(arg));
            cout << "[node " << nid << ","<<node->dim << "]: " << node->as_string(args) << endl;
            nid++;
        }
    }

    // OoC's extension to dynet's computation graph
    void ComputationGraph::gen_cdfg(bool draw, string prefix){
        vector<int> node_types(nodes.size());
        int nid = 0;
        for (auto node: nodes) {
            node_types[nid++] = node->autobatch_sig(*this, sigmap);

        }
        types.resize(sigmap.size());
        int tid = 0;
        for (auto & t:types){
            t.name = OoC::type2name[sigmap.sig2type(tid)]+to_string(tid);
            ++tid;
        }
        nid = 0;
        int n_edge = 0;
        int n_total = 0;
        for (auto node: nodes) {
            auto& type = types[node_types[nid]];
            type.cnt++;
            for (auto arg: node->args){
                int t = node_types[arg];
                if (find(type.args.begin(), type.args.end(), t) == type.args.end()){ 
                    n_edge++;
                    type.args.push_back(t);
                }
            }
            n_total += node->args.size();
            nid++;
        }
        cout << "CG nodes:" << nodes.size() 
            << ",CG edges:" << n_total 
            << ",CDFG nodes:" << types.size() 
            << ",CDFG edges:" << n_edge << endl;
        
        tid = 0;
        for (auto& type: types){
            cout << type.name << ": " << type.cnt << endl;
        }

        // use floyd method to find all nodes with self-circle;
        // (i->j,k) = U_(l \in args(j))(i->l, k-1)
        bool* reachable = new bool[types.size() * types.size()];
        memset(reachable, 0, sizeof(bool) * types.size() * types.size());
        for (int j = 0; j < types.size(); ++j) 
            for (auto arg: types[j].args) reachable[arg*types.size()+j] = true;
        for (int k = 0; k < (int)types.size(); k++)
            for (int i = 0; i < (int)types.size(); i++){
                for (int j = 0; j < (int)types.size(); j++){
                    for (auto arg: types[j].args){
                        reachable[i*types.size()+j] |= reachable[i*types.size()+arg];
                    }
                }
            }
        for (int i = 0; i < types.size(); i++) types[i].self_reachable = reachable[i*types.size()+i];
        delete reachable;

        if (draw) {
            ofstream file;
            file.open("./pics/"+prefix+".gv");
            file << "digraph G{" << endl;
            for (auto type: types) {
                for (auto arg: type.args) {
                    file << types[arg].name << "->" << type.name << endl;
                }
                if (type.self_reachable) 
                    file << type.name << "[color=\"red\"]" << endl;
            }
            file << "}" << endl;
            file.close();
        }
    }
    std::vector<ComputationGraph::Type> ComputationGraph::types;
} // namespace dynet

// A1. delay the commit of lookup ; but sync every mark_basic_block;
// A2. make placeholder for the commited but unbatched ops;