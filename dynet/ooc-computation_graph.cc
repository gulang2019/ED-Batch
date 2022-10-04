#include "ooc-computation_graph.h"

using namespace std;
using namespace OoC;
using namespace dynet;

namespace OoC
{
    int SuperNode::n_node = 0;
    int SuperNode::n_sync = 0;
    ThreadPool SuperNode::pool(1); 
    std::vector<std::future<int> > SuperNode::results;

    vector<Expression> SuperNode::operator()(
        const vector<Expression> &input,
        const vector<int> &params, 
        bool mark_basic_block)
    {
        log_t* log = &params_dict[params];
        assert(log);
        if (log->first_time)
        {
            if (autobatch_flag == 7) log->first_time = false;
            log->begin = _cg->nodes.size();
            vector<Expression> out;
            _func(input, params, out);
            log->end = _cg->nodes.size();
            int i = 0;
            for (auto &expr : input)
            {
                log->nid2aid[expr.i] = i++;
            }
            for (auto &expr : out)
            {
                log->output_indices.push_back(expr.i - log->begin);
            }
            if (mark_basic_block && dynet::autobatch_flag == 7) {
                int sid = _cg->mark_basic_block();
                log->_stid = _cg->snodes[sid]->type;
            }
            n_node += log->end - log->begin;
            return move(out);
        }
        int min_nid = _cg->nodes.size();
        if (mark_basic_block) assert(_cg->n_marked_node == min_nid);
        vector<Expression> out;
        for (auto idx : log->output_indices){
            out.push_back(Expression(_cg, min_nid + idx));
        }
        n_node += log->end - log->begin;
        int sid = _cg->snodes.size();
        vector<int> _input(input.size());
        for (size_t i = 0; i < input.size(); i++) _input[i] = input[i].i; 
        _cg->nodes.resize(_cg->nodes.size() + log->end - log->begin, nullptr);
        _cg->snodes.push_back(nullptr);
        _cg->nid2sid.resize(_cg->nodes.size(), -1);
        _cg->n_marked_node = _cg->nodes.size();

        results.emplace_back(
            async([=]{
                // if (profiling_flag > 1){
                    fprintf(stdout, "[buildGraph]: [%d,%d), sid %d, mark_basic_block%d\n", 
                        min_nid, min_nid + log->end - log->begin, sid, mark_basic_block);
                    fprintf(stdout, "\t");
                    for (auto x: _input)
                        fprintf(stdout, "%d,", x);
                    fprintf(stdout, "\n");
                // }
                for (int nid = log->begin; nid < log->end; nid++)
                {
                    Node *node = new FakeNode(_cg->nodes[nid], min_nid + nid - log->begin);
                    _cg->nodes[min_nid + nid - log->begin] = node;
                    fprintf(stdout, "create node %d %p\n", min_nid + nid - log->begin, node);
                    if (!node) return -1;
                    for (auto &arg : node->args)
                    {
                        if (log->nid2aid.count(arg)){
                            assert(log->nid2aid[arg] < _input.size());
                            arg = _input[log->nid2aid[arg]];
                        }
                        else if (arg >= log->begin) arg += min_nid - log->begin;
                    }
                }
                if (mark_basic_block && dynet::autobatch_flag == 7){
                    auto snode = new supernodeInfo;
                    snode->min_nid = min_nid;
                    snode->type = log->_stid;
                    for (auto idx: log->output_indices){
                        _cg->nid2sid[min_nid + idx] = sid;
                    }
                    _cg->snodes[sid] = snode;
                }
                return 0;
            })
        );

        
        return move(out);
    }

    void SuperNode::synchronize(){
        n_sync ++;
        for(auto& result: results) 
            assert(result.get() == 0);
        results.clear();
    }
} // namespace OoC