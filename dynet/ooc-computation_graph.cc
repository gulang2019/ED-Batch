#include "ooc-computation_graph.h"

using namespace std;
using namespace OoC;
using namespace dynet;

namespace OoC
{
    int SuperNode::n_node = 0;

    vector<Expression> SuperNode::operator()(
        const vector<Expression> &input,
        const vector<int> &params, 
        bool mark_basic_block)
    {
        log_t& log = params_dict[params];
        if (log.first_time)
        {
            if (autobatch_flag == 7) log.first_time = false;
            log.begin = _cg->nodes.size();
            vector<Expression> out;
            _func(input, params, out);
            log.end = _cg->nodes.size();
            int i = 0;
            for (auto &expr : input)
            {
                log.nid2aid[expr.i] = i++;
            }
            for (auto &expr : out)
            {
                log.output_indices.push_back(expr.i - log.begin);
            }
            if (mark_basic_block && dynet::autobatch_flag == 7) {
                int sid = _cg->mark_basic_block();
                log._stid = _cg->snodes[sid].type;
            }
            n_node += log.end - log.begin;
            return move(out);
        }
        int this_begin = _cg->nodes.size();
        for (int nid = log.begin; nid < log.end; nid++)
        {
            Node *node = new FakeNode(_cg->nodes[nid]);
            for (auto &arg : node->args)
            {
                if (log.nid2aid.count(arg)){
                    arg = input[log.nid2aid[arg]].i;
                }
                else if (arg >= log.begin) arg += this_begin - log.begin;
            }
            _cg->nodes.push_back(node);
        }
        vector<Expression> out;
        for (auto idx : log.output_indices){
            out.push_back(Expression(_cg, this_begin + idx));
        }
        if (mark_basic_block && dynet::autobatch_flag == 7){
            int sid = _cg->mark_basic_block(log._stid);
            _cg->nid2sid.resize(_cg->nodes.size(), -1);   
            for (auto idx: log.output_indices){
                _cg->nid2sid[this_begin + idx] = sid;
            }
        }
        n_node += log.end - log.begin;
        return move(out);
    }
} // namespace OoC