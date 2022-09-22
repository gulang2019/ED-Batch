#include "ooc-computation_graph.h"

using namespace std;
using namespace OoC;
using namespace dynet;

namespace OoC
{
    vector<Expression> SuperNode::operator()(
        const vector<Expression> &input,
        const vector<int> &params, 
        bool mark_basic_block)
    {
        if (first_time)
        {
            first_time = false;
            begin = _cg->nodes.size();
            vector<Expression> out;
            _func(input, params, out);
            end = _cg->nodes.size();
            int i = 0;
            for (auto &expr : input)
            {
                nid2aid[expr.i] = i++;
            }
            for (auto &expr : out)
            {
                output_indices.push_back(expr.i - begin);
            }
            if (mark_basic_block && dynet::autobatch_flag == 7) {
                int sid = _cg->mark_basic_block();
                _stid = _cg->snodes[sid].type;
            }
            return move(out);
        }
        int this_begin = _cg->nodes.size();
        for (int nid = begin; nid < end; nid++)
        {
            Node *node = new FakeNode(_cg->nodes[nid]);
            for (auto &arg : node->args)
            {
                if (nid2aid.count(arg)){
                    arg = input[nid2aid[arg]].i;
                }
                else if (arg >= begin) arg += this_begin - begin;
            }
            _cg->nodes.push_back(node);
        }
        vector<Expression> out;
        for (auto idx : output_indices){
            out.push_back(Expression(_cg, this_begin + idx));
        }
        if (mark_basic_block && dynet::autobatch_flag == 7){
            int sid = _cg->mark_basic_block(_stid);
            _cg->nid2sid.resize(_cg->nodes.size(), -1);   
            for (auto idx: output_indices){
                _cg->nid2sid[this_begin + idx] = sid;
            }
        }
        return move(out);
    }
} // namespace OoC