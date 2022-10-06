#include "ooc-computation_graph.h"


using namespace std;
using namespace OoC;
using namespace dynet;

namespace OoC
{
    int SuperNode::n_node = 0;
    int SuperNode::n_sync = 0;
    BS::thread_pool SuperNode::pool(16); 
    std::vector<std::future<void*> > SuperNode::results;
    ComputationGraph* SuperNode::_cg = nullptr;

    struct async_ret{
        vector<Node*> nodes;
        supernodeInfo *snode = nullptr;
    };

    vector<Expression> SuperNode::operator()(
        const vector<Expression> &input,
        const vector<int> &const_params, 
        const vector<int> &runtime_params,
        bool mark_basic_block)
    {
        global_timer.start("SuperNode::operator()");
        log_t* log = &params_dict[const_params];
        assert(log);
        if (log->first_time)
        {
            if (autobatch_flag == 7) log->first_time = false;
            synchronize("sync_first_time");
            if(autobatch_flag == 7 && _cg->n_marked_node != _cg->nodes.size()){
                _cg->show_log();
                fprintf(stdout, "_cg->n_marked_node %d, _cg->nodes.size() %ld\n", _cg->n_marked_node,
                    _cg->nodes.size());
                assert(false);
            }
            log->begin = _cg->nodes.size();
            vector<Expression> out;
            _func(input, const_params, runtime_params, out);
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
                log->_stid = _cg->mark_basic_block(false);
            }
            n_node += log->end - log->begin;
            return move(out);
        }
        int min_nid = _cg->n_marked_node;
        _cg->n_marked_node += log->end - log->begin;
        vector<Expression> out;
        for (auto idx : log->output_indices){
            out.push_back(Expression(_cg, min_nid + idx));
        }
        n_node += log->end - log->begin;
        vector<int>* _input = new vector<int>;
        for (auto &expr: input) _input->push_back(expr.i); 
        global_timer.start("emplace_task");
        results.emplace_back(
            pool.submit([_input,min_nid,log,mark_basic_block]{
                async_ret * ret = new async_ret;
                assert(ret);
                
                for (int nid = log->begin; nid < log->end; nid++)
                {
                    Node *node = _cg->nodes[nid]->clone();
                    ret->nodes.push_back(node);
                    for (auto &arg : node->args)
                    {
                        if (log->nid2aid.count(arg)){
                            if(log->nid2aid[arg] >= _input->size()){
                                fprintf(stderr, "[ERROR]: min_nid %d, log->nid2aid[%d] = %d; _input.size() %ld, begin %d, end %d;\n", 
                                    min_nid, arg, log->nid2aid[arg], _input->size(), log->begin, log->end);
                                for (int nid = log->begin; nid < log->end; nid++){
                                    fprintf(stderr, "\tnid %d %s\n", nid, _cg->nodes[nid]->as_dummy_string().c_str());
                                }
                                return static_cast<void*>(nullptr);
                            }
                            arg = _input->at(log->nid2aid[arg]);
                        }
                        else if (arg >= log->begin) arg += min_nid - log->begin;
                    }
                }
                // _extra_func(ret->nodes, runtime_params);
                if (mark_basic_block && dynet::autobatch_flag == 7){
                    auto snode = new supernodeInfo;
                    snode->min_nid = min_nid;
                    snode->type = log->_stid;
                    ret->snode = snode;
                }
                delete _input;
                return static_cast<void*>(ret);
            })
        );
        global_timer.stop("emplace_task");
        global_timer.stop("SuperNode::operator()");
        
        return move(out);
    }

    void SuperNode::synchronize(string info){
        global_timer.start("synchronize");
        n_sync ++;
        for(auto& result: results) {
            async_ret* ptr = static_cast<async_ret*>(result.get());
            assert(ptr);
            // for (int nid = 0; nid < ptr->nodes.size(); nid++){
            //     int target_nid = reinterpret_cast<FakeNode*>(ptr->nodes.at(nid))->_nid;
            // }
            if (profiling_flag > 1)
                _cg->log.push_back({ComputationGraph::PARA_CONSTRUCT, 
                    "para_construct",
                    _cg->nodes.size(), 
                    _cg->nodes.size() + ptr->nodes.size(), 
                    _cg->snodes.size(), 
                    ptr->snode->type});
            _cg->nodes.insert(_cg->nodes.end(), ptr->nodes.begin(), ptr->nodes.end());
            _cg->nid2sid.resize(_cg->nodes.size(), (int)_cg->snodes.size());
            _cg->snodes.push_back(ptr->snode);
        }
        if (results.size()) _cg->n_ready_node = _cg->nodes.size();
        if (profiling_flag > 1)
            _cg->log.push_back({ComputationGraph::SYNCHRONIZE, info, 
                _cg->nodes.size(), results.size()});
        results.clear();
        global_timer.stop("synchronize");
    }
} // namespace OoC
/*
    snode parallel construction;
    snode construction
*/
// n_marked_node: 2