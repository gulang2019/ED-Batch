#include "dynet/ooc-computation_graph.h"
#include "dynet/devices.h"
#include "dynet/timing.h"
#include "dynet/expr.h"
#include "dynet/nodes.h"
#include "dynet/param-nodes.h"

#ifdef HAVE_CUDA
#include "dynet/gpu-ops.h"
#endif

using namespace std;
using namespace OoC;
using namespace dynet;

namespace OoC
{
    int SuperNode::n_node = 0;
    int SuperNode::n_sync = 0;
    std::vector<std::future<void *>> SuperNode::results;
    ComputationGraph *SuperNode::_cg = nullptr;

    struct async_ret
    {
        vector<Node *> nodes;
        supernodeInfo *snode = nullptr;
    };

    vector<Expression> SuperNode::operator()(
        const vector<Expression> &input,
        const vector<int> &const_params,
        const vector<int> &runtime_params,
        bool mark_basic_block)
    {
        // global_timer.start("SuperNode::operator()");
        // global_timer.start("dict lookup");
        if (profiling_flag > 1)
        {
            fprintf(stdout, "[SuperNode::operator()]: n_marked_node %d, nodes.size() %ld\n",
                    _cg->n_marked_node, _cg->nodes.size());
        }
        log_t *log = &params_dict[const_params];
        // global_timer.stop("dict lookup");
        assert(log);
        if (log->first_time)
        {
            // global_timer.start("first_time");
            if (autobatch_flag == 7)
                log->first_time = false;
            // synchronize("sync_first_time");
            // if(autobatch_flag == 7 && _cg->n_marked_node != _cg->nodes.size()){
            //     _cg->show_nodes();
            //     _cg->show_log();
            //     fprintf(stdout, "_cg->n_marked_node %d, _cg->nodes.size() %ld\n", _cg->n_marked_node,
            //         _cg->nodes.size());
            //     assert(false);
            // }
            log->begin = _cg->n_marked_node;
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
            if (mark_basic_block && dynet::autobatch_flag == 7)
            {
                log->_stid = _cg->mark_basic_block(false);
            }
            n_node += log->end - log->begin;
            // global_timer.stop("first_time");
            // global_timer.stop("SuperNode::operator()");
            return move(out);
        }
        // global_timer.start("stype"+to_string(log->_stid));
        // global_timer.start("prepare_output");
        int min_nid = _cg->n_marked_node;
        _cg->n_marked_node += log->end - log->begin;
        vector<Expression> out;
        for (auto idx : log->output_indices)
        {
            out.push_back(Expression(_cg, min_nid + idx));
        }
        // global_timer.stop("prepare_output");
        // global_timer.start("construct node2args");
        n_node += log->end - log->begin;
        vector<int> *_input = new vector<int>;
        for (auto &expr : input)
            _input->push_back(expr.i);

        // results.emplace_back(
        //     pool.submit([_input,min_nid,log,mark_basic_block]{
        // async_ret * ret = new async_ret;
        // assert(ret);

        for (int nid = log->begin; nid < log->end; nid++)
        {
            bool is_lookup = false;
            for (auto &lookup_arg : lookup_args)
            {
                if ((nid - log->begin) == lookup_arg.nid)
                {
                    is_lookup = true;
                    dynet::lookup(*_cg, *(lookup_arg.p), runtime_params[lookup_arg.param_id]);
                    break;
                }
            }
            if (is_lookup)
                continue;
            Node *node = _cg->nodes[nid]->clone();
            // ret->nodes.push_back(node);
            _cg->nodes.push_back(node);
            for (auto &arg : node->args)
            {
                if (log->nid2aid.count(arg))
                {
                    if (log->nid2aid[arg] >= _input->size())
                    {
                        fprintf(stderr, "[ERROR]: min_nid %d, log->nid2aid[%d] = %d; _input.size() %ld, begin %d, end %d;\n",
                                min_nid, arg, log->nid2aid[arg], _input->size(), log->begin, log->end);
                        for (int nid = log->begin; nid < log->end; nid++)
                        {
                            fprintf(stderr, "\tnid %d %s\n", nid, _cg->nodes[nid]->as_dummy_string().c_str());
                        }
                        assert(false);
                        // return static_cast<void*>(nullptr);
                    }
                    arg = _input->at(log->nid2aid[arg]);
                }
                else if (arg >= log->begin)
                    arg += min_nid - log->begin;
            }
        }

        // global_timer.stop("construct node2args");
        // global_timer.start("construct snode");

        if (mark_basic_block && dynet::autobatch_flag == 7)
        {
            _cg->nid2sid.resize(_cg->nodes.size(), _cg->snodes.size());
            auto snode = new supernodeInfo;
            _cg->snodes.push_back(snode);
            snode->min_nid = min_nid;
            snode->type = log->_stid;
            // ret->snode = snode;
        }
        delete _input;
        // global_timer.stop("construct snode");
        //         return static_cast<void*>(ret);
        //     })
        // );
        // global_timer.stop("stype"+to_string(log->_stid));
        // global_timer.stop("SuperNode::operator()");
        if (!(_cg->n_marked_node == _cg->nodes.size()))
        {
            _cg->show_nodes();
            fprintf(stdout, "min_nid, log->begin, log->end, _cg->n_marked_node, _cg->nodes.size(): %d, %d, %d, %d, %d\n",
                    min_nid, log->begin, log->end, _cg->n_marked_node, _cg->nodes.size());
            for (auto &lookup_arg : lookup_args)
            {
                fprintf(stdout, "[lookup_arg]: %d, %d\n", lookup_arg.nid, lookup_arg.param_id);
            }
        }
        assert(_cg->n_marked_node == _cg->nodes.size());
        return move(out);
    }

    void SuperNode::synchronize(string info)
    {
        global_timer.start("synchronize");
        n_sync++;
        for (auto &result : results)
        {
            async_ret *ptr = static_cast<async_ret *>(result.get());
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
        if (results.size())
            _cg->n_ready_node = _cg->nodes.size();
        if (profiling_flag > 1)
            _cg->log.push_back({ComputationGraph::SYNCHRONIZE, info,
                                _cg->nodes.size(), results.size()});
        results.clear();
        global_timer.stop("synchronize");
    }


    Expression Block::operator()(
        dynet::ComputationGraph *cg,
        std::unordered_map<std::string, Expression> expr_inputs,
        std::initializer_list<unsigned> runtime_index)
    {
        assert(freezed);
        vector<dynet::VariableIndex> inputs;
        for (auto &id_name : input_nodes)
        {
            assert(expr_inputs.count(id_name.second));
            inputs.push_back(expr_inputs[id_name.second].i);
        }
        assert(runtime_index.size() == runtime_nodes.size());
        vector<vector<unsigned>> runtime_indices;
        for (auto x : runtime_index)
            runtime_indices.push_back({x});
        dynet::FunctorNode *node = new dynet::FunctorNode(inputs, runtime_indices, this);
        node->is_function = true;
        node->n_output = output_nodes.size();
        // if (memo_block_type < 0) memo_block_type = node->autobatch_sig(*cg, cg->sigmap);
        // node->_type = memo_block_type;
        VariableIndex nid = cg->add_function_node(node);
        for (int oid = 0; oid < (int)output_nodes.size(); oid++)
        {
            dynet::GetNode *get_node = new dynet::GetNode({nid}, output_nodes[oid].dim);
            get_node->is_get = true;
            get_node->index = oid;
            cg->add_function_node(get_node);
        }

        Expression out;
        // if there is only one output, we use the get expression to avoid necessary operator[]
        if (output_nodes.size() == 1)
        {
            out = Expression(cg, nid + 1);
            out.n_output = 1;
        }
        else
        {
            out = Expression(cg, nid);
            out.n_output = output_nodes.size();
        }
        return out;
    }

    Expression Block::placeholder(const Dim &d, std::string name)
    {
        if (freezed)
        {
            DYNET_RUNTIME_ERR("call Block::placeholder after freezing");
            return Expression();
        }
        Expression expr = dynet::placeholder(this, d, name);
        for (auto &node_name : input_nodes)
        {
            if (node_name.second == name)
            {
                fprintf(stderr, "[WARNING]: doubled name %s\n", name.c_str());
                name += "_";
            }
        }
        input_nodes.push_back({expr.i, name});
        return move(expr);
    }

    void Block::output(const std::vector<dynet::Expression> &exprs)
    {
        if (freezed)
            return;
        one_batch_size = 0u;
        for (auto &expr : exprs)
        {
            if (nodes[expr.i]->is_function)
                throw runtime_error("function should not be in the output position");
            for (auto kv : input_nodes)
                assert(expr.i != kv.first);
            output_indices[expr.i] = output_indices.size();
            one_batch_size += nodes[expr.i]->dim.size();
        }
    }

    Expression Block::lookup(dynet::LookupParameter p)
    {
        if (freezed)
        {
            DYNET_RUNTIME_ERR("call Block::lookup after freezing");
            return Expression();
        }
        Expression expr = dynet::Expression(this, this->add_lookup(p, 0u));
        runtime_nodes.push_back({expr.i, nt::lookup});
        return move(expr);
    }

    Expression Block::pickneglogsoftmax(const Expression &x)
    {
        if (freezed)
        {
            DYNET_RUNTIME_ERR("call Block::pickneglogsoftmax after freezing");
            return Expression();
        }
        Expression expr = dynet::pickneglogsoftmax(x, (unsigned)0);
        runtime_nodes.push_back({expr.i, nt::pnls});
        return move(expr);
    }

    void Block::freeze()
    {
        assert(n_input >= n_params);
        assert(n_params >= 0);
        assert(output_indices.size());
        freezed = true;
        nfx_cache.resize(nodes.size());
        node2offset.resize(nodes.size());
        node2batch.resize(nodes.size());

        vector<int> node2type;
        vector<vector<int>> node2args;
        vector<int> in_nodes;
        dynet::SigMap sigmap;

        for (int nid = n_input; nid < (int)nodes.size(); ++nid)
        {
            // skip input nodes
            auto node = nodes[nid];
            int sig = node->autobatch_sig(*this, sigmap);
            assert(sig > 0);
            // output node is executed singlly.
            node2type.push_back(sig + (output_indices.count(nid) ? 1e3 * (output_indices[nid] + 1) : 0));
            node2args.push_back({});
            bool is_innode = false;
            for (auto arg : node->args)
            {
                node2args.back().push_back(arg - n_input);
                is_innode = is_innode || ((arg < n_input) && (arg >= n_params));
            }
            if (is_innode)
                in_nodes.push_back(nid);
        }

        // execute the parameter nodes;
        for (int nid = 0; nid < n_params; ++nid)
        {
            batches.push_back({});
            auto &my_batch = batches.back();
            my_batch.ids.resize(1, nid);
            auto param_node = static_cast<ParameterNode *>(nodes[nid]);
            assert(param_node && param_node->params.p != nullptr);
            my_batch.nfx = param_node->params.get_storage().values;
            // my_batch.nfx.v = nullptr;
            nfx_cache[nid].v = nullptr;
            node2batch[nid] = nid;
            node2offset[nid] = 0;
            get_nfx(nid);
            // memory_allocate(my_batch);
            // execute(my_batch);
        }

        int bid = n_params;
        set<int> output_constraint;
        for (auto &kv : output_indices)
            output_constraint.insert(kv.first - n_input);
        std::string alg = (opt&1) ? "ooc" : "dynet";
        OoC::Pattern pattern = pattern_cache.get_pattern(node2args, node2type, alg);
        assert(batches.size() == n_params);
        memory_allocation_order = pattern.mem_allocation_order;
        for (auto &bid : memory_allocation_order)
            bid += n_params;
        for (auto &ids : pattern.batch_ids)
        {
            assert(ids.size());
            batches.push_back({});
            auto &batch = batches.back();
            int output_cnt = 0;
            for (auto id : ids)
            {
                int nid = id + n_input;
                batch.ids.push_back(nid);
                node2batch[nid] = bid;
            }
            ++bid;
        }

        // find proper memory arangement for output nodes
        for (auto bid : memory_allocation_order)
        {
            int output_cnt = 0;
            for (int i = 0; i < batches[bid].ids.size(); i++)
            {
                int nid = batches[bid].ids[i];
                if (output_indices.count(nid))
                {
                    output_cnt++;
                    output_nodes.push_back({nodes[nid]->dim, nid, bid, output_indices[nid]});
                }
            }
            // all nodes of ids should be in output nodes
            if (!(output_cnt == 0 || 1 == batches[bid].ids.size()))
            {
                throw runtime_error("does not support partial output");
            }
        }
        sort(output_nodes.begin(), output_nodes.end(), [](const output_t &o1, const output_t &o2)
             { return o1.idx < o2.idx; });

        // generate autobatch_concat
        vector<int> concat(n_input, -1);
        for (auto nid : in_nodes)
        {
            auto node = nodes[nid];
            vector<int> batch_concat = node->autobatch_concat(*this);
            assert(batch_concat.size() == node->args.size());
            int idx = 0;
            for (auto arg : node->args)
            {
                if (arg < n_input)
                {
                    if (concat[arg] >= 0)
                        assert(concat[arg] == batch_concat[idx]);
                    concat[arg] = batch_concat[idx];
                }
                idx++;
            }
        }
        assert(autobatch_concat.size() == 0);
        for (auto &kv : input_nodes)
        {
            assert(concat[kv.first] != -1);
            autobatch_concat.push_back(concat[kv.first]);
        }
        if (opt >= 2)
            aot_analysis();
        
        if (dynet::profiling_flag) {
            cout << as_string(true) << endl;
        }
    }

    void Block::reset()
    {
        for (int bid = n_params; bid < (int)batches.size(); ++bid)
            batches[bid].nfx.v = nullptr;
        for (int nid = n_params; nid < (int)nodes.size(); ++nid)
            nfx_cache[nid].v = nullptr;
    }

    void Block::forward(
        const vector<const Tensor *> &xs,
        vector<Tensor *> &ys,
        const vector<vector<unsigned>> &runtime_indices,
        int batch_size)
    {
        assert(freezed);
        ComputationGraph::batch_size = batch_size;
        reset();
        for (int nid = n_params; nid < (int)nodes.size(); nid++)
            nodes[nid]->dim.bd *= batch_size;

        // set nfx for input nodes
        DYNET_ASSERT(xs.size() == input_nodes.size(), "input check error in Block::forward");
        for (size_t i = 0; i < xs.size(); i++)
        {
            nfx_cache[input_nodes[i].first] = *xs[i];
        }

        // set nfx for output nodes
        assert(output_nodes.size() == ys.size());
        for (int oid = 0; oid < output_nodes.size(); ++oid)
        {
            assert(ys[oid]->v != nullptr);
            batches[output_nodes[oid].bid].nfx.v = ys[oid]->v;
            nfx_cache[output_nodes[oid].nid] = *ys[oid]; 
        }

        // set pindices for lookup nodes
        assert(runtime_nodes.size() == runtime_indices.size());

        for (int i = 0; i < runtime_nodes.size(); ++i)
        {
            auto &runtime_node = runtime_nodes[i];
            if (runtime_node.second == nt::lookup)
            {
                auto node = static_cast<dynet::LookupNode *>(nodes[runtime_node.first]);
                node->pindex = nullptr;
                node->pindices = &runtime_indices[i];
            }
            else if (runtime_node.second == nt::pnls)
            {
                auto node = static_cast<dynet::PickNegLogSoftmax *>(nodes[runtime_node.first]);
                node->pval = nullptr;
                node->pvals = &runtime_indices[i];
            }
        }

        if (aot_analysed){
            memory_allocate_opt(batch_size);
            for (int bid = n_params; bid < (int) batches.size(); bid++){
                execute_opt(bid, batch_size);
            }
        }
        else {
            for (auto bid : memory_allocation_order)
                memory_allocate(bid);

            for (int bid = n_params; bid < (int)batches.size(); bid++)
                execute(bid);
        }

        for (int nid = n_params; nid < (int)nodes.size(); nid++)
            nodes[nid]->dim.bd /= batch_size;
    }

    void Block::memory_allocate(int bid)
    {
        auto &my_batch = batches[bid];
        global_timer.start("memory allocation");
        auto &batch_ids = my_batch.ids;
        auto &nfx = my_batch.nfx;
        assert(batch_ids.size());
        if (nodes[my_batch.ids.front()]->is_function)
        {
            auto node = nodes[my_batch.ids.front()];
            if (batch_ids.size() != 1)
            {
                my_batch.concat = node->autobatch_concat(*this);
                my_batch.pseudo_node = node->autobatch_pseudo_node(*this, batch_ids);
                if (my_batch.pseudo_node != nullptr)
                {
                    my_batch.pseudo_node->device = nodes[batch_ids.front()]->device;
                }
            }
        }
        else if (batch_ids.size() == 1)
        {
            VariableIndex curr_node = batch_ids[0];
            const Node *node = nodes[curr_node];
            nfx.d = node->dim;
            nfx.device = node->device;
            nfx.mem_pool = DeviceMempool::FXS;
            auto mempool = node->device->pools[(int)DeviceMempool::FXS];
            if (curr_node < n_params)
                mempool = node->device->pools[(int)DeviceMempool::PS];
            // assert(curr_node > n_params);

            if (nfx.v == nullptr)
            {
                nfx.v = static_cast<float *>(
                    mempool->allocate(nfx.d.size() * sizeof(float)));
            }
            if (nfx.v == nullptr)
            {
                DYNET_RUNTIME_ERR("Ran out of memory when allocating for node "
                                  << curr_node << ", allocating FWD memory.");
            }
            const size_t aux_size = node->aux_storage_size();
            if (aux_size)
            {
                node->aux_mem = mempool->allocate(aux_size);
                if (!node->aux_mem)
                {
                    DYNET_RUNTIME_ERR(
                        "Ran out of auxiliary memory when allocating for node " << curr_node);
                }
            }
        }
        else
        {
            const Node *node = nullptr;
            size_t tot_main = 0, tot_aux = 0;
            for (auto curr_node : batch_ids)
            {
                node = nodes[curr_node];
                node2offset[curr_node] = tot_main;
                tot_main += node->dim.size();
                node->aux_mem = (void *)tot_aux;
                tot_aux += node->aux_storage_size();
            }

            assert(node != nullptr);
            auto mempool = node->device->pools[(int)DeviceMempool::FXS];
            float *&head_main = nfx.v;
            if (head_main == nullptr)
                head_main = static_cast<float *>(
                    mempool->allocate(tot_main * sizeof(float)));
            if (head_main == nullptr)
            {
                DYNET_RUNTIME_ERR("Ran out of memory when executing batch, allocating FWD memory.");
            }

            char *head_aux = nullptr;
            if (tot_aux > 0)
            {
                head_aux = static_cast<char *>(mempool->allocate(tot_aux));
                if (head_aux == nullptr)
                {
                    DYNET_RUNTIME_ERR("Ran out of memory when executing node, allocating FWD memory.");
                }
                for (auto curr_node : batch_ids)
                    nodes[curr_node]->aux_mem =
                        (void *)(head_aux + (ptrdiff_t)nodes[curr_node]->aux_mem);
            }

            my_batch.concat = node->autobatch_concat(*this);
            my_batch.pseudo_node = node->autobatch_pseudo_node(*this, batch_ids);
            if (my_batch.pseudo_node != nullptr)
            {
                my_batch.pseudo_node->aux_mem = head_aux;
                my_batch.pseudo_node->device = node->device;
            }
            else
                nodes[batch_ids[0]]->aux_mem = head_aux;

            nfx.device = node->device;
            nfx.mem_pool = DeviceMempool::FXS;
            nfx.d = Dim({(unsigned int)tot_main});
        }
        global_timer.stop("memory allocation");
    }

    void Block::execute(int bid)
    {
        auto &my_batch = batches[bid];
        global_timer.start("execution");
        string current_batch_name;
        Tensor temp_nfx;
        vector<const Tensor *> xs(16);

        if (dynet::profiling_flag > 1)
        {
            VariableIndex nid = my_batch.ids[0];
            Node *node = nodes[nid];
            current_batch_name = "FWD " + node->as_dummy_string();
            dynet::timer.start(current_batch_name);
        }

        if (nodes[my_batch.ids.front()]->is_get)
        {
            // do nothing
        }
        else if (my_batch.ids.size() == 1)
        {
            VariableIndex nid = my_batch.ids[0];
            Node *node = nodes[nid];
            xs.resize(node->arity());
            unsigned ai = 0;
            for (auto arg : node->args)
            {
                xs[ai] = &get_nfx(arg);
                ++ai;
            }
            global_timer.start("computation");
            if (dynet::profiling_flag > 1)
            {
                vector<string> input_dims;
                for (auto &x : xs)
                {
                    ostringstream o;
                    o << x->d << "@" << x->v;
                    input_dims.push_back(o.str());
                }
                cout << "[" << name << "::forward] out{";
                for (auto id : my_batch.ids)
                    cout << id << "," << my_batch.nfx.v;
                cout << "} = " << node->as_string(input_dims);
            }
            if (node->is_function)
            {
                FunctorNode *functor_node = static_cast<FunctorNode *>(node);
                vector<Tensor *> ys;
                for (int oid = 0; oid < functor_node->n_output; ++oid)
                    ys.push_back(&batches[bid + 1 + oid].nfx);
                if (dynet::profiling_flag > 1)
                {
                    cout << "dims={";
                    for (auto y : ys)
                        cout << y->d << ",";
                    cout << "}" << endl;
                }
                functor_node->forward(xs, ys);
            }
            else
            {
                if (dynet::profiling_flag > 1)
                    cout << "dim=" << my_batch.nfx.d << endl;
                node->forward(xs, my_batch.nfx);
            }
            global_timer.stop("computation");
        }
        else
        {
            size_t arity = my_batch.concat.size();
            Node *node = my_batch.pseudo_node;
            if (node == nullptr)
                node = nodes[my_batch.ids[0]];
            my_batch.arg_nfxs.resize(arity);
            global_timer.start("memtransfer");
            for (size_t i = 0; i < arity; i++)
            {
                if (!my_batch.concat[i])
                {
                    my_batch.arg_nfxs[i] = &get_nfx(node->args[i]);
                }
                else
                {
                    Tensor *my_xsi = new Tensor;
                    my_xsi->device = node->device;
                    my_xsi->mem_pool = DeviceMempool::FXS;
                    global_timer.start("check contig");

                    auto it = my_batch.ids.begin();
                    auto itend = my_batch.ids.end();
                    VariableIndex aid = nodes[*(it++)]->args[i];
                    float *min_node = get_nfx(aid).v;
                    unsigned int tot_arg = nodes[aid]->dim.size();
                    bool contig = true;
                    while (it != itend && contig)
                    {
                        aid = nodes[*(it++)]->args[i];
                        float *v = get_nfx(aid).v;
                        contig = contig && (v == (min_node + tot_arg));
                        tot_arg += nodes[aid]->dim.size();
                    }

                    global_timer.stop("check contig");
                    if (contig)
                    {
                        my_xsi->v = min_node;
                        my_xsi->d = Dim({tot_arg});
                        my_batch.concat[i] = 2;
                    }
                    else
                    {
                        
                        combine_tensors(my_batch.ids, i, *my_xsi);
                    }
                    my_batch.arg_nfxs[i] = my_xsi;
                }
            }
            global_timer.stop("memtransfer");
            node->autobatch_reshape(*this, my_batch.ids, my_batch.concat, my_batch.arg_nfxs, my_batch.nfx);
            global_timer.start("computation");
            if (dynet::profiling_flag > 1)
            {
                vector<string> input_dims;
                int i = 0;
                for (auto &x : my_batch.arg_nfxs)
                {
                    ostringstream o;
                    o << x->d << "@" << x->v;
                    o << my_batch.concat[i++];
                    input_dims.push_back(o.str());
                }
                cout << "[" << name << "::forward] out{";
                for (auto id : my_batch.ids)
                    cout << id << ",";
                cout << "}@" << my_batch.nfx.v << "=" << node->as_string(input_dims);
                my_batch.nfx.check();
            }
            if (node->is_function)
            {
                auto functor_node = static_cast<FunctorNode *>(node);
                vector<Tensor *> ys;
                for (int oid = 0; oid < functor_node->n_output; ++oid)
                    ys.push_back(&batches[oid + bid + 1].nfx);
                if (dynet::profiling_flag > 1)
                {
                    cout << ",dim={";
                    for (auto &y : ys)
                        cout << y->d << ",";
                    cout << "}" << endl;
                }
                functor_node->forward(my_batch.arg_nfxs, ys);
            }
            else
            {
                if (dynet::profiling_flag > 1)
                    cout << ",dim=" << my_batch.nfx.d << endl;
                node->forward(my_batch.arg_nfxs, my_batch.nfx);
            }
            global_timer.stop("computation");
        }
        if (dynet::profiling_flag > 1)
            timer.stop(current_batch_name);

        global_timer.stop("execution");
        return;
    }

    const Tensor &Block::get_nfx(VariableIndex i)
    {
        if (nfx_cache[i].v == nullptr)
        {
            if (!(node2batch[i] < batches.size()))
            {
                fprintf(stderr, "[get_nfx::ERROR]: i %d, bid %d, batches.size() %ld\n",
                        i, node2batch[i], batches.size());
                assert(false);
            }
            const Tensor &bt = batches[node2batch[i]].nfx;
            Tensor &t = nfx_cache[i];
            t.v = bt.v + node2offset[i];
            t.d = nodes[i]->dim;
            t.mem_pool = bt.mem_pool;
            t.device = bt.device;
        }
        return nfx_cache[i];
    }

    // To minimize the number of host-to-device memory copies, we put a bunch of
    // data in contiguous memory. Since we need to pass both pointers and sizes,
    // we use this union.
    union CopyArgs
    {
        float *ptr;
        std::size_t n;
    };
    static_assert(sizeof(float *) == sizeof(std::size_t),
                  "Pointer size must be the same size as size_t");

    void Block::combine_tensors(const std::vector<dynet::VariableIndex> &batch_ids,
                                int aid, dynet::Tensor &tout)
    {
        
        // global_timer.start("EXT combine tensors");
        // global_timer.start("EXT combine preparation");
        AlignedMemoryPool *mempool = tout.device->pools[(int)DeviceMempool::FXS];
        // Determine needed memory for tout and get list of nodes corresponding to
        // specified argument.
        size_t total_dsize = 0;
        vector<VariableIndex> arg_nodes(batch_ids.size());
        for (unsigned i = 0; i < batch_ids.size(); ++i)
        {
            const auto nid = nodes[batch_ids[i]]->args[aid];
            total_dsize += nodes[nid]->dim.size();
            arg_nodes[i] = nid;
        }
        DYNET_ASSERT(total_dsize <= (unsigned)(-1),
            "ooc::block mem allocation error");
        tout.d = Dim({total_dsize});

        global_timer.cumint("n_combine", 1);
        global_timer.cumint("n_memcpy", total_dsize * sizeof(float));
        // global_timer.stop("EXT combine preparation");

        // allocate memory for tout
        // global_timer.start("EXT combined allocation");
        float *dest =
            static_cast<float *>(mempool->allocate(total_dsize * sizeof(float)));
        // global_timer.stop("EXT combined allocation");
        // global_timer.start("EXT prepare args");

#if HAVE_CUDA
        vector<CopyArgs> locs(batch_ids.size() * 3);
        size_t i = 0;
        size_t max_length = 0;
        const size_t TRG = batch_ids.size();
        const size_t LEN = batch_ids.size() * 2;
#endif
        tout.v = dest;
        // copy
        for (const auto id : arg_nodes)
        {
            const size_t sz = nodes[id]->dim.size();

            float *my_src = get_nfx(id).v;
            if (tout.device->type == DeviceType::CPU)
            {
                memcpy(dest, my_src, sz * sizeof(float));
            }
            else if (tout.device->type == DeviceType::GPU)
            {
#if HAVE_CUDA
                locs[i].ptr = my_src;
                locs[i + TRG].ptr = dest;
                locs[i + LEN].n = sz;
                if (max_length < sz)
                    max_length = sz;
                i++;
#endif
            }
            else
            {
                throw std::runtime_error("Bad device type");
            }
            dest += sz; // pointer arith
        }
        // global_timer.stop("EXT prepare args");
        if (tout.device->type == DeviceType::GPU)
        {
#if HAVE_CUDA
            // global_timer.start("EXT idx transfer");
            size_t req_sz = batch_ids.size() * 3 * sizeof(CopyArgs);
            void *basemem = mempool->allocate(req_sz);
            float **srcs = static_cast<float **>(basemem);
            float **trgs = static_cast<float **>(basemem) + TRG;
            std::size_t *lens = static_cast<std::size_t *>(basemem) + LEN;
            CUDA_CHECK(cudaMemcpyAsync(basemem,
                                       &(locs)[0],
                                       locs.size() * sizeof(CopyArgs),
                                       cudaMemcpyHostToDevice,
                                       static_cast<Device_GPU *>(tout.device)->estream->stream()));
            // global_timer.stop("EXT idx transfer");
            // global_timer.start("EXT memcpy");
            gpu::parallel_memcpy(batch_ids.size(), max_length, srcs, trgs, lens);
            // global_timer.stop("EXT memcpy");
#endif
        }
        else if (tout.device->type == DeviceType::CPU)
        {
            ; // Nothing more to do, memory was already copied.
        }
        else
        {
            throw std::runtime_error("Bad device type");
        }
        // global_timer.stop("EXT combine tensors");
    }

    Block::~Block()
    {
        for (auto &batch : batches)
        {
            delete batch.pseudo_node;
            batch.pseudo_node = nullptr;
            for (size_t i = 0; i < batch.arg_nfxs.size(); ++i)
            {
                if (batch.concat[i])
                {
                    delete batch.arg_nfxs[i];
                    batch.arg_nfxs[i] = nullptr;
                }
            }
        }
        batches.clear();
    }

    int Block::self_check()
    {
        vector<int> memid(nodes.size(), -1);
        int n_combine = 0;
        int idx = 0;
        for (auto &bid : memory_allocation_order)
        {
            for (auto id : batches[bid].ids)
                memid[id] = idx++;
        }
        for (auto &batch : batches)
        {
            assert(batch.ids.size());
            auto examplar = batch.ids.front();
            int n_arg = nodes[examplar]->args.size();
            for (auto id : batch.ids)
                assert(nodes[id]->args.size() == n_arg);
            for (int aid = 0; aid < n_arg; ++aid)
            {
                int example_arg = nodes[examplar]->args[aid];
                if (example_arg < n_input)
                    continue;
                int pos = memid[example_arg] - 1;
                for (auto id : batch.ids)
                {
                    if (memid[nodes[id]->args[aid]] < 0)
                        break;
                    if (memid[nodes[id]->args[aid]] != ++pos)
                    {
                        n_combine += 1;
                        cout << "mem combine for batch ";
                        for (auto id : batch.ids)
                            cout << id << ",";
                        cout << "aid" << aid << ", inputs";
                        for (auto id : batch.ids)
                            cout << nodes[id]->args[aid] << ",";
                        cout << endl;
                        break;
                    }
                }
            }
        }
        return n_combine;
    }

    string Block::as_string(bool verbose)
    {
        if (!freezed)
            return "not freezed";
        if (!verbose)
            return name;
        ostringstream s;
        s << "-----------" << name << "---------" << endl;
        s << "id: " << id << endl;
        s << "opt: " << opt << endl;
        s << "aot_analysed: " << aot_analysed << endl;
        int idx = 0;
        for (auto node : nodes)
        {
            assert(node != nullptr);
            vector<string> args;
            for (auto arg : node->args)
            {
                args.push_back(to_string(arg));
            }
            s << idx << "\t:" << node->as_string(args) << node->dim
              << "," << node2batch[idx] << "," << node2offset[idx] << endl;
            idx++;
        }
        s << "n_params: " << n_params <<", n_input:" << n_input << endl;
        s << memory_allocation_order.size() << " batches: ";
        for (int bid = n_params; bid < (int)batches.size(); bid++){
            s << "[" << bid << "]{";
            for (auto id : batches[bid].ids)
                s << id << ", ";
            s << "}, ";
        }
        // for (auto &bid : memory_allocation_order)
        // {
        //     s << "[" << bid << "]{";
        //     for (auto id : batches[bid].ids)
        //         s << id << ", ";
        //     s << "}, ";
        // }
        s << endl;
        s << "input_nodes(nid, name, concat): ";
        assert(input_nodes.size() == autobatch_concat.size());
        for (int i = 0; i < input_nodes.size(); i++)
            s << "(" << input_nodes[i].first << "," << input_nodes[i].second << ", " << autobatch_concat[i] << "), ";
        s << endl;
        s << "output_nodes(nid,bid,is_examplar,dim,idx): ";
        assert(output_nodes.size() == output_indices.size());
        for (auto &output : output_nodes)
        {
            s << "(" << output.nid << ", " << output.bid << ","
              << ","
              << output.dim << "," << output.idx << "), ";
        }
        s << endl;
        s << "one_batch_size: " << one_batch_size << endl;
        s << "runtime_nodes: ";
        for (auto kv : runtime_nodes)
            s << "(" << kv.first << "," << dynet::nt::type2name[kv.second] << "), ";
        s << endl;
        s << "mem_combine: " << self_check() << endl;
        if (aot_analysed) {
            s << "tot_mem: " << tot_mem << endl;
            for (int bid = 0; bid < batches.size(); bid ++){
                vector<string> inputs;
                auto & batch = batches[bid];
                auto examplar = nodes[batch.ids.front()];
                for (int aid = 0; aid < examplar->arity(); aid++){
                    ostringstream o;
                    size_t l, r;
                    if (!batch.concat[aid]) {
                        l = offsets[examplar->args[aid]] & 0xffff;
                        r = l + nodes[examplar->args[aid]]->dim.size();
                    }
                    else {
                        auto contig = batch.gathers[aid].contig;
                        if (contig == 0 || contig == 2){
                            l = (contig == 0)? batch.gathers[aid].dst_offset : (size_t)batch.arg_nfxs[aid]->v;
                            r = l + batch.arg_nfxs[aid]->d.size();
                            s << ((contig == 0)?"(R)":"(C)");
                            s << "[" << l << "," << r << "]=Gather(";
                            for (auto nid: batch.gathers[aid].src_ids) {
                                s << "(" << offsets[nid] << "," << nodes[nid]->dim.size() << "),";
                            }
                            s << ")" << endl;
                        }
                        else if (contig == 1){
                            l = offsets[batch.gathers[aid].src_ids.front()];
                            r = l + batch.arg_nfxs[aid]->d.size();
                        }
                    }
                    o << "[" << l << "," << r << "]";
                    inputs.push_back(o.str());
                }
                s << "B" << bid << ":";
                s << "[" << offsets[batch.ids.front()] << "," << offsets[batch.ids.front()] + batch.nfx.d.size() << "] = ";
                s << examplar -> as_string(inputs);
                s << endl;
            }
        }
        s << "-------------------------------------" << endl;
        return s.str();
    }

    void Block::memory_allocate_opt(int batch_size){
        auto mempool = nodes.front()->device->pools[(int)DeviceMempool::FXS];
        base = (float*)mempool->allocate(batch_size * tot_mem * sizeof(float));
        if (base == nullptr) {
            DYNET_RUNTIME_ERR("Ran out of memory when allocating FWD memory for " << name << ".");
        }
        for (int nid = 0; nid < nodes.size(); nid ++){
            if (nfx_cache[nid].v == nullptr){
                nfx_cache[nid].v = base + offsets[nid] * batch_size;
                assert(nfx_cache[nid].v + nodes[nid]->dim.size() <= (base + batch_size * tot_mem));
            }
        }
        for (int bid = n_params; bid < batches.size(); ++bid){
            auto & my_batch = batches[bid];
            if (my_batch.nfx.v == nullptr) {
                my_batch.nfx.v = nfx_cache[my_batch.ids.front()].v;
            }
        }
        if (dynet::profiling_flag > 1){
            cout << "Temp Memory: " << base << ", " << base +  batch_size * tot_mem << endl;
            for (int nid = n_params; nid < n_input; ++nid){
                cout << "NID" << nid << ":" << nfx_cache[nid].v << "," << nfx_cache[nid].d << endl;
            }
            for (int bid = 0; bid < batches.size(); bid ++){
                vector<string> inputs;
                auto & batch = batches[bid];
                auto examplar = nodes[batch.ids.front()];
                for (int aid = 0; aid < examplar->arity(); aid++){
                    ostringstream o;
                    float *l, *r;
                    if (!batch.concat[aid]) {
                        l = nfx_cache[examplar->args[aid]].v;
                        r = l + nodes[examplar->args[aid]]->dim.size();
                    }
                    else {
                        int contig = batch.gathers[aid].contig;
                        if (contig == 0 || contig == 2) {
                            l = (contig == 0)? base + batch.gathers[aid].dst_offset * batch_size : batch.arg_nfxs[aid]->v;
                            r = l + batch.arg_nfxs[aid]->d.size() * batch_size;
                            cout << ((contig == 0)? "(R)":"(C)");
                            cout << "[" << l  << "," << r << "] = Gather(";
                            for (auto nid: batch.gathers[aid].src_ids) 
                                cout << "(" << nfx_cache[nid].v << "," << nodes[nid]->dim.size() << "),";
                            cout << ")" << endl;
                        }
                        else { // contig is 1
                            l = nfx_cache[batch.gathers[aid].src_ids.front()].v;
                            r = l + batch.arg_nfxs[aid]->d.size() * batch_size;
                        }
                    }
                    o << "[" << l << "," << r << "]";
                    inputs.push_back(o.str());
                }
                cout << "B" << bid << ":";
                cout << "[" << batch.nfx.v << "," << batch.nfx.v + batch.nfx.d.size() * batch_size << "] = ";
                cout << examplar -> as_string(inputs);
                cout << endl;
            }
        }
    }

    void Block::execute_opt(int bid, int batch_size){
        if (!aot_analysed) {
            execute(bid);
            return;
        }
        // gather
        auto& my_batch = batches[bid];
        assert(my_batch.nfx.d.bd == 1);
        my_batch.nfx.d.bd = batch_size;
        auto node = nodes[my_batch.ids.front()];
        for (int aid = 0; aid < node->arity(); aid++){
            if (!my_batch.concat[aid]) continue;
            auto &gather = my_batch.gathers[aid];
            auto t = const_cast<Tensor*>(my_batch.arg_nfxs[aid]);
            t->d.bd = gather.src_ids[0] >= n_params? batch_size:1;
            if (gather.contig == 0) {
                t->v = base + gather.dst_offset * batch_size;
                combine_tensors_opt(gather.src_ids, gather.lens, t, batch_size);
            }
            else if (gather.contig == 1){
                t->v = nfx_cache[gather.src_ids[0]].v;
            }
            else {
                // do nothing here.
            }
        }

        // execute
        size_t aux_size = 0;
        Node *pseudo_node = nullptr;
        if (my_batch.ids.size() > 1 && ((pseudo_node = node->autobatch_pseudo_node(*this, my_batch.ids)) != nullptr)) {
            pseudo_node->device = node->device;
            node = pseudo_node;
            aux_size = node->aux_storage_size();
        }
        else {
            for (auto nid: my_batch.ids) 
                aux_size += nodes[nid]->aux_storage_size();
        }
        if (aux_size) {
            auto mempool = node->device->pools[(int)DeviceMempool::FXS];
            node->aux_mem = mempool->allocate(aux_size);
        }
        node->autobatch_reshape(*this, my_batch.ids, my_batch.concat, my_batch.arg_nfxs, my_batch.nfx);
        if (dynet::profiling_flag > 1) {
            vector<string> input_dims;
            int i = 0;
            for (auto &x : my_batch.arg_nfxs)
            {
                ostringstream o;
                o << x->d << "@" << x->v;
                input_dims.push_back(o.str());
            }
            cout << "[" << name << "::forward] B" << bid << ": out{";
            for (auto id : my_batch.ids)
                cout << id << ",";
            cout << "}@" << my_batch.nfx.d << "@" << my_batch.nfx.v << "=" << node->as_string(input_dims) << endl;
            my_batch.nfx.check();
            for (auto x: my_batch.arg_nfxs) x->check();
        }
        node->forward(my_batch.arg_nfxs, my_batch.nfx);
        my_batch.nfx.d.bd = 1;
    }

    void Block::aot_analysis(){
        // do not support function node
        for (auto node: nodes) 
            if (node->is_function){
                std::cerr << "[Block]: doesn't support block with functor node" << endl;
                aot_analysed = false;
                return;
            }
        aot_analysed = true;
        for (int bid = n_params; bid < (int)batches.size(); bid++){
            auto & my_batch = batches[bid];
            auto node = nodes[my_batch.ids.front()];
            my_batch.concat = node->autobatch_concat(*this);
            my_batch.nfx.device = node->device;
            my_batch.nfx.mem_pool = DeviceMempool::FXS;
            size_t tot_main = 0;
            for (auto id: my_batch.ids) 
                tot_main += nodes[id]->dim.size();
            my_batch.nfx.d = Dim({tot_main});
            my_batch.nfx.v = nullptr;
        }
        
        // offset caculation
        size_t offset = 0;
        offsets.resize(nodes.size(),1000);
        for (int nid = 0; nid < n_params; ++nid) {
            offsets[nid] = (size_t)nfx_cache[nid].v;
        }
        for (auto bid: memory_allocation_order){
            bool is_io = false;
            for (auto& o: output_nodes) 
                if (bid == o.bid) 
                    is_io = true;
            if (is_io) continue;
            auto& my_batch = batches[bid];
            for (auto nid: my_batch.ids){
                offsets[nid] = offset;
                offset += nodes[nid]->dim.size();
            }
        }
        for (int bid = n_params; bid < batches.size(); ++bid) {
            auto & my_batch = batches[bid];
            auto examplar = nodes[my_batch.ids.front()];
            my_batch.gathers.resize(examplar->arity());
            my_batch.arg_nfxs.resize(examplar->arity());
            for (int aid = 0; aid < examplar->arity(); aid++){
                if (!my_batch.concat[aid]) {
                    if (aid >= n_params) {
                        throw("concat = 0 should be param node");
                    }
                    my_batch.arg_nfxs[aid] = &get_nfx(aid);
                    assert(nodes[aid]->dim.bd == 1);
                    continue;
                }
                Tensor* ptr = new Tensor;
                ptr->device = examplar->device;
                ptr->mem_pool = DeviceMempool::FXS;
                ptr->v = nullptr;
                size_t mem = 0;
                for (auto nid: my_batch.ids) 
                    mem += nodes[nodes[nid]->args[aid]]->dim.size();
                ptr->d = Dim({mem});
                my_batch.arg_nfxs[aid] = ptr;

                auto& gather = my_batch.gathers[aid];
                gather.src_ids.clear();
                gather.lens.clear();
                bool has_param, has_input, has_normal;
                has_param = has_input = has_normal = false;
                for (auto nid: my_batch.ids) {
                    int arg_nid = nodes[nid]->args[aid];
                    gather.lens.push_back(nodes[arg_nid]->dim.size());
                    gather.src_ids.push_back(arg_nid);
                    has_param |= (arg_nid < n_params);
                    has_input |= (arg_nid >= n_params && arg_nid < n_input);
                    has_normal  |= arg_nid >= n_input;
                }
                // if (has_param & (has_input | has_normal)) {
                //     freezed = true;
                //     aot_analysed = false;
                //     cerr << as_string(true) << endl;
                //     cerr << "my_batch:"; for (auto id: my_batch.ids) cerr << id << ","; cerr << endl;
                //     cerr << "args:" << aid << ":"; for (auto id:my_batch.ids) cerr << nodes[id]->args[aid] << ","; cerr << endl;
                //     cerr << "has_param,has_input,has_normal:" << has_param << "," << has_input << "," << has_normal << endl;
                //     throw std::runtime_error("not supported arg");
                // }
                // necessary condition for contiguous: 100 | 010 | 001
                gather.contig = (has_param ^ has_input ^ has_normal) & !(has_param & has_input & has_normal);
                if (gather.contig){
                    if (has_normal) {
                        size_t min_offset = offsets[examplar->args[aid]];
                        for (auto nid: my_batch.ids) {
                            int arg_nid = nodes[nid]->args[aid];
                            if (!(gather.contig = (min_offset == offsets[arg_nid]))) 
                                break;
                            min_offset += nodes[arg_nid]->dim.size();
                        }
                    }
                    else if (has_param) {
                        float* min_ptr = get_nfx(examplar->args[aid]).v;
                        for (auto nid: my_batch.ids){
                            if (!(gather.contig = (min_ptr == get_nfx(nodes[nid]->args[aid]).v))) 
                                break;
                        }
                        // we can pre allocate memory for this computation
                        if (opt >= 4 && !gather.contig){
                            auto mempool = nodes.front()->device->pools[(int) DeviceMempool::PS];
                            ptr->v = (float*)mempool->allocate(ptr->d.size() * sizeof(float));
                            combine_tensors_opt(gather.src_ids, gather.lens, ptr, 1);
                            gather.contig = 2;
                        }
                    }
                    else if (has_input) {
                        gather.contig = gather.src_ids.size() == 1;
                    }
                }

                if (!gather.contig){
                    gather.dst_offset = offset;
                    for (auto len: gather.lens) {
                        offset += len;
                    }
                }
            }
        }
        tot_mem = offset;

        for (int nid = n_input; nid < nodes.size(); ++nid){
            nfx_cache[nid].v = nullptr;
            nfx_cache[nid].d = nodes[nid]->dim;
            nfx_cache[nid].mem_pool = DeviceMempool::FXS;
            nfx_cache[nid].device = nodes[nid]->device;
        }
    }

    void Block::combine_tensors_opt(
        const std::vector<size_t>& src_ids, 
        const std::vector<size_t>& lens, 
        const Tensor* tout, int batch_size){
        auto mempool = tout->device->pools[(int) DeviceMempool::FXS];
        float* dst = tout->v;

        if (profiling_flag > 1){
            cout << "combine_tensors_opt:" << endl;
            cout << "\tbatch_size:" << batch_size << endl;
            cout << "\tsrc_ids:";
            for (auto id: src_ids) cout << id << ",";
            cout << endl;
            cout << "\tlens:"; 
            for (auto id: lens) cout << id << ",";
            cout << endl;
            cout << "\tout:" << tout->d << "," << tout->v << endl; 
        }
        global_timer.cumint("n_combine", 1);
        // size_t sz = 0;
        // for (auto l: lens) sz += l;
        // cerr << "n_memcpy: " << (double)(sz * sizeof(float) * (size_t)batch_size) / (1 << 20) << std::endl;
        // global_timer.cumint("n_memcpy", (double)(sz * sizeof(float) * (size_t)batch_size) / (1<<20));
#if HAVE_CUDA
        vector<CopyArgs> locs(src_ids.size() * 3);
        const size_t TRG = src_ids.size();
        const size_t LEN = src_ids.size() * 2;
        size_t max_len = 0;
#endif 
        for (int i = 0; i < src_ids.size(); i++){
            if (tout->device->type == DeviceType::CPU){
                memcpy(dst, nfx_cache[src_ids[i]].v, batch_size * lens[i] * sizeof(float));
            }
#if HAVE_CUDA
            else if (tout->device->type == DeviceType::GPU){
                locs[i].ptr = nfx_cache[src_ids[i]].v;
                locs[i + TRG].ptr = dst;
                locs[i + LEN].n = lens[i] * batch_size;
                max_len = std::max(lens[i] * batch_size, max_len);
            }
#endif 
            else {
                throw runtime_error("bad device type");
            }
            dst += batch_size * lens[i];
        }

        if (tout->device->type == DeviceType::GPU){
#if HAVE_CUDA 
            size_t req_sz = src_ids.size() * 3 * sizeof(CopyArgs);
            void *basemem = mempool->allocate(req_sz);
            float **srcs = static_cast<float **>(basemem);
            float **trgs = static_cast<float **>(basemem) + TRG;
            std::size_t *lens = static_cast<std::size_t *>(basemem) + LEN;
            CUDA_CHECK(cudaMemcpyAsync(basemem,
                                       &(locs)[0],
                                       locs.size() * sizeof(CopyArgs),
                                       cudaMemcpyHostToDevice,
                                       static_cast<Device_GPU *>(tout->device)->estream->stream()));
            // global_timer.stop("EXT idx transfer");
            // global_timer.start("EXT memcpy");
            gpu::parallel_memcpy(src_ids.size(), max_len, srcs, trgs, lens);
#endif 
        }
        
    }

    std::unordered_map<std::string, int> Block::names;

    int Block::autobatch_sig(SigMap& sigmap){
        int& sig = sig_cache[&sigmap];
        if (sig == 0) {
            Sig s(nt::block);
            s.add_int(id);
            sig = sigmap.get_idx(s);
        } 
        return sig;
    }

} // namespace OoC

/**
 * output_indices: <nid, offset_indices>
 * output_nodes: <nid, bid, user_specified_order>
 */