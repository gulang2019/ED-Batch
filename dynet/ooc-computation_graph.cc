#include "dynet/ooc-computation_graph.h"
#include "dynet/devices.h"
#include "dynet/timing.h"
#include "dynet/expr.h"
#include "dynet/param-nodes.h"
#include "dynet/nodes-functor.h"


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

    int Block::block_id = 0;

    Expression Block::operator()(
        dynet::ComputationGraph *cg,
        std::unordered_map<std::string, Expression> expr_inputs,
        std::initializer_list<unsigned> lookup_index)
    {
        assert(freezed);
        vector<dynet::VariableIndex> inputs;
        for (auto &id_name : input_nodes)
        {
            assert(expr_inputs.count(id_name.second));
            inputs.push_back(expr_inputs[id_name.second].i);
        }
        assert(lookup_index.size() == lookup_nodes.size());
        vector<vector<unsigned>> lookup_indices;
        for (auto x : lookup_index)
            lookup_indices.push_back({x});
        dynet::FunctorNode *node = new dynet::FunctorNode(inputs, lookup_indices, this);
        // if (memo_block_type < 0) memo_block_type = node->autobatch_sig(*cg, cg->sigmap);
        // node->_type = memo_block_type;
        Expression out = Expression(cg, cg->add_function_node(node));
        out.n_output = output_dims.size();
        if (output_dims.size() > 1)
        {
            for (int oid = 0; oid < (int)output_dims.size(); oid++)
            {
                node->offsets.push_back(make_shared<ptrdiff_t>(0));
                dynet::GetNode *get_node = new dynet::GetNode({out.i}, output_dims[oid]);
                // if (memo_get_type < 0) memo_get_type = get_node->autobatch_sig(*cg, cg->sigmap);
                // get_node->_type = memo_get_type;
                get_node->offset = node->offsets.back();
                get_node->is_get = true;
                cg->add_function_node(get_node);
            }
        }
        return out;
    }

    Expression Block::placeholder(const Dim &d, std::string name)
    {
        if (freezed)
        {
            DYNET_RUNTIME_ERR("call Block::lookup after freezing");
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

    void Block::output(std::initializer_list<dynet::Expression> exprs)
    {
        if (freezed)
            return;
        one_batch_size = 0u;
        for (auto &expr : exprs)
        {
            for (auto kv : input_nodes)
                assert(expr.i != kv.first);
            output_nodes.push_back({expr.i, 0});
            output_dims.push_back(nodes[expr.i]->dim);
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
        lookup_nodes.push_back(expr.i);
        return move(expr);
    }

    void Block::freeze()
    {
        assert(n_input >= n_params);
        assert(n_params >= 0);
        assert(output_nodes.size());
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
            node2type.push_back(sig);
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
        for (int nid = 0; nid < n_params; ++nid) {
            batches.push_back({});
            auto& my_batch = batches.back();
            my_batch.ids.resize(1, nid);
            auto param_node = static_cast<ParameterNode*> (nodes[nid]);
            assert(param_node && param_node->params.p != nullptr);
            my_batch.nfx = param_node->params.get_storage().values;
            // my_batch.nfx.v = nullptr;
            nfx_cache[nid].v = nullptr;
            node2batch[nid] = nid;
            node2offset[nid] = 0;    
            // memory_allocate(my_batch);
            // execute(my_batch);
        }


        int bid = n_params;
        OoC::Pattern *pattern = pattern_cache.add_pattern(id, node2args, node2type);
        assert(batches.size() == n_params);
        memory_allocation_order = pattern->mem_allocation_order;
        for (auto &ids : pattern->batch_ids)
        {
            batches.push_back({});
            auto &batch = batches.back();
            bool is_output = false;
            for (auto id : ids)
            {
                int nid = id + n_input;
                batch.ids.push_back(nid);
                node2batch[nid] = bid;
                for (auto &output : output_nodes)
                {
                    if (output.first == nid)
                    {
                        is_output = true;
                        output.second = bid;
                    }
                }
            }
            if (is_output)
                assert(ids.size() == 1);
            ++bid;
        }

        // generate autobatch_concat
        vector<int> concat(n_input, -1);
        for (auto nid : in_nodes)
        {
            auto node = nodes[nid];
            vector<int> batch_concat = node->autobatch_concat(*this);
            assert(batch_concat.size() == node->args.size());
            cout << "innode: " << nid << " concat: ";
            for (auto c: batch_concat) cout << c << ",";
            cout << endl;
            int idx = 0;
            for (auto arg : node->args)
            {
                if (arg < n_input){
                    if (concat[arg] >= 0) assert(concat[arg] == batch_concat[idx]);
                    concat[arg] = batch_concat[idx];
                }
                idx++;
            }
        }
        assert(autobatch_concat.size() == 0);
        for (auto&kv : input_nodes)
        {
            assert(concat[kv.first] != -1);
            autobatch_concat.push_back(concat[kv.first]);
        }
    }

    void Block::reset()
    {
        for (int bid = n_params; bid < (int)batches.size(); ++bid)
            batches[bid].nfx.v = nullptr;
        for (int nid = n_params; nid < (int) nodes.size(); ++nid)
            nfx_cache[nid].v = nullptr;
    }

    void Block::forward(
        const vector<const Tensor *> &xs,
        Tensor &fx,
        const vector<vector<unsigned>> &lookup_indices,
        int batch_size)
    {
        assert(freezed);
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
        float *ptr = fx.v;
        assert(ptr != nullptr);

        for (auto& nid_bid : output_nodes)
        {
            batches[nid_bid.second].nfx.v = ptr;
            ptr += nodes[nid_bid.first]->dim.size();
        }
        
        // set pindices for lookup nodes
        assert(lookup_nodes.size() == lookup_indices.size());

        for (int i = 0; i < lookup_nodes.size(); ++i)
        {
            auto node = static_cast<dynet::LookupNode *>(nodes[lookup_nodes[i]]);
            node->pindex = nullptr;
            node->pindices = &lookup_indices[i];
        }

        for (auto bid: memory_allocation_order){
            memory_allocate(batches[bid + n_params]);
        }

        for (int bid = n_params; bid < (int)batches.size(); bid++){
            execute(batches[bid]);
        }

        for (int nid = n_params; nid < (int)nodes.size(); nid++)
            nodes[nid]->dim.bd /= batch_size;
    }

    void Block::memory_allocate(BatchInfo &my_batch)
    {
        auto &batch_ids = my_batch.ids;
        auto &nfx = my_batch.nfx;
        assert(batch_ids.size());
        if (batch_ids.size() == 1)
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
    }

    void Block::execute(BatchInfo &my_batch)
    {
        global_timer.start("execution");
        string current_batch_name;
        Tensor temp_nfx;
        vector<const Tensor *> xs(16);

        if (profiling_flag)
        {
            VariableIndex nid = my_batch.ids[0];
            Node *node = nodes[nid];
            current_batch_name = "FWD " + node->as_dummy_string();
            dynet::timer.start(current_batch_name);
        }

        if (my_batch.ids.size() == 1)
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
            node->forward(xs, my_batch.nfx);
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
                        global_timer.cumint("n_memtransfer", my_batch.ids.size() * nodes[my_batch.ids.front()]->dim.size());
                        global_timer.cumint("n_combine", 1);
                        combine_tensors(my_batch.ids, i, *my_xsi);
                    }
                    my_batch.arg_nfxs[i] = my_xsi;
                }
            }
            global_timer.stop("memtransfer");

            node->autobatch_reshape(*this, my_batch.ids, my_batch.concat, my_batch.arg_nfxs, my_batch.nfx);
            node->forward(my_batch.arg_nfxs, my_batch.nfx);
        }
        if (profiling_flag)
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
    union CopyArgs {
    float* ptr;
    std::size_t n;
    };
    static_assert(sizeof(float*) == sizeof(std::size_t),
                "Pointer size must be the same size as size_t");

    void Block::combine_tensors(const std::vector<dynet::VariableIndex> &batch_ids,
                         int aid, dynet::Tensor &tout)
    {
        global_timer.start("EXT combine tensors");
        global_timer.start("EXT combine preparation");
        AlignedMemoryPool *mempool = tout.device->pools[(int)DeviceMempool::FXS];
        // Determine needed memory for tout and get list of nodes corresponding to
        // specified argument.
        unsigned total_dsize = 0;
        vector<VariableIndex> arg_nodes(batch_ids.size());
        for (unsigned i = 0; i < batch_ids.size(); ++i)
        {
            const auto nid = nodes[batch_ids[i]]->args[aid];
            total_dsize += nodes[nid]->dim.size();
            arg_nodes[i] = nid;
        }
        tout.d = Dim({total_dsize});
        global_timer.stop("EXT combine preparation");

        // allocate memory for tout
        global_timer.start("EXT combined allocation");
        float *dest =
            static_cast<float *>(mempool->allocate(total_dsize * sizeof(float)));
        global_timer.stop("EXT combined allocation");
        global_timer.start("EXT prepare args");

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
        global_timer.stop("EXT prepare args");
        if (tout.device->type == DeviceType::GPU)
        {
#if HAVE_CUDA
            global_timer.start("EXT idx transfer");
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
            global_timer.stop("EXT idx transfer");
            global_timer.start("EXT memcpy");
            gpu::parallel_memcpy(batch_ids.size(), max_length, srcs, trgs, lens);
            global_timer.stop("EXT memcpy");
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
        global_timer.stop("EXT combine tensors");
    }

    Block::~Block(){
        for (auto& batch: batches) {
            delete batch.pseudo_node;
            batch.pseudo_node = nullptr;
            for (size_t i = 0; i < batch.arg_nfxs.size(); ++i){
                if (batch.concat[i]){
                    delete batch.arg_nfxs[i];
                    batch.arg_nfxs[i] = nullptr;
                }
            }
        }
        batches.clear();
    }

    int Block::self_check(){
        vector<int> memid(nodes.size(), -1);
        int n_combine = 0;
        int idx = 0;
        for (auto & bid: memory_allocation_order) {
            for (auto id: batches[bid+n_params].ids)
                memid[id] = idx++;
        }
        for (auto & batch: batches){
            assert(batch.ids.size());
            auto examplar = batch.ids.front();
            int n_arg = nodes[examplar]->args.size();
            for (auto id: batch.ids) assert(nodes[id]->args.size() == n_arg);
            for (int aid = 0; aid < n_arg; ++aid){
                int example_arg = nodes[examplar]->args[aid];
                if (example_arg < n_input) continue;
                int pos = memid[example_arg]-1;
                for (auto id: batch.ids){
                    if (memid[nodes[id]->args[aid]] < 0) break;
                    if (memid[nodes[id]->args[aid]] != ++pos) {
                        n_combine += 1; 
                        cout << "mem combine for batch ";
                        for (auto id: batch.ids) cout << id << ",";
                        cout << "aid" << aid << ", inputs";
                        for (auto id: batch.ids) cout << nodes[id]->args[aid] << ",";
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
        if (!freezed) return "not freezed";
        if (!verbose) return name;
        ostringstream s;
        s << "-----------" << name << "---------" << endl;
        int idx = 0;
        for (auto node : nodes)
        {
            assert(node != nullptr);
            vector<string> args;
            for (auto arg: node->args) {
                args.push_back(to_string(arg));
            }
            s << idx << "\t:" << node->as_string(args) << node->dim 
            << ","<< node2batch[idx] << "," << node2offset[idx]  << endl;
            idx++;
        }
        s << "batches: ";
        for (auto& bid: memory_allocation_order) {
            s << "[" << bid << "]{";
            for (auto id: batches[bid + n_params].ids) s << id << ", ";
            s << "}, ";
        }
        s << endl;
        s << "input_nodes(nid, name, concat): ";
        assert(input_nodes.size() == autobatch_concat.size());
        for (int i = 0; i < input_nodes.size(); i++)
            s << "(" << input_nodes[i].first << "," << input_nodes[i].second << ", " << autobatch_concat[i] << "), ";
        s << endl;
        s << "output_nodes(nid, bid, dim): ";
        assert(output_nodes.size() == output_dims.size());
        for (int i = 0; i < (int) output_nodes.size(); i++){
            s << "(" << output_nodes[i].first << ", " << output_nodes[i].second << ", " << output_dims[i] << "), ";
        }
        s << endl;
        s << "one_batch_size: " << one_batch_size << endl;
        s << "lookups: ";
        for (auto nid: lookup_nodes) s << nid << ", ";
        s << endl;
        s << "mem_combine: " << self_check() << endl;
        s << "-------------------------------------" << endl;
        return s.str();
    }


    

} // namespace OoC
  /*
      snode parallel construction;
      snode construction
  */
  // n_marked_node: 2