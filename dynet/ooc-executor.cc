#include "dynet/ooc-executor.h"
#include "dynet/nodes-functor.h"
#include "dynet/ooc-scheduler.h"

#ifdef HAVE_CUDA
#include "dynet/gpu-ops.h"
#endif

using namespace dynet;
using namespace std;

namespace OoC
{
    void Executor::invalidate(){}
    void Executor::invalidate(unsigned i){}

    Executor::~Executor()
    {
        for (auto &batch : batches)
            delete batch.pseudo_node;
        for (Device *dev : device_manager->get_devices())
            dev->pools[(int)DeviceMempool::FXS]->free();
    }

    void Executor::init()
    {
        batches.clear();
        memory_blocks.clear();
    }

    const Tensor &Executor::forward()
    {
        return incremental_forward();
    }

    const Tensor &Executor::forward(VariableIndex upto)
    {
        return incremental_forward(upto);
    }

    const Tensor &Executor::get_value(VariableIndex i)
    {
        return incremental_forward(i);
    }

    void Executor::backward(bool full)
    {
        throw("not implementaed");
    }

    void Executor::backward(VariableIndex i, bool full)
    {
        throw("not implemented");
    }

    const Tensor &Executor::get_gradient(VariableIndex i)
    {
        throw("not implemented");
        return incremental_forward(i);
    }

    const Tensor &Executor::incremental_forward()
    {
        return incremental_forward(cg.nodes.size() - 1);
    }

    const Tensor &Executor::incremental_forward(VariableIndex upto)
    {
        init();
        
        if (cg.nodes[upto]->is_get)
        {
            auto get_node = static_cast<GetNode *>(cg.nodes[upto]);
            assert(cg.nodes[upto - get_node->index - 1]->is_function);
            auto func_node = static_cast<FunctorNode *>(cg.nodes[upto - get_node->index - 1]);
            upto = upto - get_node->index + func_node->n_output - 1;
        }

        nfx_cache.resize(upto + 1);
        node2offset.resize(upto + 1);
        node2size.resize(upto + 1);
        node2batch.resize(upto+1);
        for (int nid = 0; nid < (int)cg.nodes.size(); ++nid)
            node2size[nid] = cg.nodes[nid]->dim.size();

        BaseScheduler *scheduler = nullptr;
        if (ooc_autobatch_flag == 1)
            scheduler = &agenda_scheduler;
        else if (ooc_autobatch_flag >= 2) 
            scheduler = &rl_scheduler;
        else
            throw("bad ooc_autobatch_flag");

        global_timer.start("scheduling");
        scheduler->init(&cg, upto);
        global_timer.stop("scheduling");
        VariableIndex batch_id = 0;
        while (true)
        {
            int suc = -1;
            global_timer.start("scheduling");
            while(suc == -1){
                batches.push_back({});
                suc = scheduler->schedule(batches.back().ids);
            }
            global_timer.stop("scheduling");
            if (!suc){
                batches.pop_back();
                break;
            }
            for (int bid = batch_id; bid < (int)batches.size(); ++bid)
                for (auto nid: batches[bid].ids) 
                    node2batch[nid] = bid;
            global_timer.start("execution");
            for (int bid = batch_id; bid < (int)batches.size(); ++bid)
                memory_allocation(bid);
            for (int bid = batch_id; bid < (int)batches.size(); ++bid)
                execute(bid);
            global_timer.stop("execution");
            batch_id = batches.size();
        }
        scheduler->post_process();
        global_timer.cumint("n_kernels", batch_id);
        cout << "[OoC::Executor]: batch_strategy" << ooc_autobatch_flag << ":" << batch_id << "kernels" << endl;
        global_timer.start("execution");
        auto &t = get_nfx(upto);
        global_timer.stop("execution");

        as_vector(t);
        return t;
    }

    void Executor::memory_allocation(VariableIndex bid)
    {
        auto &my_batch = batches[bid];
        auto &nfx = my_batch.nfx;
        auto &batch_ids = my_batch.ids;
        // for function node do things other than allocate nfx
        if (cg.nodes[my_batch.ids.front()]->is_function)
        {
            if (batch_ids.size() != 1)
            {
                auto node = cg.nodes[my_batch.ids.front()];
                my_batch.concat = node->autobatch_concat(cg);
                my_batch.pseudo_node = node->autobatch_pseudo_node(cg, batch_ids);
                if (my_batch.pseudo_node != nullptr)
                {
                    my_batch.pseudo_node->device = cg.nodes[batch_ids.front()]->device;
                }
            }
        }
        else if (batch_ids.size() == 1)
        {
            VariableIndex curr_node = batch_ids[0];
            const Node *node = cg.nodes[curr_node];
            DYNET_ASSERT(node->device != nullptr,
                         "Attempt to access null device in "
                         "BatchedExecutionEngine::incremental_forward");
            // Save the node profile
            nfx.d = node->dim;
            nfx.device = node->device;
            nfx.mem_pool = DeviceMempool::FXS;
            // Allocate memory
            auto mempool = node->device->pools[(int)DeviceMempool::FXS];
            nfx.v = static_cast<float *>(mempool->allocate(node2size[curr_node] * sizeof(float)));
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
        { // here: batch_ids.size() > 1
            // Set up the configuration of each component node, including pointer
            // differential from the start of the batch.
            const Node *node = nullptr;
            size_t tot_main = 0, tot_aux = 0, my_main, my_aux;
            for (auto curr_node : batch_ids)
            {
                node = cg.nodes[curr_node];
                my_main = node2size[curr_node];
                my_aux = node->aux_storage_size();
                node2offset[curr_node] = tot_main;
                tot_main += my_main;
                node->aux_mem = (void *)tot_aux;
                tot_aux += my_aux;
            }

            // Allocate main/auxiliary memory for the batch
            auto mempool = node->device->pools[(int)DeviceMempool::FXS];

            float *head_main = static_cast<float *>(mempool->allocate(tot_main * sizeof(float)));

            if (head_main == nullptr)
            {
                DYNET_RUNTIME_ERR("Ran out of memory when executing batch " << bid << ", allocating FWD memory.");
            }
            // cout << "[MEM" << bid << "]: {";
            // for (auto id: batch_ids) cout << id << ",";
            // cout << "}:" << is_allocated << endl;

            // for(auto curr_node : batch_ids) nfxs[curr_node].v = head_main + node2diff[curr_node];
            char *head_aux = nullptr;
            if (tot_aux > 0)
            {
                head_aux = static_cast<char *>(mempool->allocate(tot_aux));
                if (head_aux == nullptr)
                {
                    DYNET_RUNTIME_ERR("Ran out of memory when executing node " << bid << ", allocating FWD memory.");
                }
                for (auto curr_node : batch_ids)
                    cg.nodes[curr_node]->aux_mem =
                        (void *)(head_aux + (ptrdiff_t)cg.nodes[curr_node]->aux_mem);
            }

            // Get the concatenation and pseudo-node info
            my_batch.concat = node->autobatch_concat(cg);
            my_batch.pseudo_node = node->autobatch_pseudo_node(cg, batch_ids);
            if (my_batch.pseudo_node != nullptr)
            {
                my_batch.pseudo_node->aux_mem = head_aux;
                my_batch.pseudo_node->device = node->device;
            }
            else
            {
                cg.nodes[batch_ids[0]]->aux_mem = head_aux;
            }
            // Set the size for the final output
            nfx.device = node->device;
            nfx.mem_pool = DeviceMempool::FXS;
            nfx.d = Dim({(unsigned int)tot_main});
            nfx.v = head_main;
        } // batch_ids.size() > 1 condition
    }

    void Executor::execute(VariableIndex bid)
    {
        Tensor temp_nfx;
        vector<const Tensor *> xs(16), ts(16);
        auto &my_batch = batches[bid];
        // Read in the stuff for this batch
        if (cg.nodes[my_batch.ids.front()]->is_get)
        {
            // do nothing
            auto node = cg.nodes[my_batch.ids.front()];
            node->autobatch_reshape(cg, my_batch.ids, my_batch.concat, xs, my_batch.nfx);
        }
        else if (my_batch.ids.size() == 1)
        { // execute a single node
            VariableIndex nid = my_batch.ids[0];
            Node *node = cg.nodes[nid];
            xs.resize(node->arity());
            unsigned ai = 0;
            for (VariableIndex arg : node->args)
            {
                xs[ai] = &get_nfx(arg);
                ++ai;
            }

            // vector<string> inputs;
            // ai = 0;
            // for (VariableIndex arg : node->args) {
            //     ostringstream o;
            //     o << arg << "," << xs[ai++]->d;
            //     inputs.push_back(o.str());
            // }
            // cout << "[forward] out{";
            // if (!node->is_function)
            //   for (auto id: my_batch.ids) cout << id << ",";
            // else for (auto id: my_batch.ids) cout << id << ",";
            // cout << "}=" << node->as_string(inputs) << endl;

            global_timer.start("computation");
            global_timer.lock();
            if (node->is_function)
            {
                FunctorNode *functor_node = static_cast<FunctorNode *>(node);
                vector<Tensor *> ys;
                for (int oid = 0; oid < functor_node->n_output; ++oid)
                    ys.push_back(&batches[bid + oid + 1].nfx);
                // cout <<"dim={";
                // for (auto y:ys) cout << y->d << ",";
                // cout << "}" << endl;
                functor_node->forward(xs, ys);
            }
            else
            {
                // cout << ",dim=" << my_batch.nfx.d << endl;
                node->forward(xs, my_batch.nfx);
            }

            global_timer.unlock();
            global_timer.stop("computation");
        }
        else
        { // execute a batch node
            size_t arity = my_batch.concat.size();
            Node *node = my_batch.pseudo_node;
            if (node == nullptr)
                node = cg.nodes[my_batch.ids[0]];
            xs.resize(arity);
            // Figure out whether we need to create the inputs
            my_batch.arg_nfxs.resize(arity);
            // timer.start("EXT memtransfer");
            global_timer.start("memtransfer");
            for (auto &block : memory_blocks)
                block.avail = true;
            for (size_t i = 0; i < arity; ++i)
            {
                // 1) the inputs don't need to be concatenated. Just use the tensor
                if (!my_batch.concat[i])
                {
                    my_batch.arg_nfxs[i] = &get_nfx(node->args[i]);
                    // 2) the inputs need to be concatenated
                }
                else
                {
                    // 2.a) the inputs need to be concatenated, but are already in the
                    // right order within a contiguous block of memory.
                    // TODO: make this work completely
                    Tensor *my_xsi = new Tensor;
                    my_xsi->device = node->device;
                    my_xsi->mem_pool = DeviceMempool::FXS;

                    // check contig memory
                    global_timer.start("check contig");
                    auto it = my_batch.ids.begin();
                    auto itend = my_batch.ids.end();
                    VariableIndex aid = cg.nodes[*(it++)]->args[i];
                    float *min_node = get_nfx(aid).v;
                    unsigned int tot_arg = node2size[aid];
                    bool contig = true;
                    while (it != itend && contig)
                    {
                        aid = cg.nodes[*(it++)]->args[i];
                        float *v = get_nfx(aid).v;
                        contig = contig && v == min_node + tot_arg;
                        tot_arg += node2size[aid];
                    }
                    global_timer.stop("check contig");
                    if (contig)
                    { // if contig, use current mem for xs_i
                        // xs[i] = &batched_nfxs[...];
                        my_xsi->v = min_node;
                        my_xsi->d = Dim({tot_arg});
                        my_batch.concat[i] = 2;
                        //   autobatch_garbage[i] = false;
                    }
                    else
                    { // if non-contig, copy xs_i into new mem.
                        // 2.b) the inputs need to be concatenated, and are not contiguous
                        if (profiling_flag > 0)
                        {
                            for (auto id : my_batch.ids)
                            {
                                mem_transfer_edges.insert({cg.nodes[id]->args[i], id});
                            }
                        }
                        global_timer.start("combine tensors");
                        global_timer.cumint("gather_memtransfer", my_batch.ids.size() * node2size[my_batch.ids.front()]);
                        global_timer.cumint("n_combine", 1);
                        combine_tensors(my_batch.ids, i, *my_xsi);
                        global_timer.stop("combine tensors");
                    }
                    my_batch.arg_nfxs[i] = my_xsi;
                }
            }
            global_timer.stop("memtransfer");
            node->autobatch_reshape(cg, my_batch.ids, my_batch.concat, my_batch.arg_nfxs, my_batch.nfx);
            global_timer.start("computation");
            global_timer.lock();
            // vector<string> input_dims;
            // for (auto& x: my_batch.arg_nfxs) {
            //     ostringstream o;
            //     o << x->d;
            //     input_dims.push_back(o.str());
            // }
            // cout << "[forward] out{";
            // for (auto id: my_batch.ids) cout << id << ",";
            // cout << "}=" << node->as_string(input_dims);

            if (node->is_function)
            {
                FunctorNode *functor_node = static_cast<FunctorNode *>(node);
                vector<Tensor *> ys;
                for (int i = 0; i < functor_node->n_output; ++i)
                    ys.push_back(&batches[bid + i + 1].nfx);
                // cout <<"dim={";
                // for (auto y:ys) cout << y->d << ",";
                // cout << "}" << endl;
                functor_node->forward(my_batch.arg_nfxs, ys);
            }
            else
            {
                // cout << "dim=" << my_batch.nfx.d << endl;
                node->forward(my_batch.arg_nfxs, my_batch.nfx);
            }
            global_timer.unlock();
            global_timer.stop("computation");
            // cerr << "batched forward[" << num_batches_evaluated << "] (nodes:"; for(auto id : my_batch.ids) cerr << ' ' << id; cerr << ") == " << print_vec(as_vector(my_batch.nfx)) << endl;
        } // execute a batch node (not a single instance node)
    }

    const Tensor &Executor::get_nfx(VariableIndex i)
    {
        DYNET_ASSERT(!cg.nodes[i]->is_function, "call get_nfx on function node");
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
            t.d = cg.nodes[i]->dim;
            t.mem_pool = bt.mem_pool;
            t.device = bt.device;
        }
        return nfx_cache[i];
    }

    union CopyArgs{
        float *ptr;
        std::size_t n;
    };

    void Executor::combine_tensors(
        const std::vector<VariableIndex> &batch_ids,
        int aid,
        Tensor &tout)
    {
        // global_timer.start("EXT combine tensors");
        //   global_timer.start("EXT combine preparation");
        AlignedMemoryPool *mempool = tout.device->pools[(int)DeviceMempool::FXS];
        // Determine needed memory for tout and get list of nodes corresponding to
        // specified argument.
        unsigned total_dsize = 0;
        vector<VariableIndex> arg_nodes(batch_ids.size());
        for (unsigned i = 0; i < batch_ids.size(); ++i)
        {
            const auto nid = cg.nodes[batch_ids[i]]->args[aid];
            total_dsize += cg.nodes[nid]->dim.size();
            arg_nodes[i] = nid;
        }
        tout.d = Dim({total_dsize});
        // global_timer.stop("EXT combine preparation");

        // allocate memory for tout
        // global_timer.start("EXT combined allocation");
        float *dest = nullptr;
        for (auto &block : memory_blocks)
        {
            if (block.avail && block.sz >= total_dsize)
            {
                block.avail = false;
                dest = block.base;
            }
        }
        if (dest == nullptr)
        {
            dest = static_cast<float *>(mempool->allocate(total_dsize * sizeof(float)));
            memory_blocks.push_back({dest, total_dsize, false});
        }
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
            const size_t sz = cg.nodes[id]->dim.size();

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

} // namespace OoC