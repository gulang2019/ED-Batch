#include "OoC.h"

#include "dynet/exec.h"
#include "dynet/param-nodes.h"
#include "dynet/globals.h"
#include "dynet/timing.h"
#include "dynet/devices.h"
#include "dynet/nodes-flow.h"

using namespace std;
using namespace OoC;

namespace dynet
{
    struct nodeInfo
    {
        vector<VariableIndex> preds;
        vector<int> succs;
        int hash = 0;
        int type;
        bool visited = false;
        int supernode_id = -1;
    };

    void BatchedExecutionEngine::memory_allocation(BatchInfo &my_batch)
    {
        global_timer.start("memory allocation");
        auto &batch_ids = my_batch.ids;
        auto &nfx = my_batch.nfx;
        assert(batch_ids.size());
        if (batch_ids.size() == 1)
        {
            VariableIndex curr_node = batch_ids[0];
            if (profiling_flag > 1)
                node2mem_pos[batch_ids[0]] = mem_id++;
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
            // assert(node2size[curr_node] == cg.nodes[curr_node]->dim.size());
            nfx.v = static_cast<float *>(
                mempool->allocate(cg.nodes[curr_node]->dim.size() * sizeof(float)));
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
                if (profiling_flag > 1)
                    node2mem_pos[curr_node] = mem_id++;
                node = cg.nodes[curr_node];
                // assert(node2size[curr_node] == cg.nodes[curr_node]->dim.size());
                my_main = cg.nodes[curr_node]->dim.size();
                my_aux = node->aux_storage_size();
                node2offset[curr_node] = tot_main;
                tot_main += my_main;
                node->aux_mem = (void *)tot_aux;
                tot_aux += my_aux;
            }

            // Allocate main/auxiliary memory for the batch
            assert(node != nullptr);
            auto mempool = node->device->pools[(int)DeviceMempool::FXS];
            float *head_main = static_cast<float *>(
                mempool->allocate(tot_main * sizeof(float)));
            if (head_main == nullptr)
            {
                DYNET_RUNTIME_ERR("Ran out of memory when executing batch, allocating FWD memory.");
            }
            // for(auto curr_node : batch_ids) nfxs[curr_node].v = head_main + node2diff[curr_node];
            char *head_aux = nullptr;
            if (tot_aux > 0)
            {
                head_aux = static_cast<char *>(mempool->allocate(tot_aux));
                if (head_aux == nullptr)
                {
                    DYNET_RUNTIME_ERR("Ran out of memory when executing node, allocating FWD memory.");
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
        global_timer.stop("memory allocation");
    }

    bool BatchedExecutionEngine::commit_batch_OoC(int tid)
    {
        start_indices.push_back(num_batch_committed);

        bool hasUpdate = true;
        auto &frontierType = cg.stypes[tid];
        vector<int> snode_batch = move(frontierType.frontiers);
        assert(snode_batch.size());
        sort(snode_batch.begin(), snode_batch.end(), [&](int sid1, int sid2)
             { return memory_affinity[sid1] < memory_affinity[sid2]; });
        int batch_size = snode_batch.size();
        // frontierType.cnt -= batch_size;

        for (auto nid : snode_batch)
        {
            auto &snode = cg.snodes[nid];
            snode.bid = num_batch_committed;
            frontierType.pureNodeCnt -= (snode.dirtyInputCnt == 0);
            int aid = 0;
            for (auto &succ : snode.succs)
            {
                memory_affinity[succ] = memory_affinity_tag + aid * batch_size;
                auto &succNode = cg.snodes[succ];
                if (--succNode.inputCnt == 0)
                    cg.stypes[succNode.type].frontiers.push_back(succ);
                aid++;
            }
            memory_affinity_tag++;
        }
        memory_affinity_tag += batch_size * (cg.snodes[snode_batch.front()].succs.size() - 1);

        Pattern *pattern = frontierType.pattern;
        // execution allocation;
        int bid = num_batch_committed;
        for (auto &ids : pattern->batch_ids)
        {
            auto &batch = batches[bid];
            batch.ids.clear();
            for (auto id : ids)
            {
                for (auto snid : snode_batch)
                {
                    int nid = cg.snodes[snid].min_nid + id;
                    node2batch[nid] = bid;
                    batch.ids.push_back(nid);
                }
            }
            bid++;
        }
        if (profiling_flag > 2)
        {
            fprintf(stdout, "cg.snodes: ");
            for (auto snode : snode_batch)
            {
                fprintf(stdout, "(%d, %d), ", snode, cg.snodes[snode].min_nid);
            }
            fprintf(stdout, "\n");
            fprintf(stdout, "batch.ids(%d, %d): ", (int)pattern->batch_ids.size(), pattern->n_batch);
            for (int bid = num_batch_committed;
                 bid < num_batch_committed + pattern->batch_ids.size(); bid++)
            {
                fprintf(stdout, "(");
                for (auto id : batches[bid].ids)
                {
                    fprintf(stdout, "%d, ", id);
                }
                fprintf(stdout, "), ");
            }
            fprintf(stdout, "\n");
        }
        // memory allocation & execution
        for (auto bid : pattern->mem_allocation_order)
            memory_allocation(batches[bid + num_batch_committed]);

        // for (int bid = num_batch_committed; bid < num_batch_committed + pattern->n_batch; bid++){
        //     execute_batch(batches[bid]);
        // }
        assert(pattern->mem_allocation_order.size() == pattern->n_batch);
        num_batch_committed += pattern->n_batch;
        return hasUpdate;
    }

    void BatchedExecutionEngine::schedule_snode_graph_OoC()
    {
        // (1) prune rule1, rule2 ==> (dfs)
        bool hasUpdate = true;
        int useHeuristic = 0;

        while (hasUpdate)
        {
            while (hasUpdate)
            {
                hasUpdate = false;
                int frontierTypeCnt = 0, frontierTypeIdx;
                for (int idx = 0; idx < (int)cg.stypes.size(); idx++)
                {
                    auto &type = cg.stypes[idx];
                    if (type.frontiers.size())
                    {
                        ++frontierTypeCnt;
                        frontierTypeIdx = idx;
                    }
                }
                if (frontierTypeCnt == 1)
                    hasUpdate = commit_batch_OoC(frontierTypeIdx);

                for (int idx = 0; idx < (int)cg.stypes.size(); idx++)
                {
                    auto &type = cg.stypes[idx];
                    if (type.pureNodeCnt == type.cnt)
                    {
                        while (type.cnt)
                            hasUpdate = commit_batch_OoC(idx);
                    }
                }
            }

            for (int idx = 0; idx < (int)cg.stypes.size(); idx++)
            {
                auto &type = cg.stypes[idx];
                if (type.frontiers.size())
                {
                    ++useHeuristic;
                    hasUpdate = commit_batch_OoC(idx);
                    break;
                }
            }
        }
        if (profiling_flag > 2)
            fprintf(stdout, "[getBatch] use heuristic %d times\n", useHeuristic);
    }

    void BatchedExecutionEngine::execute_batch(BatchInfo &my_batch)
    {
        global_timer.start("execution");
        string current_batch_name;
        Tensor temp_nfx;
        vector<const Tensor *> xs(16), ts(16);
        if (profiling_flag)
        {
            VariableIndex nid = my_batch.ids[0];
            Node *node = cg.nodes[nid];
            current_batch_name = "FWD " + node->as_dummy_string();
            timer.start(current_batch_name);
        }
        if (my_batch.ids.size() == 1)
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
            // timer.start("EXT computation");
            global_timer.start("computation");
            node->forward(xs, my_batch.nfx);
            // timer.stop("EXT computation");
            global_timer.stop("computation");
            ++num_batches_evaluated;
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
                    global_timer.start("check contig");

                    // check contig memory
                    auto it = my_batch.ids.begin();
                    auto itend = my_batch.ids.end();
                    VariableIndex aid = cg.nodes[*(it++)]->args[i];
                    float *min_node =
                        batches[node2batch[aid]].nfx.v + node2offset[aid];
                    // assert(node2size[aid] == cg.nodes[aid]->dim.size());
                    unsigned int tot_arg = cg.nodes[aid]->dim.size();
                    bool contig = true;
                    while (it != itend && contig)
                    {
                        aid = cg.nodes[*(it++)]->args[i];
                        float *v = batches[node2batch[aid]].nfx.v + node2offset[aid];
                        contig = contig && v == min_node + tot_arg;
                        // assert(node2size[aid] == cg.nodes[aid]->dim.size());
                        tot_arg += cg.nodes[aid]->dim.size();
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
                        global_timer.cumint("n_memtransfer", my_batch.ids.size() * cg.nodes[my_batch.ids.front()]->dim.size());
                        global_timer.cumint("n_combine", 1);
                        combine_tensors(my_batch.ids, i, *my_xsi);
                    }
                    my_batch.arg_nfxs[i] = my_xsi;
                }
            }
            global_timer.stop("memtransfer");
            node->autobatch_reshape(cg, my_batch.ids, my_batch.concat, my_batch.arg_nfxs, my_batch.nfx);
            global_timer.start("computation");
            node->forward(my_batch.arg_nfxs, my_batch.nfx);
            global_timer.stop("computation");
            ++num_batches_evaluated;
        } // execute a batch node (not a single instance node)
        if (profiling_flag)
            timer.stop(current_batch_name);
        global_timer.stop("execution");
    }

    void BatchedExecutionEngine::execution(int upto)
    {
        string current_batch_name;
        Tensor temp_nfx;
        vector<const Tensor *> xs(16), ts(16);
        unordered_set<pair<int, int>, hash_pair> mem_transfer_edges;

        for (int si = start_indices.size() - 1; si > 0; --si)
        {
            for (int bid = start_indices[si - 1]; bid < start_indices[si]; ++bid)
            {
                // Read in the stuff for this batch
                auto &my_batch = batches[bid];
                if (profiling_flag)
                {
                    VariableIndex nid = my_batch.ids[0];
                    Node *node = cg.nodes[nid];
                    current_batch_name = "FWD " + node->as_dummy_string();
                    timer.start(current_batch_name);
                }
                if (my_batch.ids.size() == 1)
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
                    // timer.start("EXT computation");
                    global_timer.start("computation");
                    node->forward(xs, my_batch.nfx);
                    // timer.stop("EXT computation");
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
                            float *min_node =
                                batches[node2batch[aid]].nfx.v + node2offset[aid];
                            // assert(node2size[aid] == cg.nodes[aid]->dim.size());
                            unsigned int tot_arg = cg.nodes[aid]->dim.size();
                            bool contig = true;
                            while (it != itend && contig)
                            {
                                aid = cg.nodes[*(it++)]->args[i];
                                float *v = batches[node2batch[aid]].nfx.v + node2offset[aid];
                                contig = contig && v == min_node + tot_arg;
                                // assert(node2size[aid] == cg.nodes[aid]->dim.size());
                                tot_arg += cg.nodes[aid]->dim.size();
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
                                if (profiling_flag > 1)
                                {
                                    for (auto id : my_batch.ids)
                                        mem_transfer_edges.insert({cg.nodes[id]->args[i], id});
                                }
                                global_timer.cumint("n_memtransfer", my_batch.ids.size() * cg.nodes[my_batch.ids.front()]->dim.size());
                                global_timer.cumint("n_combine", 1);
                                combine_tensors(my_batch.ids, i, *my_xsi);
                            }
                            my_batch.arg_nfxs[i] = my_xsi;
                        }
                    }
                    // timer.stop("EXT memtransfer");
                    global_timer.stop("memtransfer");
                    // timer.start("EXT mem reshape");
                    node->autobatch_reshape(cg, my_batch.ids, my_batch.concat, my_batch.arg_nfxs, my_batch.nfx);
                    // timer.stop("EXT mem reshape");
                    // timer.start("EXT computation");
                    global_timer.start("computation");
                    node->forward(my_batch.arg_nfxs, my_batch.nfx);
                    // timer.stop("EXT computation");
                    global_timer.stop("computation");
                    // cerr << "batched forward[" << num_batches_evaluated << "] (nodes:"; for(auto id : my_batch.ids) cerr << ' ' << id; cerr << ") == " << print_vec(as_vector(my_batch.nfx)) << endl;
                } // execute a batch node (not a single instance node)
                ++num_batches_evaluated;
                if (profiling_flag)
                    timer.stop(current_batch_name);
            }
        }
        // timer.stop("EXT execution");
        assert(num_batches_evaluated == num_batch_committed);
        string graphname = "B7_" + to_string(graph_id);
        visualize(upto, "./pics/" + graphname + ".gv", graphname, &mem_transfer_edges);
        graphname = "S" + to_string(graph_id);
        visualize_snode(upto, "./pics/" + graphname + ".gv", graphname, &mem_transfer_edges);
    }

    void BatchedExecutionEngine::forward_OoC(VariableIndex upto)
    {
        if (profiling_flag > 1)
        {
            int sum_cnt = 0;
            for (int j = num_nodes_evaluated; j <= upto; ++j)
            {
                int type = cg.sigmap.sig2type(cg.nodes[j]->autobatch_sig(cg, cg.sigmap));
                if (type == op_type::sum)
                    sum_cnt++;
            }
            fprintf(stdout, "sum: %d\n", sum_cnt);
        }
        // global_timer.clear();
        global_timer.start("total");
        num_batch_committed = num_batches_evaluated;
        const size_t uptop1 = upto + 1;
        nfx_cache.resize(uptop1);
        node2batch.resize(uptop1, -1);
        node2offset.resize(uptop1, 0);
        // node2size.resize(uptop1, 0);
        batches.resize(upto - num_nodes_evaluated + num_batches_evaluated + 1);

        // cg.stypes[cg.snodes[cg.nid2sid[upto]].type].frontiers.push_back(cg.nid2sid[upto]);
        int frontier_type_cnt = 0;
        for (int sid = cg.snodes.size()-1; sid >= 0; sid--){  
            auto & snode = cg.snodes[sid]; 
            if (snode.inputCnt == 0){
                cg.stypes[snode.type].frontiers.push_back(sid);
                frontier_type_cnt += 1;
            }
            for (auto succ: snode.succs) cg.snodes[succ].inputCnt++;   
        }
        fprintf(stdout, "[OoC::forward]: frontier_type_cnt: %d\n", frontier_type_cnt);

        memory_affinity.resize(cg.snodes.size(), 0);

        if (profiling_flag > 1)
        {
            node2mem_pos.resize(uptop1, 0);
            mem_id = 0;
        }

        // Create the necessary info for batching in the future
        // construct_snode_graph_OoC(upto);
        int old_num_nodes_evaluated = num_nodes_evaluated;
        if (cg.schedule_mode == INFERENCE)
            fprintf(stdout, "inference\n");
        else
            fprintf(stdout, "train\n");
        schedule_snode_graph_rl();
        commit_unbatchables(upto);
        start_indices.push_back(num_batch_committed);

        if (profiling_flag > 1)
        {
            vector<bool> visited(uptop1, false);
            for (int bid = old_num_nodes_evaluated; bid != num_batch_committed; bid++)
            {
                assert(batches[bid].nfx.v);
                int sig = cg.nodes[batches[bid].ids.front()]->autobatch_sig(cg, cg.sigmap);
                for (auto id : batches[bid].ids)
                {
                    assert(sig == cg.nodes[id]->autobatch_sig(cg, cg.sigmap));
                    visited[id] = true;
                }
            }
            function<int(int)> get_sid = [&](int bid)
            {
                int ret = 0;
                while (start_indices[ret] <= bid)
                    ret++;
                return ret - 1;
            };
            for (int i = old_num_nodes_evaluated; i <= upto; i++)
            {
                if (!visited[i] || !(node2batch[i] >= 0))
                {
                    fprintf(stderr, "[OoC::error]: nid %d, bid %d, old_num_nodes_evaluated %d, upto %d, num_batch_commited %d\n",
                            i, node2batch[i], old_num_nodes_evaluated, upto, num_batch_committed);
                }
                assert(visited[i]);
                assert(node2batch[i] >= 0);
                assert(node2batch[i] < num_batch_committed);

                const Node *node = cg.nodes[i];
                auto this_sbid = get_sid(node2batch[i]);
                for (auto arg : node->args)
                {
                    auto that_sbid = get_sid(node2batch[arg]);
                    if (this_sbid == that_sbid)
                    {
                        if (node2batch[i] <= node2batch[arg])
                        {
                            int this_type = cg.sigmap.sig2type(cg.nodes[i]->autobatch_sig(cg, cg.sigmap));
                            int that_type = cg.sigmap.sig2type(cg.nodes[arg]->autobatch_sig(cg, cg.sigmap));
                            fprintf(stdout, "num_batch_committed %d\n", num_batch_committed);
                            fprintf(stdout, "node %d %s %d; input %d %s %d\n", i, type2name[this_type].c_str(), node2batch[i], arg, type2name[that_type].c_str(), node2batch[arg]);
                        }
                        assert(node2batch[i] > node2batch[arg]);
                    }
                    else
                    {
                        if (this_sbid > that_sbid)
                        {
                            int this_type = cg.sigmap.sig2type(cg.nodes[i]->autobatch_sig(cg, cg.sigmap));
                            int that_type = cg.sigmap.sig2type(cg.nodes[arg]->autobatch_sig(cg, cg.sigmap));
                            fprintf(stdout, "num_batch_committed %d\n", num_batch_committed);
                            fprintf(stdout, "node %d %s %d; input %d %s %d\n", i, type2name[this_type].c_str(), node2batch[i], arg, type2name[that_type].c_str(), node2batch[arg]);
                            fprintf(stdout, "thisbid: \n");
                            assert(this_sbid >= 0);
                            for (int bid = start_indices[this_sbid]; bid < start_indices[this_sbid + 1]; bid++)
                            {
                                fprintf(stdout, "\tbid %d: ", bid);
                                for (auto id : batches[bid].ids)
                                {
                                    int type = cg.sigmap.sig2type(cg.nodes[id]->autobatch_sig(cg, cg.sigmap));
                                    fprintf(stdout, "(%d, %s), ", id, type2name[type].c_str());
                                }
                                fprintf(stdout, "\n");
                            }
                            fprintf(stdout, "thatbid: \n");
                            assert(that_sbid >= 0);
                            for (int bid = start_indices[that_sbid]; bid < start_indices[that_sbid + 1]; bid++)
                            {
                                fprintf(stdout, "\tbid %d: ", bid);
                                for (auto id : batches[bid].ids)
                                {
                                    int type = cg.nodes[id]->autobatch_sig(cg, cg.sigmap);
                                    fprintf(stdout, "(%d, %s), ", id, type2name[type].c_str());
                                }
                                fprintf(stdout, "\n");
                            }
                        }
                        assert(this_sbid < that_sbid);
                    }
                }
            }
            for (auto &stype : cg.stypes)
            {
                assert(stype.frontiers.size() == 0);
                // assert(stype.cnt == 0);
            }
            fprintf(stdout, "[OoC::check]: [%d, %d], dependency test passed!\n", old_num_nodes_evaluated, upto);
        }
        fprintf(stdout, "OoC: commit %d batch\n", num_batch_committed);
        global_timer.log("n_kernel", num_batch_committed);
        global_timer.cumint("n_kernels", num_batch_committed);
        global_timer.start("execution");
        // fprintf(stdout, "OoC: begin execution\n");
        execution(upto);
        global_timer.stop("execution");
        assert(num_batch_committed == num_batches_evaluated);
        global_timer.stop("total");
        // global_timer.show();
        graph_id++;
        return;
    }

    void BatchedExecutionEngine::visualize_snode(int upto, string filename, std::string graphname, std::unordered_set<std::pair<int, int>, hash_pair> *mem_transfer_edges)
    {
        if (profiling_flag <= 1)
            return;
        ofstream file;
        file.open(filename);
        file << "digraph " << graphname << "{\n";
        function<string(int)> getName = [&](int sid)
        {
            return "S" + to_string(sid) + "_" + to_string(cg.snodes[sid].type) + "_" + to_string(cg.snodes[sid].bid);
        };

        unordered_map<int, string> bid2color;
        int nid = num_nodes_evaluated;
        file << "\tnode [style=filled]\n";
        while (nid <= upto)
        {
            int sid = cg.nid2sid[nid];
            if (sid < 0)
            {
                nid++;
                continue;
            }
            unordered_map<int, bool> preds;
            while (nid <= upto && cg.nid2sid[nid] == sid)
            {
                for (auto arg : cg.nodes[nid]->args)
                {
                    int that_sid = cg.nid2sid[arg];
                    if (that_sid < 0)
                        continue;
                    if (preds.count(that_sid) == 0)
                        preds[that_sid] = false;
                    if (mem_transfer_edges && mem_transfer_edges->count({arg, nid}) != 0)
                    {
                        preds[that_sid] = true;
                    }
                }
                nid++;
            }
            auto this_name = getName(sid);
            for (auto &kv : preds)
            {
                file << "\t" << getName(kv.first) << "->" << this_name;
                if (kv.second)
                    file << "\t[color=\"red\"]";
                file << endl;
            }
            int bid = cg.snodes[sid].bid;
            if (bid2color.count(bid) == 0)
            {
                char tmp[10];
                sprintf(tmp, "#%2x%2x%2x", rand() & 0xff, rand() & 0xff, rand() & 0xff);
                bid2color[bid] = string(tmp);
            }
            file << this_name << "\t[color=\"" << bid2color[bid] << "\"]" << endl;
        }

        file << "}\n";
    }

    void BatchedExecutionEngine::commit_unbatchables(VariableIndex upto)
    {
        start_indices.push_back(num_batch_committed);
        list<int> &ops = cg.unbatchable_ops;
        while (ops.size() && ops.front() <= upto)
        {
            int nid = ops.front();
            Node *node = cg.nodes[nid];
            node2batch[nid] = num_batch_committed;
            auto &batch = batches[num_batch_committed];
            batch.ids.resize(1);
            batch.ids[0] = nid;
            memory_allocation(batch);
            num_batch_committed++;
            ops.pop_front();
        }
    }

    void BatchedExecutionEngine::export_graph(VariableIndex upto, string filename)
    {
        // bid n_input inputs
        if (profiling_flag <= 2)
            return;
        ofstream file;
        file.open(filename);
        file << upto + 1 << " " << num_batches_evaluated << endl;
        for (VariableIndex j = 0; j <= upto; j++)
        {
            auto node = cg.nodes[j];
            file << j << " " << cg.nid2sid[j] << " " << node2batch[j] << " " << node->args.size();
            for (auto arg : node->args)
            {
                file << " " << arg;
            }
            file << endl;
        }
        file.close();
    }

    void BatchedExecutionEngine::export_snode_graph(string filename)
    {
        if (profiling_flag <= 2)
            return;
        ofstream file;
        file.open(filename);
        file << cg.snodes.size() << endl;
        int sid = 0;
        for (auto &snode : cg.snodes)
        {
            file << snode.type << " " << snode.succs.size() << " ";
            for (auto succ : snode.succs)
                file << succ << " ";
            file << endl;
        }
        file.close();
    }

    void BatchedExecutionEngine::schedule_snode_graph_rl()
    {
        if (cg.schedule_mode == TRAIN)
        {
            fprintf(stdout, "[BatchedExecutionEngine::scheduler]: begin training\n");
            scheduler.train(cg.snodes, cg.stypes);
            fprintf(stdout, "[BatchedExecutionEngine::scheduler]: after training\n");
        }
        while (true)
        {
            set<int> state;
            for (int stid = 0; stid < cg.stypes.size(); stid++)
            {
                if (cg.stypes[stid].frontiers.size())
                    state.insert(stid);
            }
            if (!state.size())
                break;
            int act = scheduler.get_action(state);
            commit_batch_OoC(act);
        }
        return;
    }

    void BatchedExecutionEngine::visualize(int upto, string filename, string graphname, unordered_set<pair<int, int>, hash_pair> *mem_transfer_edges)
    {
        if (profiling_flag <= 1)
            return;
        ofstream file;
        file.open(filename);
        file << "digraph " << graphname << " {\n";
        file << "\tnode [style=filled]\n";
        function<string(int)> getName = [&](int nid)
        {
            auto sig = cg.nodes[nid]->autobatch_sig(cg, cg.sigmap);
            string ret = OoC::type2name[cg.sigmap.sig2type(sig)] + "_" + to_string(sig) + "_" + to_string(nid);
            if (autobatch_flag == 7)
                ret += "_" + to_string(node2mem_pos[nid]) + "_" + to_string(cg.nid2sid[nid]);
            return ret;
        };

        for (int j = num_nodes_evaluated; j <= upto; j++)
        {
            const Node *node = cg.nodes[j];
            auto sig = node->autobatch_sig(cg, cg.sigmap);
            if (sig == 0)
                continue;
            string node_str = getName(j);
            for (auto arg : node->args)
            {
                string from_str = getName(arg);
                file << "\t" << from_str << "->" << node_str;
                file << "\t[";
                if (mem_transfer_edges && mem_transfer_edges->count({arg, j}))
                    file << " color=\"red\"";
                file << "]";
                file << endl;
            }
        }

        if (autobatch_flag == 7)
        {
            int nid = num_nodes_evaluated;
            unordered_map<int, string> bid2color;
            while (nid <= upto)
            {
                int sid = cg.nid2sid[nid];
                int bid = cg.snodes[sid].bid;
                if (sid == -1)
                {
                    nid++;
                    continue;
                }
                if (bid2color.count(bid) == 0)
                {
                    char tmp[10];
                    sprintf(tmp, "#%2x%2x%2x", rand() & 0xff, rand() & 0xff, rand() & 0xff);
                    bid2color[bid] = string(tmp);
                }
                while (nid <= upto && cg.nid2sid[nid] == sid)
                {
                    file << "\t" << getName(nid) << "\t[color=\"" << bid2color[bid] << "\"];\n";
                    nid++;
                }
            }
        }

        file << "}\n";
        file.close();
        return;
    }

    int BatchedExecutionEngine::lower_bound()
    {
        int n_type = cg.sigmap.size();
        vector<int> depth(cg.nodes.size());

        int ret = 0;
        for (int j = 0; j < n_type; j++)
        {
            int max_depth = 0;
            for (int i = 0; i < cg.nodes.size(); i++)
            {
                auto this_sig = cg.nodes[i]->autobatch_sig(cg, cg.sigmap);
                depth[i] = (this_sig == j);
                for (auto arg : cg.nodes[i]->args)
                {
                    depth[i] = max(depth[i], (this_sig == j) + depth[arg]);
                }
                max_depth = max(max_depth, depth[i]);
            }
            ret += max_depth;
        }
        return ret;
    }

} // namespace dynet