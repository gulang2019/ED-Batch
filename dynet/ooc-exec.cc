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
        localTimer.start("memory allocation");
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
            nfx.v = static_cast<float *>(
                mempool->allocate(node2size[curr_node] * sizeof(float)));
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
                my_main = node2size[curr_node];
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
        localTimer.stop("memory allocation");
    }

    bool BatchedExecutionEngine::commit_batch_OoC(int tid)
    {
        bool hasUpdate = true;
        auto &frontierType = stypes[tid];
        vector<int> snode_batch = frontierType.frontiers;
        assert(snode_batch.size());
        sort(snode_batch.begin(), snode_batch.end());
        frontierType.cnt -= snode_batch.size();
        frontierType.frontiers.clear();
        for (auto nid : snode_batch)
        {
            auto &snode = snodes[nid];
            snode.bid = num_batch_committed;
            frontierType.pureNodeCnt -= (snode.dirtyInputCnt == 0);
            for (auto &succ : snode.succs)
            {
                auto &succNode = snodes[succ];
                if (--succNode.inputCnt == 0)
                    stypes[succNode.type].frontiers.push_back(succ);
                if (succNode.type != snode.type && (--succNode.dirtyInputCnt == 0))
                    stypes[succNode.type].pureNodeCnt++;
            }
        }
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
                    int nid = snodes[snid].min_nid + id;
                    node2batch[nid] = bid;
                    assert(nid <= node2size.size());
                    batch.ids.push_back(nid);
                }
            }
            bid++;
        }
        if (profiling_flag > 1)
        {
            fprintf(stdout, "snodes: ");
            for (auto snode : snode_batch)
            {
                fprintf(stdout, "(%d, %d), ", snode, snodes[snode].min_nid);
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
        for (auto bid : pattern->mem_allocation_order){
            memory_allocation(batches[bid + num_batch_committed]);
        }

        // for (int bid = num_batch_committed; bid < num_batch_committed + pattern->n_batch; bid++){
        //     execute_batch(batches[bid]);
        // }

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
                for (int idx = 0; idx < (int)stypes.size(); idx++)
                {
                    auto &type = stypes[idx];
                    if (type.frontiers.size())
                    {
                        ++frontierTypeCnt;
                        frontierTypeIdx = idx;
                    }
                }
                if (frontierTypeCnt == 1)
                    hasUpdate = commit_batch_OoC(frontierTypeIdx);

                for (int idx = 0; idx < (int)stypes.size(); idx++)
                {
                    auto &type = stypes[idx];
                    if (type.pureNodeCnt == type.cnt)
                    {
                        while (type.cnt)
                            hasUpdate = commit_batch_OoC(idx);
                    }
                }
            }

            for (int idx = 0; idx < (int)stypes.size(); idx++)
            {
                auto &type = stypes[idx];
                if (type.frontiers.size())
                {
                    ++useHeuristic;
                    hasUpdate = commit_batch_OoC(idx);
                    break;
                }
            }
        }
        if (profiling_flag > -1)
            fprintf(stdout, "[getBatch] use heuristic %d times\n", useHeuristic);
    }

    void BatchedExecutionEngine::execute_batch(BatchInfo &my_batch)
    {
        localTimer.start("execution");
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
            localTimer.start("computation");
            node->forward(xs, my_batch.nfx);
            // timer.stop("EXT computation");
            localTimer.stop("computation");
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
            localTimer.start("memtransfer");
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
                    localTimer.start("check contig");

                    // check contig memory
                    auto it = my_batch.ids.begin();
                    auto itend = my_batch.ids.end();
                    VariableIndex aid = cg.nodes[*(it++)]->args[i];
                    float *min_node =
                        batches[node2batch[aid]].nfx.v + node2offset[aid];
                    unsigned int tot_arg = node2size[aid];
                    bool contig = true;
                    while (it != itend && contig)
                    {
                        aid = cg.nodes[*(it++)]->args[i];
                        float *v = batches[node2batch[aid]].nfx.v + node2offset[aid];
                        contig = contig && v == min_node + tot_arg;
                        tot_arg += node2size[aid];
                    }
                    localTimer.stop("check contig");
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
                        localTimer.cumint("n_memtransfer", my_batch.ids.size() * node2size[my_batch.ids.front()]);
                        localTimer.cumint("n_combine", 1);
                        combine_tensors(my_batch.ids, i, *my_xsi);
                    }
                    my_batch.arg_nfxs[i] = my_xsi;
                }
            }
            localTimer.stop("memtransfer");
            node->autobatch_reshape(cg, my_batch.ids, my_batch.concat, my_batch.arg_nfxs, my_batch.nfx);
            localTimer.start("computation");
            node->forward(my_batch.arg_nfxs, my_batch.nfx);
            localTimer.stop("computation");
            ++num_batches_evaluated;
        } // execute a batch node (not a single instance node)
        if (profiling_flag)
            timer.stop(current_batch_name);
        localTimer.stop("execution");
    }

    void BatchedExecutionEngine::execution(int upto)
    {
        string current_batch_name;
        Tensor temp_nfx;
        vector<const Tensor *> xs(16), ts(16);
        unordered_set<pair<int, int>, hash_pair> mem_transfer_edges;
        while (num_batches_evaluated < num_batch_committed)
        {
            // Read in the stuff for this batch
            auto &my_batch = batches[num_batches_evaluated];
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
                localTimer.start("computation");
                node->forward(xs, my_batch.nfx);
                // timer.stop("EXT computation");
                localTimer.stop("computation");
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
                localTimer.start("memtransfer");
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
                        localTimer.start("check contig");
                        auto it = my_batch.ids.begin();
                        auto itend = my_batch.ids.end();
                        VariableIndex aid = cg.nodes[*(it++)]->args[i];
                        float *min_node =
                            batches[node2batch[aid]].nfx.v + node2offset[aid];
                        unsigned int tot_arg = node2size[aid];
                        bool contig = true;
                        while (it != itend && contig)
                        {
                            aid = cg.nodes[*(it++)]->args[i];
                            float *v = batches[node2batch[aid]].nfx.v + node2offset[aid];
                            contig = contig && v == min_node + tot_arg;
                            tot_arg += node2size[aid];
                        }
                        localTimer.stop("check contig");
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
                            localTimer.cumint("n_memtransfer", my_batch.ids.size() * node2size[my_batch.ids.front()]);
                            localTimer.cumint("n_combine", 1);
                            combine_tensors(my_batch.ids, i, *my_xsi);
                        }
                        my_batch.arg_nfxs[i] = my_xsi;
                    }
                }
                // timer.stop("EXT memtransfer");
                localTimer.stop("memtransfer");
                // timer.start("EXT mem reshape");
                node->autobatch_reshape(cg, my_batch.ids, my_batch.concat, my_batch.arg_nfxs, my_batch.nfx);
                // timer.stop("EXT mem reshape");
                // timer.start("EXT computation");
                localTimer.start("computation");
                node->forward(my_batch.arg_nfxs, my_batch.nfx);
                // timer.stop("EXT computation");
                localTimer.stop("computation");
                // cerr << "batched forward[" << num_batches_evaluated << "] (nodes:"; for(auto id : my_batch.ids) cerr << ' ' << id; cerr << ") == " << print_vec(as_vector(my_batch.nfx)) << endl;
                ++num_batches_evaluated;
            } // execute a batch node (not a single instance node)
            if (profiling_flag)
                timer.stop(current_batch_name);
        }
        // timer.stop("EXT execution");
        string graphname = "B7_" + to_string(graph_id);
        visualize(upto, "./pics/" + graphname + ".gv", graphname, &mem_transfer_edges);
    }

    void BatchedExecutionEngine::forward_OoC(VariableIndex upto)
    {
        if (profiling_flag > 1)
        {
            for (auto j = num_nodes_evaluated; j <= upto; j++)
            {
                int sig = cg.nodes[j]->autobatch_sig(cg, sigmap);
                int type = sigmap.sig2type(sig);
                fprintf(stdout, "%d: %s", j, type2name[type].c_str());
                if (type == nt::bbmark)
                {
                    auto bbmark = static_cast<BBMark *>(cg.nodes[j]);
                    fprintf(stdout, "%d", bbmark->block_id);
                }
                fprintf(stdout, ", %d ||", sig);
                for (auto arg : cg.nodes[j]->args)
                {
                    fprintf(stdout, "%d, ", arg);
                }
                fprintf(stdout, "\n");
            }
        }
        localTimer.clear();
        localTimer.start("total");
        num_batch_committed = num_batches_evaluated;
        const size_t uptop1 = upto + 1;
        nfx_cache.resize(uptop1);
        node2batch.resize(uptop1, -1);
        node2offset.resize(uptop1, 0);
        node2size.resize(uptop1, 0);
        batches.resize(upto - num_nodes_evaluated + num_batches_evaluated + 1);
        if (profiling_flag > 1)
        {
            node2sid.resize(uptop1, 0);
            node2mem_pos.resize(uptop1, 0);
            mem_id = 0;
        }

        // Create the necessary info for batching in the future
        // construct_snode_graph_OoC(upto);
        int old_num_nodes_evaluated = num_nodes_evaluated;
        localTimer.start("scheduling");
        fprintf(stdout, "begin construction...\n");
        construct_snode_graph_from_bb_OoC(upto);
        if (profiling_flag > 1)
            visualize_trie();
        localTimer.stop("scheduling");
        fprintf(stdout, "begin scheduling: ");
        if (schedule_mode == INFERENCE)
            fprintf(stdout, "inference\n");
        else
            fprintf(stdout, "train\n");
        schedule_snode_graph_rl();
        if (profiling_flag > 1)
        {
            vector<bool> visited(uptop1, false);
            for (int bid = old_num_nodes_evaluated; bid != num_batch_committed; bid++)
            {
                int sig = cg.nodes[batches[bid].ids.front()]->autobatch_sig(cg, sigmap);
                for (auto id : batches[bid].ids)
                {
                    assert(sig == cg.nodes[id]->autobatch_sig(cg, sigmap));
                    visited[id] = true;
                }
            }
            for (int i = old_num_nodes_evaluated; i <= upto; i++)
            {
                DYNET_ASSERT(visited[i], "scheduling incomplete");
                DYNET_ASSERT(node2batch[i] >= 0, "scheduling incomplete");

                const Node *node = cg.nodes[i];
                for (auto arg : node->args)
                {
                    assert(node2batch[i] > node2batch[arg]);
                }
            }
            fprintf(stdout, "dependency test passed!");
        }
        fprintf(stdout, "OoC: commit %d batch\n", num_batch_committed);
        localTimer.cumint("n_kernels", num_batch_committed);
        localTimer.start("execution");
        // fprintf(stdout, "OoC: begin execution\n");
        execution(upto);
        localTimer.stop("execution");
        assert(num_batch_committed == num_batches_evaluated);
        localTimer.stop("total");
        localTimer.show();
        if (store_file.size())
        {
            localTimer.save(store_file);
        }
        localTimer.save("./timing");
        string graphname = "S" + to_string(graph_id);
        visualize_snode("./pics/" + graphname + ".gv", graphname);
        export_snode_graph("./graph/" + graphname + ".txt");
        graphname = "G" + to_string(graph_id);
        export_graph(upto, "./graph/" + graphname + ".txt");
        for (auto &stype : stypes)
        {
            assert(stype.cnt == 0 && stype.frontiers.size() == 0);
        }
        graph_id++;
        return;
    }

    void BatchedExecutionEngine::visualize_snode(std::string filename, std::string graphname)
    {
        if (profiling_flag <= 1)
            return;
        ofstream file;
        file.open(filename);
        file << "digraph " << graphname << "{\n";
        function<string(int)> getName = [&](int sid)
        {
            return "S" + to_string(sid) + "_" + to_string(snodes[sid].type) + "_" + to_string(snodes[sid].bid);
        };
        int sid = 0;
        unordered_map<int, vector<int>> bid2color;
        for (auto &snode : snodes)
        {
            auto this_name = getName(sid);
            for (auto to : snode.succs)
                file << "\t" << this_name << "->" << getName(to) << endl;
            if (bid2color.count(snode.bid) == 0)
            {
                bid2color[snode.bid] = {rand() & 0xff, rand() & 0xff, rand() & 0xff};
            }
            char tmp[10];
            sprintf(tmp, "#%2x%2x%2x", bid2color[snode.bid][0], bid2color[snode.bid][1], bid2color[snode.bid][2]);
            file << "\t" << this_name << "\t[color=\"" << tmp << "\"]\n";
            sid++;
        }

        file << "}\n";
    }

    void BatchedExecutionEngine::construct_snode_graph_from_bb_OoC(VariableIndex upto)
    {
        int sig, type;
        snodes.clear();

        vector<int> nid2sid(upto + 1, -1);

        while (num_nodes_evaluated <= upto)
        {
            Node *node = cg.nodes[num_nodes_evaluated];
            sig = node->autobatch_sig(cg, sigmap);
            if (sig)
                break;
            assert(num_nodes_evaluated < node2size.size());
            node2size[num_nodes_evaluated] = node->dim.size();
            node2batch[num_nodes_evaluated] = num_batch_committed;
            auto &batch = batches[num_batch_committed];
            batch.ids.resize(1);
            batch.ids[0] = num_nodes_evaluated;
            memory_allocation(batch);
            num_nodes_evaluated++, num_batch_committed++;
        }

        VariableIndex j = num_nodes_evaluated;
        int n_new_stype_node = 0;
        int n_old_stype = stypes.size();
        while (j <= upto)
        {
            // fprintf(stdout, "construction of snode graph %d\n", j);
            auto node = cg.nodes[j];
            type = sigmap.sig2type(node->autobatch_sig(cg, sigmap));
            // assert(type == nt::bbmark);
            int block_id = static_cast<BBMark *>(node)->block_id;
            j++;

            int sig = cg.nodes[j]->autobatch_sig(cg, sigmap);
            if (sig == 0)
            {
                while (j <= upto)
                { // unbatchable
                    Node *node = cg.nodes[j];
                    sig = node->autobatch_sig(cg, sigmap);
                    if (sig)
                        break;
                    // assert(j < node2size.size());
                    node2size[j] = node->dim.size();
                    node2batch[j] = num_batch_committed;
                    auto &batch = batches[num_batch_committed];
                    batch.ids.resize(1);
                    batch.ids[0] = j;
                    memory_allocation(batch);
                    j++, num_batch_committed++;
                }
                continue;
            }

            if (pattern_cache.patterns.count(block_id) == 0)
            {
                vector<vector<int>> node2args;
                vector<int> node2type;
                for (int nid = j; nid <= upto; nid++)
                {
                    auto node = cg.nodes[nid];
                    int sig = node->autobatch_sig(cg, sigmap);
                    auto type = sigmap.sig2type(sig);
                    if (type == nt::bbmark)
                        break;
                    node2args.push_back({});
                    for (auto arg : node->args)
                        node2args.back().push_back(arg - j);
                    node2type.push_back(sig);
                }
                fprintf(stdout, "add pattern %d, %ld\n", block_id, node2args.size());
                pattern_cache.add_pattern(block_id, node2args, node2type);
                fprintf(stdout, "add pattern finished!\n");
                if (profiling_flag > 1)
                {
                    fprintf(stdout, "add pattern %d: \n", block_id);
                    pattern_cache.patterns[block_id]->show();
                }
            }

            auto &pattern = pattern_cache.patterns[block_id];

            unordered_set<int> preds;
            int this_sid = snodes.size();
            snodes.push_back({});
            auto &snode = snodes.back();
            snode.min_nid = j;

            int nid = j;
            Trie *curr = &head;
            while (nid <= upto){
                int sig = cg.nodes[nid]->autobatch_sig(cg, sigmap);
                if (sigmap.sig2type(sig) == nt::bbmark)
                    break;
                nid2sid[nid] = this_sid;
                Node* node = cg.nodes[nid];
                node2size[nid] = node->dim.size();
                for (auto arg: node->args){
                    auto that_sid = nid2sid[arg];
                    if (arg < j && that_sid >= 0){
                        preds.insert(nid2sid[arg]);
                    }
                }
                if (curr->next.count(sig) == 0)
                    curr->next[sig] = new OoC::Trie();
                curr = curr->next[sig];
                nid++;
            }
            if (!curr->isLeaf)
            {
                curr->isLeaf = true;
                curr->stid = stypes.size();
                curr->bbid = block_id;
                curr->gid = graph_id;
                // fprintf(stdout, "stypes.size() %ld\n",stypes.size());
                stypes.push_back({});
                stypes.back().pattern = pattern.get();
            }
            int stid = curr->stid;
            if (stid >= n_old_stype)
                n_new_stype_node += pattern->nop;

            snode.type = stid;
            snode.inputCnt = preds.size();
            
            for (auto pred: preds){
                snodes[pred].succs.push_back(this_sid);
            }

            auto &stype = stypes[stid];
            stype.cnt += 1;
            if (preds.size() == 0)
                stype.frontiers.push_back(this_sid);

            j += pattern->nop;
        }

        if (n_new_stype_node <= 1)
            schedule_mode = INFERENCE;
        else
            schedule_mode = TRAIN;

        fprintf(stdout, "OoC: %ld snode types\n", stypes.size());
    }

    void BatchedExecutionEngine::export_graph(VariableIndex upto, string filename)
    {
        // bid n_input inputs
        if (profiling_flag <= 1)
            return;
        ofstream file;
        file.open(filename);
        file << upto + 1 << " " << num_batches_evaluated << endl;
        for (VariableIndex j = 0; j <= upto; j++)
        {
            auto node = cg.nodes[j];
            file << j << " " << node2sid[j] << " " << node2batch[j] << " " << node->args.size();
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
        if (profiling_flag <= 1)
            return;
        ofstream file;
        file.open(filename);
        file << snodes.size() << endl;
        int sid = 0;
        for (auto &snode : snodes)
        {
            file << snode.type << " " << snode.succs.size() << " ";
            for (auto succ : snode.succs)
                file << succ << " ";
            file << endl;
        }
        file.close();
    }

    void BatchedExecutionEngine::visualize_trie()
    {
        vector<int> sigs;
        FILE *fp;
        fp = fopen("./graph/trie.txt", "w+");
        assert(fp);
        function<void(Trie *)> printer = [&](Trie *node)
        {
            if (node->isLeaf)
            {
                fprintf(fp, "stid %d, bbid %d, gid %d: ", node->stid, node->bbid, node->gid);
                for (auto sig : sigs)
                {
                    fprintf(fp, "%s%d, ", type2name[sigmap.sig2type(sig)].c_str(), sig);
                }
                fprintf(fp, "\n");
            }
            for (auto kv : node->next)
            {
                sigs.push_back(kv.first);
                printer(kv.second);
                sigs.pop_back();
            }
        };
        printer(&head);
        fclose(fp);
    }

    void BatchedExecutionEngine::schedule_snode_graph_rl()
    {
        if (schedule_mode == TRAIN)
        {
            fprintf(stdout, "[BatchedExecutionEngine::scheduler]: begin trainning\n");
            scheduler.train(snodes, stypes);
            fprintf(stdout, "[BatchedExecutionEngine::scheduler]: after trainning\n");
        }
        while (true)
        {
            set<int> state;
            for (int stid = 0; stid < stypes.size(); stid++)
            {
                if (stypes[stid].frontiers.size())
                    state.insert(stid);
            }
            if (!state.size())
                break;
            int act = scheduler.get_action(state);
            commit_batch_OoC(act);
        }
        return;
    }

} // namespace dynet