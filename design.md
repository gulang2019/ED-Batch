memory_allocation
    cg.nodes[curr_node]->dim.size()
    node->aux_storage_size()
    node->aux_mem = mempool->allocate(aux_size)
    node->device->pools[(int)DeviceMempool::FXS]
    node->autobatch_concat(cg)
    node->autobatch_pseudo_node(cg, batch_ids)

execute_batch 
    Node* node = cg.nodes[nid]
    node->arity()
    node->args
    xs[ai] = &get_nfx(arg);
    VariableIndex aid = cg.nodes[*(it++)]->args[i];
    node->autobatch_reshape(cg, my_batch.ids, my_batch.concat, my_batch.arg_nfxs, my_batch.nfx);
    node->forward(my_batch.arg_nfxs, my_batch.nfx);

class Node{
    dim_forward [shared]
    as_string   [shared]
    as_dummy_string  [shared]
    aux_storage_size [shared]
    forward_impl  [shared]
    backward_impl [shared]
    supports_multibatch  [shared]
    supports_multidevice [shared]
    forward [shared]
    backward [shared]
    autobatch_sig    [shared]
    autobatch_concat [shared]
    autobatch_pseudo_node [shared]
    autobatch_reshape[shared]
    autobatch_reshape_concatonly[shared]
    arity [shared]
    set_cg [shared]
    args [private]
    dim  [shared]
    cg_  [shared]
    device  [shared]
    aux_mem [private]
}

'''
cg.nodes[i]
'''
node->args 是一致的
'''
node->dim.size()
'''
构建图用supernode, 实际的图的构建并行来做;
nodes 的fu
node的复制机制
