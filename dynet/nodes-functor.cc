#include "dynet/tensor-eigen.h"
#include "dynet/nodes-functor.h"

#include "dynet/nodes-impl-macros.h"
#include "dynet/functors.h"


using namespace std;

namespace dynet{

#ifndef __CUDACC__

int FunctorNode::autobatch_sig(const ComputationGraph& cg, SigMap &sm) const {
    Sig s(nt::block);
    s.add_int(block->id);
   return sm.get_idx(s);
}

Node* FunctorNode::autobatch_pseudo_node(
    const ComputationGraph& cg, 
    const std::vector<VariableIndex>& batch_ids
) const {
    FunctorNode * new_node = new FunctorNode(block);
    new_node->offsets.resize(batch_ids.size() * offsets.size(), nullptr);
    new_node->lookup_indices.resize(lookup_indices.size());
    new_node->batch_size = batch_ids.size();
    int bid = 0;
    for (auto id: batch_ids){
        auto node = static_cast<FunctorNode*>(cg.nodes[id]);
        assert(node && node->block == block);
        int idx = 0;
        assert(node->lookup_indices.size() == lookup_indices.size());
        for (auto& indices: new_node->lookup_indices){
            indices.insert(indices.end(),node->lookup_indices[idx].begin(),node->lookup_indices[idx].end());
            idx++;
        }
        for (int oid = 0; oid < (int) node->offsets.size(); oid++){
            new_node->offsets[oid * batch_ids.size() + bid] =
                node->offsets[oid]; 
        }
        bid++;
    }
    return new_node;
}

vector<int> FunctorNode::autobatch_concat(const ComputationGraph & cg) const {
    return block->autobatch_concat;
}

string FunctorNode::as_string(const vector<string>& args) const {
    ostringstream s;
    s << "FunctorNode(offsets.size()=" << offsets.size() << ", lookup_indices = {";
    for (auto & lookup_index: lookup_indices) {
        s << "{";
        for (auto idx: lookup_index) s << idx << ",";
        s << "},";
    }
    s << "},inputs= {";
    for(auto arg: args) s << arg << ",";
    s << "},block="<< block->as_string() << ")";
    return s.str();
}

Dim FunctorNode::dim_forward(const vector<Dim>& xs) const {
    assert(block->output_dims.size());
    int example_bd = block->output_dims.front().bd;
    for (auto& dim: block->output_dims){
        assert(dim.bd == example_bd);
    }  
    if (block->output_dims.size() == 1) 
        return block->output_dims.front();
    return Dim({block->one_batch_size}, example_bd);
}

void FunctorNode::forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const {
    global_timer.stop("computation");
    global_timer.stop("execution");
    block->forward(xs, fx, lookup_indices, batch_size);
    global_timer.start("execution");
    if (block->output_dims.size() > 1){
        if(offsets.size() != block->output_dims.size() * batch_size){
            fprintf(stdout, "offsets.size() %ld, block->output_dims.size() %ld, batch_size %d\n",
                offsets.size(), block->output_dims.size(), batch_size);
            throw std::runtime_error("offsets error");
        }
        ptrdiff_t offset = 0;
        for (int oid = 0; oid < (int)block->output_dims.size(); oid++){
            int dim = block->output_dims[oid].size();
            for (int b = 0; b < batch_size; b++){
                assert(offsets[oid*batch_size+b] != nullptr);
                *offsets[oid*batch_size+b] = offset;
                offset += dim;
            }
        }
    }
    global_timer.start("computation");
}

void FunctorNode::backward_impl(const std::vector<const Tensor*>& xs,
                const Tensor& fx,
                const Tensor& dEdf,
                unsigned i,
                Tensor& dEdxi) const {
    DYNET_RUNTIME_ERR("call backward_impl() on FunctorNode");
}

string GetNode::as_string(const std::vector<std::string>& arg_names) const{
    ostringstream s;
    s << "get(dim=" << dim << 
    ",offset="<< ((offset.get() == nullptr)? -1:*offset)
    << ",ptr=" << offset << ")";
    return s.str();
}

int GetNode::autobatch_sig(const ComputationGraph &cg, SigMap &sm) const { 
    // if (_type < 0){
        Sig s(nt::get); 
        s.add_int(cg.nodes[args[0]]->autobatch_sig(cg, sm));
        // _type = sm.get_idx(s);
    // }
    return sm.get_idx(s); 
}


string PlaceHolderNode::as_string(const vector<string>&arg_names) const{
    ostringstream s;
    s << "placeholder(name=" << _name << ");";
    return s.str();
}

Dim PlaceHolderNode::dim_forward(const vector<Dim>& xs) const {
    DYNET_RUNTIME_ERR("call dim_forward() on placeholder node");
    return Dim();
}

#endif 

} // namespace dynet
