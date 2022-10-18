#include "dynet/tensor-eigen.h"
#include "dynet/nodes-ooc-select.h"
#include "dynet/nodes-impl-macros.h"


using namespace std;

#ifdef HAVE_CUDA
#include "dynet/gpu-ops.h"
#endif

namespace dynet {
#ifndef __CUDACC__

int SelectNode::autobatch_sig(const ComputationGraph & cg, SigMap &sm) const {
  Sig s(nt::select);
  s.add_int((size_t)params.p.get());
  return sm.get_idx(s);
}

string SelectNode::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << "select(|x|=" << params.get_storage().values.size() << " --> " << dim << ") @ " << &params.get_storage();
  return s.str();
}

Dim SelectNode::dim_forward(const vector<Dim>& xs) const {
  return dim;
}

#endif

template<class MyDevice>
void SelectNode::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
    DYNET_ASSERT(xs.size() == 1, "Failed dimension check in FUNCNAME");
    DYNET_ASSERT(xs[0]->d.nd == 1, "Failed dimension check in FUNCNAME");
    vector<unsigned> pindices(xs[0]->d.size());
    if (xs[0]->device->type == DeviceType::CPU) {
      memcpy(&pindices[0], xs[0]->v, sizeof(unsigned) * pindices.size());
  #if HAVE_CUDA
    } else if (xs[0]->device->type == DeviceType::GPU) {
      CUDA_CHECK(cudaSetDevice(((Device_GPU*)xs[0]->device)->cuda_device_id));
      CUDA_CHECK(cudaMemcpy(&pindices[0], xs[0]->v, sizeof(unsigned) * pindices.size(), cudaMemcpyDeviceToHost));
  #endif
    }

    DYNET_ARG_CHECK(fx.d.batch_elems() == pindices.size(),
                            "In SelectNode, in index vector size (" << pindices.size() << ") "
                            "doesn't match batch size in expressions (" << fx.d.batch_elems() << ")");
#ifdef __CUDACC__
    aux_mem = device->pools[(int)DeviceMempool::FXS]->allocate(sizeof(unsigned) * pindices.size());
    if (!aux_mem)
      DYNET_RUNTIME_ERR("Ran out of memory when allocating for SelectNode");
    CUDA_CHECK(cudaMemcpyAsync((unsigned*)aux_mem, &pindices[0], fx.d.bd * sizeof(unsigned), cudaMemcpyHostToDevice));
    dynet::gpu::sparse_to_dense_block_assign_and_multiply(fx.d.bd, (unsigned*)aux_mem, fx.d.batch_size(), params.current_weight_decay(), params.get_storage().all_values.v, fx.v);
#else
    for (unsigned b = 0; b < pindices.size(); ++b) {
      unsigned i = pindices[b];
      DYNET_ARG_CHECK(i < params.get_storage().values.size(),
                              "Out-of-bounds attempt to access index " << i << " for LookupParameter of size " << params.get_storage().values.size());
      tbvec(fx).chip<1>(b).device(*dev.edevice) = tvec(params.get_storage().values[i]) * params.current_weight_decay();
    }
#endif
}

template<class MyDevice>
void SelectNode::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_RUNTIME_ERR("called backward() on arity 0 node: i = " << i);
}
DYNET_NODE_INST_DEV_IMPL(SelectNode)

} //namespace dynet