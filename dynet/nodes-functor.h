#ifndef DYNET_NODES_TUPLE_H_
#define DYNET_NODES_TUPLE_H_

#include "dynet/dynet.h"
#include "dynet/nodes-def-macros.h"
#include "dynet/ooc-computation_graph.h"

using namespace std;

namespace dynet{
    struct FunctorNode: public Node {
        explicit FunctorNode(std::vector<VariableIndex> a, 
            const vector<vector<unsigned> >& lookup_indices, 
            OoC::Block* block): Node(a), lookup_indices(lookup_indices),  block(block){}
        explicit FunctorNode(OoC::Block* block): block(block){}
        virtual bool supports_multibatch() const override { return true; }
        virtual int autobatch_sig(const ComputationGraph &cg, SigMap &sm) const override;
        virtual Node* autobatch_pseudo_node(
            const ComputationGraph& cg,
            const std::vector<VariableIndex>& batch_ids) const;
        virtual std::vector<int> autobatch_concat(const ComputationGraph & cg) const override;
        virtual void autobatch_reshape(const ComputationGraph & cg,
                                        const std::vector<VariableIndex> & batch_ids,
                                        const std::vector<int> & concat,
                                        std::vector<const Tensor*>& xs,
                                        Tensor& fx) const override {
            autobatch_reshape_concatonly(cg, batch_ids, concat, xs, fx);
        }
        std::string as_string(const std::vector<std::string>& arg_names) const override;
        Dim dim_forward(const std::vector<Dim>& xs) const override; 
        void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
        void backward_impl(const std::vector<const Tensor*>& xs,
                const Tensor& fx,
                const Tensor& dEdf,
                unsigned i,
                Tensor& dEdxi) const override;
        Node* clone() override {return new std::remove_reference<decltype(*this)>::type(*this);}

        vector<shared_ptr<ptrdiff_t> > offsets;
        vector<vector<unsigned> > lookup_indices;
        OoC::Block* block;
        int batch_size = 1;
    };

    struct GetNode: public Node {
        explicit GetNode(const initializer_list<VariableIndex>&a, const Dim & dim): 
        Node(a), offset(nullptr),dim(dim){}
        virtual bool supports_multibatch() const override {return true; }
        virtual int autobatch_sig(const ComputationGraph &cg, SigMap &sm) const override;
        std::string as_string(const std::vector<std::string>& arg_names) const override;
        Dim dim_forward(const std::vector<Dim>& xs) const override {return dim;} 
        void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const {
            DYNET_RUNTIME_ERR("call forward_impl() on get node");
        }
        void backward_impl(const std::vector<const Tensor*>& xs,
            const Tensor& fx,
            const Tensor& dEdf,
            unsigned i,
            Tensor& dEdxi) const {
                DYNET_RUNTIME_ERR("call backward_impl() on get node");   
            }
        Node* clone() override {return new std::remove_reference<decltype(*this)>::type(*this);}

        shared_ptr<ptrdiff_t> offset;
        Dim dim;
    };

    struct PlaceHolderNode: public Node {
        explicit PlaceHolderNode(std::string name): Node(), _name(name){}
        virtual bool supports_multibatch() const override { return true; }
        std::string as_string(const std::vector<std::string>& arg_names) const override;
        Dim dim_forward(const std::vector<Dim>& xs) const override;
        void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const {
            DYNET_RUNTIME_ERR("call forward_impl() on get node");
        }
        void backward_impl(const std::vector<const Tensor*>& xs,
            const Tensor& fx,
            const Tensor& dEdf,
            unsigned i,
            Tensor& dEdxi) const {
                DYNET_RUNTIME_ERR("call backward_impl() on get node");   
            }
        Node* clone() override {return new std::remove_reference<decltype(*this)>::type(*this);}
        std::string _name;
    };
} // namespace dynet 

#endif