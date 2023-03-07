#ifndef DYNET_NODES_TUPLE_H_
#define DYNET_NODES_TUPLE_H_

#include "dynet/dynet.h"
#include "dynet/nodes-def-macros.h"
#include "dynet/ooc-block.h"

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
        inline virtual std::vector<int> autobatch_concat(const ComputationGraph & cg) const override;
        std::string as_string(const std::vector<std::string>& arg_names) const override;
        Dim dim_forward(const std::vector<Dim>& xs) const override; 
        virtual void autobatch_reshape(const ComputationGraph & cg,
                                        const std::vector<VariableIndex> & batch_ids,
                                        const std::vector<int> & concat,
                                        std::vector<const Tensor*>& xs,
                                        Tensor& fx) const override {
            size_t bid = 0;
            for(auto vid : batch_ids)
                bid += cg.nodes[vid]->dim.bd;
            const Node* exemplar = cg.nodes[batch_ids[0]];
            for(size_t i = 0; i < xs.size(); ++i) {
                const_cast<Tensor*>(xs[i])->d = cg.nodes[exemplar->args[i]]->dim;
                if(concat[i])
                const_cast<Tensor*>(xs[i])->d.bd = bid;
            }
        }
        void forward(const std::vector<const Tensor*>& xs, std::vector<Tensor*>& ys) const; 
        void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
        void backward_impl(const std::vector<const Tensor*>& xs,
                const Tensor& fx,
                const Tensor& dEdf,
                unsigned i,
                Tensor& dEdxi) const override;
        Node* clone() override {return new std::remove_reference<decltype(*this)>::type(*this);}

        vector<vector<unsigned> > lookup_indices;
        OoC::Block* block;
        int batch_size = 1;
        int n_output; // number of output tensor
    };

    struct GetNode: public Node {
        explicit GetNode(const initializer_list<VariableIndex>&a, const Dim & dim): 
        Node(a), dim(dim){}
        virtual bool supports_multibatch() const override {return true; }
        virtual int autobatch_sig(const ComputationGraph &cg, SigMap &sm) const override;
        std::string as_string(const std::vector<std::string>& arg_names) const override;
        Dim dim_forward(const std::vector<Dim>& xs) const override {return dim;} 
        virtual void autobatch_reshape(const ComputationGraph & cg,
                                        const std::vector<VariableIndex> & batch_ids,
                                        const std::vector<int> & concat,
                                        std::vector<const Tensor*>& xs,
                                        Tensor& fx) const override {
            size_t bid = 0;
            for(auto vid : batch_ids)
                bid += cg.nodes[vid]->dim.bd;
            const Node* exemplar = cg.nodes[batch_ids[0]];
            fx.d = exemplar->dim; fx.d.bd = bid;
        }
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

        Dim dim;
        int index; // the index of output
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