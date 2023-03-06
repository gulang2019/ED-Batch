#include <vector>
#include <stdexcept>
#include <fstream>
#include <chrono>
#include <dynet/training.h>
#include <dynet/expr.h>
#include <dynet/dict.h>
#include <string>
#include <dynet/param-init.h>
#include <dynet/ooc-computation_graph.h>

#include "lstm.hpp"

using namespace std;
using namespace std::chrono;
using namespace dynet;

namespace OoCTest
{
    NdLSTMBuilder::NdLSTMBuilder(ParameterCollection &model, unsigned wdim, unsigned hdim, unsigned vocab_size, int ndim) 
    : lstm_cell("lstm_cell"), vocab_size(vocab_size), ndim(ndim)
    {
        word_embed = model.get_lookup_parameters(vocab_size, {hdim}, "LuT");
        H0 = model.get_lookup_parameters(vocab_size, {hdim}, "H0");
        C0 = model.get_lookup_parameters(vocab_size, {hdim}, "C0");
        WH = model.add_parameters({hdim, ndim * hdim});
        BH = model.add_parameters({hdim});
        WC = model.add_parameters({hdim, ndim * hdim});
        BC = model.add_parameters({hdim});

        WS = {
            model.add_parameters({hdim, wdim}), // 0: Uf
            model.add_parameters({hdim, wdim}), // 1: Uc
            model.add_parameters({hdim, wdim}), // 2: Ui
            model.add_parameters({hdim, wdim}), // 3: Uo
            model.add_parameters({hdim, wdim}), // 4: Wf
            model.add_parameters({hdim, wdim}), // 5: Wc
            model.add_parameters({hdim, hdim}), // 6: Wi
            model.add_parameters({hdim, hdim}), // 7: Wo
            model.add_parameters({hdim}),       // 8: Bf
            model.add_parameters({hdim}),       // 9: Bc
            model.add_parameters({hdim}),       // 10: Bi
            model.add_parameters({hdim})        // 11: Bo
        };

        {
            vector<Expression> params;
            for (int i = 0; i < 12; i++)
                params.push_back(parameter(lstm_cell, WS[i]));
            Expression wh = parameter(lstm_cell, WH);
            Expression wc = parameter(lstm_cell, WC);
            Expression bh = parameter(lstm_cell, BH);
            Expression bc = parameter(lstm_cell, BC);
            lstm_cell.finish_params();
            Expression x;
            vector<Expression> hs, cs;
            for (int i = 0; i < ndim; ++i){
                hs.push_back(lstm_cell.placeholder({{hdim}}, "h" + to_string(i)));
                cs.push_back(lstm_cell.placeholder({{hdim}}, "c" + to_string(i)));
            }
            lstm_cell.finish_input();
            x = lstm_cell.lookup(word_embed);
            Expression h_in, c_in;
            h_in = tanh(matmul(wh, concatenate(hs), false) + bh);
            c_in = tanh(matmul(wc, concatenate(cs), false) + bc);
            Expression f, c_, i, o;
            // f = logistic(affine_transform({params[8], params[0], x, params[4], h_in}));
            f = logistic(params[8] + matmul(params[0], x, false) + matmul(params[4], h_in, false));
            // c_ = tanh(affine_transform({params[9], params[1], x, params[5], h_in}));
            c_ = tanh(params[9] + matmul(params[1], x, false) + matmul(params[5], h_in, false));
            // i = logistic(affine_transform({params[10], params[2], x, params[6], h_in}));
            i = logistic(params[10] + matmul(params[2], x, false) + matmul(params[6], h_in, false));
            // o = logistic(affine_transform({params[11], params[3], x, params[7], h_in}));
            o = logistic(params[11] + matmul(params[3], x, false) + matmul(params[7], h_in, false));

            Expression h_out, c_out;
            c_out = cmult(i, c_) + cmult(f, identity(c_in));
            h_out = cmult(o, tanh(c_out));
            lstm_cell.output({h_out, c_out});
            lstm_cell.freeze();
        }
    }

    pair<Expression, Expression> NdLSTMBuilder::build(const std::vector<int> & instance){
        if (visited.count(instance)) 
            return visited[instance];
        
        bool at_boundary = false;
        for (auto ax: instance) at_boundary |= (ax == 0);
        if (at_boundary) {
            int idx = 0;
            assert(instance.size() == dims.size());
            for (int i = 0; i < instance.size(); i++){
                idx = idx * dims[i] + instance[i];
            }
            idx = std::abs(idx) % vocab_size; // hack here 
            Expression h0 = lookup(*cg, H0, idx);
            Expression c0 = lookup(*cg, C0, idx);
            visited[instance] = {h0, c0};
            return {h0, c0};
        }

        int idx = 0;
        for (auto & i: instance) idx += i;
        
        Expression h_out, c_out;
        if (dynet::blocked){
            unordered_map<std::string, Expression> input;
            for (int i = 0; i < instance.size(); ++i){
                vector<int> child = instance;
                child[i] -= 1;
                auto hc = build(child);
                input["h" + to_string(i)] = hc.first;
                input["c" + to_string(i)] = hc.second;
            }
            Expression hc = lstm_cell(cg, input, {(unsigned)(idx % vocab_size)});
            h_out = hc[0], c_out = hc[1];
        }
        else {
            Expression x;
            vector<Expression> hs, cs;
            for (int i = 0; i < instance.size(); ++i){
                vector<int> child = instance;
                child[i] -= 1;
                auto hc = build(child);
                hs.push_back(hc.first);
                cs.push_back(hc.second);
            }
            x = lookup(*cg, word_embed, (unsigned)(idx % vocab_size));
            Expression h_tmp = concatenate(hs), c_tmp = concatenate(cs); 
            Expression h_in, c_in;
            h_in = tanh(matmul(wh, h_tmp, false) + bh);
            c_in = tanh(matmul(wc, c_tmp, false) + bc);
            Expression f, c_, i, o;
            // f = logistic(affine_transform({ws[8], ws[0], x, ws[4], h_in}));
            f = logistic(ws[8] + matmul(ws[0], x, false) + matmul(ws[4], h_in, false));
            // c_ = tanh(affine_transform({ws[9], ws[1], x, ws[5], h_in}));
            c_ = tanh(ws[9] + matmul(ws[1], x, false) + matmul(ws[5], h_in, false));
            // i = logistic(affine_transform({ws[10], ws[2], x, ws[6], h_in}));
            i = logistic(ws[10] + matmul(ws[2], x, false) + matmul(ws[6], h_in, false));
            // o = logistic(affine_transform({ws[11], ws[3], x, ws[7], h_in}));
            o = logistic(ws[11] + matmul(ws[3], x, false) + matmul(ws[7], h_in, false));

            c_out = cmult(i, c_) + cmult(f, identity(c_in));
            h_out = cmult(o, tanh(c_out));
        }

        visited[instance] = {h_out, c_out};
        return {h_out, c_out};
    }

    void NdLSTMBuilder::start_graph(dynet::ComputationGraph& _cg) {
        cg = & _cg;
        if (!dynet::blocked) {
            ws.resize(WS.size());
            for (int i = 0; i < (int)WS.size(); ++i) {
                ws[i] = parameter(_cg, WS[i]);
            }
            wh = parameter(_cg, WH);
            wc = parameter(_cg, WC);
            bh = parameter(_cg, BH);
            bc = parameter(_cg, BC);
        }
    }

    Expression NdLSTMBuilder::build_graph(const std::vector<int>& _dims) {
        visited.clear();
        dims = _dims;
        assert(dims.size() == ndim);
        return build(dims).first;
    }

    NdLSTM::NdLSTM(
        dynet::ParameterCollection& model, 
        unsigned wembed_size, 
        unsigned hidden_size,
        unsigned vocab_size, 
        std::initializer_list<std::pair<int, int>> dims): model(model)
    {
        for (int i = 0; i < 100; i++)
        {
            vector<int> instance;
            for (auto &range : dims)
            {
                instance.push_back(random() % (range.second - range.first) + range.first);
            }
            data.push_back(move(instance));
        }
        builder = new NdLSTMBuilder(model, wembed_size, hidden_size, vocab_size, dims.size());
    }

    Expression NdLSTM::build_graph(ComputationGraph &cg, int batch_size)
    {
        vector<Expression> losses;
        builder->start_graph(cg);
        for (int i = 0; i < batch_size; i++)
            losses.push_back(builder->build_graph(data[(data_idx + i) % data.size()]));
        data_idx = (data_idx + batch_size) % data.size();
        return pickneglogsoftmax(sum(losses, true), 1);
    }
} // namespace OoCTest