
#include "block.hpp"

using namespace std;
using namespace dynet;

namespace OoCTest{
    TreennInternal::TreennInternal(unsigned hdim, unsigned wdim){
        WS = {model.add_parameters({hdim, 2 * hdim}, ParameterInitConst(0.1)), // 0: Ui
              model.add_parameters({hdim, 2 * hdim}, ParameterInitConst(0.1)), // 1: Uo
              model.add_parameters({hdim, 2 * hdim}, ParameterInitConst(0.1)), // 2: Uu
              model.add_parameters({hdim, hdim}, ParameterInitConst(0.1)),     // 3: UFS1
              model.add_parameters({hdim, hdim}, ParameterInitConst(0.1)),     // 4: UFS2
              model.add_parameters({hdim}, ParameterInitConst(0.1)),           // 5: Bi
              model.add_parameters({hdim}, ParameterInitConst(0.1)),           // 6: Bo
              model.add_parameters({hdim}, ParameterInitConst(0.1)),           // 7: Bu
              model.add_parameters({hdim}, ParameterInitConst(0.1))};          // 8: Bf
        L = model.add_lookup_parameters(4, {{hdim}});

        Expression bi, bo, bu, bf;
        bi = parameter(block, WS[5]);
        bo = parameter(block, WS[6]);
        bu = parameter(block, WS[7]);
        bf = parameter(block, WS[8]);
        Expression ui, uo, uu, ufs1, ufs2;
        ui = parameter(block, WS[0]);
        uo = parameter(block, WS[1]);
        uu = parameter(block, WS[2]);
        ufs1 = parameter(block, WS[3]);
        ufs2 = parameter(block, WS[4]);
        block.finish_params();
        Expression h1, c1, h2, c2;
        h1 = block.placeholder({{hdim}}, "h1");
        c1 = block.placeholder({{hdim}}, "c1");
        h2 = block.placeholder({{hdim}}, "h2");
        c2 = block.placeholder({{hdim}}, "c2");
        block.finish_input();
        Expression e, i, o, u, f1, f2, h, c;
        e = concatenate({h1, h2});
        i = logistic(matmul(ui, e, false) + bi);
        o = logistic(matmul(uo, e, false) + bo);
        u = tanh(matmul(uu, e, false) + bu);
        f1 = logistic(matmul(ufs1, h1, false) + bf);
        f2 = logistic(matmul(ufs2, h2, false) + bf);
        c = cmult(f2, c2) + cmult(i, u) + cmult(f1, c1);
        h = cmult(o, tanh(c));
        block.output({h, c});
        block.freeze();
        if (verbose)
            cout << "block: " << endl
                    << block.as_string(true) << endl;
    }

    dynet::Expression TreennInternal::build_graph(ComputationGraph& cg,
        int batch_size){
        Expression h1,c1,h2,c2;
        h1 = lookup(cg, L, {(unsigned)0});
        c1 = lookup(cg, L, {1});
        h2 = lookup(cg, L, {2});
        h2 = lookup(cg, L, {3});

        vector<Expression> losses;
        for (int i = 0; i < batch_size; i++){
            Expression hc = block(&cg, {{"h1", h1}, {"c1", c1}, {"h2", h2}, {"c2", c2}}, {});
            losses.push_back(hc[0]);
        }
        return sum(losses);
    }

    TreennLeaf::TreennLeaf(unsigned hdim, unsigned wdim, unsigned vocab_size): vocab_size(vocab_size){
        WS = {model.add_parameters({hdim, wdim}, ParameterInitConst(0.1)),     // 0: Wi
              model.add_parameters({hdim, wdim}, ParameterInitConst(0.1)),     // 1: Wo
              model.add_parameters({hdim, wdim}, ParameterInitConst(0.1)),      // 2: Wu
              model.add_parameters({hdim}, ParameterInitConst(0.1)),           // 3: Bi
              model.add_parameters({hdim}, ParameterInitConst(0.1)),           // 4: Bo
              model.add_parameters({hdim}, ParameterInitConst(0.1))};           // 5: bu
        E = model.add_lookup_parameters(vocab_size, {wdim}, ParameterInitConst(0.1));
        Expression bi, bo, bu, wi, wo, wu;
        bi = parameter(block, WS[3]);
        bo = parameter(block, WS[4]);
        bu = parameter(block, WS[5]);
        wi = parameter(block, WS[0]);
        wo = parameter(block, WS[1]);
        wu = parameter(block, WS[2]);
        block.finish_params();

        block.finish_input();

        Expression emb, i, o, u, h, c;
        emb = block.lookup(E);
        i = logistic(matmul(wi, emb, false) + bi);
        o = logistic(matmul(wo, emb, false) + bo);
        u = tanh(matmul(wu, emb, false) + bu);
        c = cmult(i, u);
        h = cmult(o, tanh(c));
        block.output({h, c});
        block.freeze();
        if (verbose)
            cout << "block: " << endl
                    << block.as_string(true) << endl;
    }

    dynet::Expression TreennLeaf::build_graph(ComputationGraph & cg, int batch_size){
        vector<Expression> losses;
        for (int i = 0; i < batch_size; i++){
            Expression hc = block(&cg, {}, {i % vocab_size});
            losses.push_back(hc[0]);
        }
        return sum(losses);
    }
    
    LSTMCell::LSTMCell(unsigned hdim, unsigned wdim, unsigned vocab_size)
    : vocab_size(vocab_size){
        word_embed = model.add_lookup_parameters(vocab_size, {hdim});

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
                params.push_back(parameter(block, WS[i]));
            block.finish_params();
            Expression h_in = block.placeholder({{hdim}}, "h");
            Expression c_in = block.placeholder({{hdim}}, "c");
            block.finish_input();
            Expression x = block.lookup(word_embed);
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
            block.output({h_out, c_out});
            block.freeze();
            if (verbose)
                cout << block.as_string(true);
        }
    }

    Expression LSTMCell::build_graph(ComputationGraph & cg, 
        int batch_size){
        vector<Expression> losses;
        Expression h = lookup(cg, word_embed, {unsigned(0)});
        Expression c = lookup(cg, word_embed, {1});
        for (int i = 0; i < batch_size; i++){
            Expression hc = block(&cg, {{"h", h}, {"c", c}}, {i % vocab_size});
            losses.push_back(hc[0]);
        }        
        return sum(losses);
    }

    NChildGRU::NChildGRU(unsigned hdim, int n_child): hdim(hdim), n_child(n_child){
        E = model.add_lookup_parameters(n_child, {hdim});

        for (int i = 0; i < n_child; i++){
            for (int j = 0; j < 3; j++) {
                WS.push_back(model.add_parameters({hdim, hdim})); // U
                WS.push_back(model.add_parameters({hdim})); // bias
            }
        }

        vector<Expression>ws;
        for (auto & w: WS) ws.push_back(parameter(block, w));
        block.finish_params();
        vector<Expression> hs;
        for (int i = 0; i < n_child; i++){
            hs.push_back(block.placeholder({hdim}, "h" + to_string(i+1)));
        }
        block.finish_input();
        Expression h;
        for (int i = 0; i < n_child; i++){
            // r, z, c 
            Expression r, z, c;
            r = logistic(matmul(ws[6*i],hs[i],false)+ws[6*i+1]);
            z = logistic(matmul(ws[6*i+2], hs[i], false) + ws[6*i+3]);
            c = tanh(matmul(ws[6*i+4], cmult(r, hs[i])) + ws[6*i+5]);
            if (i==0) 
                h = cmult(z, hs[i]) + cmult(1-z, c);
            else h = h + (cmult(z,hs[i]) + cmult(1-z,c));
        }
        h = h / n_child;
        block.output({h});
        block.freeze();
    }

    Expression NChildGRU::build_graph(dynet::ComputationGraph& cg, int batch_size) {
        vector<Expression> losses;
        unordered_map<std::string, Expression> inputs;
        for (int i = 0; i < n_child; i++)
            inputs["h" + to_string(i+1)] = lookup(cg, E, {unsigned(i)});
        for (int i = 0; i < batch_size; i++){
            Expression h;
            h = block(&cg, inputs, {});
            losses.push_back(h);
        }        
        return sum(losses);
    }

    GRUCell::GRUCell(unsigned hdim): hdim(hdim){
        E = model.add_lookup_parameters(2, {hdim});

        for (int j = 0; j < 3; j++) {
            WS.push_back(model.add_parameters({hdim})); // bias
            WS.push_back(model.add_parameters({hdim, hdim})); // U
            WS.push_back(model.add_parameters({hdim, hdim})); // W
        }

        vector<Expression>ws;
        for (auto & w: WS) ws.push_back(parameter(block, w));
        block.finish_params();
        Expression h_in, x;
        h_in = block.placeholder({hdim}, "h");
        x = block.placeholder({hdim}, "x");
        block.finish_input();
        Expression h;
        Expression r, z, c;
        r = logistic(matmul(ws[1],h,false)+matmul(ws[2],x,false)+ws[0]);
        z = logistic(matmul(ws[4],h,false)+matmul(ws[5],x,false)+ws[3]);
        c = tanh(matmul(ws[7],cmult(r,h_in),false)+matmul(ws[8],x,false)+ws[6]);
        h = cmult(z, h_in) + cmult(1-z, c);
        block.output({h});
        block.freeze();
    }

    Expression GRUCell::build_graph(dynet::ComputationGraph& cg, int batch_size) {
        vector<Expression> losses;
        Expression h_in, x;
        h_in = lookup(cg, E, {(unsigned)0});
        x = lookup(cg, E, {(unsigned)1});
        for (int i = 0; i < batch_size; i++){
            Expression h;
            h = block(&cg, {{"h", h_in}, {"x", x}}, {});
            losses.push_back(h);
        }        
        return sum(losses);
    }
    
    MVCell::MVCell(unsigned hdim):hdim(hdim) {
        WS = {
            model.add_parameters({hdim, 2 * hdim}), // 0: Wv
            model.add_parameters({hdim}),           // 1: Bv
            model.add_parameters({hdim, 2 * hdim}), // 2: WM
            model.add_parameters({hdim}),           // 3: BM
            model.add_parameters({hdim, hdim}),     // 4: M0
            model.add_parameters({hdim})     // 5: V0
        };

        Expression ml, mr, vl, vr;
        vector<Expression> cg_WS;
        for (size_t i = 0; i < WS.size(); ++i)
            cg_WS.push_back(parameter(block, WS[i]));
        block.finish_params();
        ml = block.placeholder({{hdim, hdim}}, "ml");
        mr = block.placeholder({{hdim, hdim}}, "mr");
        vl = block.placeholder({{hdim}}, "vl");
        vr = block.placeholder({{hdim}}, "vr");
        block.finish_input();

        Expression v_comp = concatenate({matmul(ml, vr, false), matmul(mr, vl, false)});
        Expression m_comp = concatenate({ml, mr});

        Expression v = tanh(affine_transform({cg_WS[1], cg_WS[0], v_comp}));
        Expression m = affine_transform({cg_WS[3], cg_WS[2], m_comp});

        block.output({v, m});
        block.freeze();
    }

    Expression MVCell::build_graph(dynet::ComputationGraph& cg, int batch_size) {
        vector<Expression> losses;
        Expression m0, v0;
        m0 = parameter(cg, WS[4]);
        v0 = parameter(cg, WS[5]);
        for (int i = 0; i < batch_size; i++){
            Expression h;
            h = block(&cg, {{"ml", m0}, {"mr", m0}, {"vl", v0}, {"vr", v0}}, {});
            losses.push_back(h[0]);
        }        
        return sum(losses);
    }

} // namesapce OoCTest 