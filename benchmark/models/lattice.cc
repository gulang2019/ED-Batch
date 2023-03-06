#include "lattice.hpp"
// #include "model.hpp"

using namespace dynet;
using namespace std;

namespace OoCTest
{

    unsigned c_vocab_size;
    unsigned w_vocab_size;
    unsigned type_size;

    std::vector<DAG> read_dataset(string filename)
    {
        ifstream file(filename);
        unsigned n_data, n_type = 0;
        c_vocab_size = w_vocab_size = 0;
        file >> n_data;
        std::vector<DAG> dataset;
        for (int data_id = 0; data_id < n_data; data_id++)
        {
            DAG data;
            int length;
            file >> length;
            for (int i = 0; i < length; i++)
            {
                data.nodes.push_back({});
                file >> data.nodes.back().tag;
                data.nodes.back().tag %= MAX_VOCAB_SIZE;
                n_type = max((unsigned)data.nodes.back().tag + 1, n_type);
            }
            for (int i = 0; i < length; i++)
            {
                int n_arc;
                auto &node = data.nodes[i];
                file >> n_arc;
                for (int j = 0; j < n_arc; j++)
                {
                    Arc arc;
                    file >> arc.from >> arc.id;
                    arc.id %= MAX_VOCAB_SIZE; // so many words ... 
                    arc.trivial = arc.from == (i - 1);
                    if (arc.trivial)
                        c_vocab_size = max(c_vocab_size, (unsigned)arc.id + 1);
                    else
                        w_vocab_size = max(w_vocab_size, (unsigned)arc.id + 1);
                    node.arcs.emplace_back(arc);
                }
            }
            dataset.emplace_back(data);
        }
        type_size = n_type;

        return move(dataset);
    }

    LatticeLSTMBuilder::LatticeLSTMBuilder(ParameterCollection &model, unsigned hdim)
    {
        unsigned wdim = hdim;
        cerr << "c_vocab_size" << c_vocab_size << endl;
        cerr << "w_vocab_size" << w_vocab_size << endl;
        cerr << "type_size" << type_size << endl;
        E_c = model.get_lookup_parameters(c_vocab_size, {wdim}, "CharLuT");
        E_w = model.get_lookup_parameters(w_vocab_size, {wdim}, "WordLuT");
        H0 = model.add_parameters({hdim});
        C0 = model.add_parameters({hdim});
        WO = model.add_parameters({type_size, hdim});

        WS = {model.add_parameters({hdim, wdim}), // 0: Uf
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
              model.add_parameters({hdim}),       // 11: Bo
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
              model.add_parameters({hdim})};      // 11: Bo
        
        // pre-define the basic blocks
        for (bool trivial : {false, true})
        {
            blocks.push_back(make_unique<OoC::Block>("lattice_lstm" + to_string(blocks.size())));
            auto &block = *blocks.back();
            int base = trivial ? 0 : 12;
            std::vector<Expression> params;
            for (int i = 0; i < 12; i++)
                params.push_back(parameter(block, WS[base + i]));
            block.finish_params();
            Expression h_in, c_in, x;
            h_in = block.placeholder({{hdim}}, "h_in");
            c_in = block.placeholder({{hdim}}, "c_in");
            block.finish_input();
            if (trivial)
                x = block.lookup(E_c);
            else
                x = block.lookup(E_w);
            Expression f, c_, i, o;
            f = logistic(affine_transform({params[8], params[0], x, params[4], h_in}));
            c_ = tanh(affine_transform({params[9], params[1], x, params[5], h_in}));
            i = logistic(affine_transform({params[10], params[2], x, params[6], h_in}));
            o = logistic(affine_transform({params[11], params[3], x, params[7], h_in}));

            Expression h_out, c_out;
            c_out = cmult(i, c_) + cmult(f, c_in);
            h_out = cmult(o, tanh(c_out));
            block.output({h_out, c_out});
            block.freeze();
        }
    }

    void LatticeLSTMBuilder::build_graph(DAG & g, std::vector<Expression> & losses)
    {
        for (auto &node : g.nodes)
        {
            std::vector<Expression> hs;
            std::vector<Expression> cs;

            for (auto &arc : node.arcs)
            {
                assert(g.nodes[arc.from].h.pg!=nullptr);
                assert(g.nodes[arc.from].c.pg!=nullptr);
                if (dynet::blocked)
                {
                    Expression o = blocks[arc.trivial]->operator()(_cg, {{"h_in", g.nodes[arc.from].h}, {"c_in", g.nodes[arc.from].c}},
                                                       {(unsigned)arc.id});
                    hs.push_back(o[0]);
                    cs.push_back(o[1]);
                }
                else
                {
                    // lstm: arc.h, arc.c = LSTM(nodes[arc.from.h], nodes[arc.from.c])
                    int base = arc.trivial ? 0 : 12;
                    Expression h_in, c_in, x;
                    h_in = g.nodes[arc.from].h;
                    c_in = g.nodes[arc.from].c;
                    if (arc.trivial)
                        x = lookup(*_cg, E_c, arc.id);
                    else
                        x = lookup(*_cg, E_w, arc.id);
                    Expression f, c_, i, o;
                    f = logistic(affine_transform({cg_WS[base + 8], cg_WS[base + 0], x, cg_WS[base + 4], h_in}));
                    c_ = tanh(affine_transform({cg_WS[base + 9], cg_WS[base + 1], x, cg_WS[base + 5], h_in}));
                    i = logistic(affine_transform({cg_WS[base + 10], cg_WS[base + 2], x, cg_WS[base + 6], h_in}));
                    o = logistic(affine_transform({cg_WS[base + 11], cg_WS[base + 3], x, cg_WS[base + 7], h_in}));

                    Expression h_out, c_out;
                    c_out = dynet::cmult(i, c_) + dynet::cmult(f, c_in);
                    h_out = dynet::cmult(o, tanh(c_out));
                    _cg->mark_basic_block(false);
                    hs.push_back(h_out);
                    cs.push_back(c_out);
                }
            }

            if (hs.size() == 0)
                hs.push_back(h0);
            if (cs.size() == 0)
                cs.push_back(c0);

            node.h = hs.size() > 1 ? dynet::sum(hs) : hs.front();
            node.c = cs.size() > 1 ? dynet::sum(cs) : cs.front();
        }

        for (auto &node : g.nodes)
        {
            losses.push_back(pickneglogsoftmax(wo * node.h, node.tag));
        }
    }

    void LatticeLSTMBuilder::start_graph(ComputationGraph & cg){
        _cg = &cg;
        h0 = parameter(cg, H0);
        c0 = parameter(cg, C0);
        wo = parameter(cg, WO);
        if (!dynet::blocked){
            cg_WS.resize(WS.size());
            for (size_t i = 0; i < WS.size(); ++i)
                cg_WS[i] = parameter(*_cg, WS[i]);
        }
        cg.mark_basic_block(false);
    }

    LatticeGRUBuilder::LatticeGRUBuilder(ParameterCollection& model, unsigned hdim){
        unsigned wdim = hdim;
        // cerr << "build lattice gru "
        // cerr << "c_vocab_size" << c_vocab_size << endl;
        // cerr << "w_vocab_size" << w_vocab_size << endl;
        // cerr << "type_size" << type_size << endl;
        E_c = model.get_lookup_parameters(c_vocab_size, {wdim}, "CharLuT");
        E_w = model.get_lookup_parameters(w_vocab_size, {wdim}, "WordLuT");
        H0 = model.add_parameters({hdim});
        C0 = model.add_parameters({hdim});
        WO = model.add_parameters({type_size, hdim});
        
        WS = {
            model.add_parameters({hdim, wdim}), // 0: Ur
            model.add_parameters({hdim, wdim}), // 1: Uz
            model.add_parameters({hdim, wdim}), // 2: Uc
            model.add_parameters({hdim, hdim}), // 3: Wr,
            model.add_parameters({hdim, hdim}), // 4: Wz
            model.add_parameters({hdim, hdim}), // 5: Wc
            model.add_parameters({hdim}), // 6: br 
            model.add_parameters({hdim}), // 7: bz
            model.add_parameters({hdim}), // 8: bc
            model.add_parameters({hdim, wdim}), // 0: Ur
            model.add_parameters({hdim, wdim}), // 1: Uz
            model.add_parameters({hdim, wdim}), // 2: Uc
            model.add_parameters({hdim, hdim}), // 3: Wr,
            model.add_parameters({hdim, hdim}), // 4: Wz
            model.add_parameters({hdim, hdim}), // 5: Wc
            model.add_parameters({hdim}), // 6: br 
            model.add_parameters({hdim}), // 7: bz
            model.add_parameters({hdim}), // 8: bc
        };

        for (bool trivial: {false, true}) {
            blocks.push_back(make_unique<OoC::Block>("lattice_gru" + to_string(blocks.size())));
            auto & block = *blocks.back();
            int base = trivial? 0:9;
            vector<Expression> params;
            for (int i = 0; i < 9; ++i) params.push_back(parameter(block, WS[base + i]));
            block.finish_params();
            Expression h_in, x;
            h_in = block.placeholder({{hdim}}, "h"); 
            block.finish_input();
            if (trivial) x = block.lookup(E_c);
            else x = block.lookup(E_w);
            Expression r,z,c,h;
            r = logistic(matmul(params[0], h_in, false) + matmul(params[3], x, false) + params[6]);
            z = logistic(matmul(params[1], h_in, false) + matmul(params[4], x, false) + params[7]);
            c = tanh(matmul(params[2], cmult(r, h_in), false) + matmul(params[5], x, false) + params[8]);
            h = cmult(z, h_in) + cmult(1-z, c);
            block.output({h});
            block.freeze();
        }
    }

    void LatticeGRUBuilder::build_graph(DAG& g, std::vector<Expression>& losses) {
        for (auto &node : g.nodes)
        {
            std::vector<Expression> hs;

            for (auto &arc : node.arcs)
            {
                assert(g.nodes[arc.from].h.pg!=nullptr);
                if (dynet::blocked)
                {
                    Expression h = blocks[arc.trivial]->operator()(_cg, {{"h", g.nodes[arc.from].h}},
                                                       {(unsigned)arc.id});
                    hs.push_back(h);
                }
                else
                {
                    // lstm: arc.h, arc.c = GRU(nodes[arc.from.h], nodes[arc.from.c])
                    int base = arc.trivial ? 0 : 9;
                    Expression h_in, c_in, x;
                    h_in = g.nodes[arc.from].h;
                    if (arc.trivial)
                        x = lookup(*_cg, E_c, arc.id);
                    else
                        x = lookup(*_cg, E_w, arc.id);
                    Expression r,z,c,h;
                    r = logistic(affine_transform({cg_WS[base+6], cg_WS[base], h_in, cg_WS[3], x}));
                    z = logistic(affine_transform({cg_WS[base+7], cg_WS[base+1], h_in, cg_WS[4], x}));
                    c = tanh(affine_transform({cg_WS[base+8], cg_WS[base+2], cmult(r, h_in), cg_WS[5], x}));
                    h = cmult(z, h_in) + cmult(1-z, c);
                    hs.push_back(h);
                }
            }

            if (hs.size() == 0)
                hs.push_back(h0);

            node.h = hs.size() > 1 ? dynet::sum(hs) : hs.front();
        }

        for (auto &node : g.nodes) {
            losses.push_back(pickneglogsoftmax(wo * node.h, node.tag));
        }
    }

    void LatticeGRUBuilder::start_graph(ComputationGraph & cg){
        _cg = &cg;
        h0 = parameter(cg, H0);
        c0 = parameter(cg, C0);
        wo = parameter(cg, WO);
        if (!dynet::blocked){
            cg_WS.resize(WS.size());
            for (size_t i = 0; i < WS.size(); ++i)
                cg_WS[i] = parameter(*_cg, WS[i]);
        }
    }

    Lattice::Lattice(ParameterCollection& model, int hdim, Lattice::type_t type): model(model), type(type)
    {
        train = read_dataset("./data/lattice/train.txt");
        if (type == Lattice::LSTM)
            builder = new LatticeLSTMBuilder(model, hdim);
        else if (type == Lattice::GRU)
            builder = new LatticeGRUBuilder(model, hdim);
    }

    Expression Lattice::build_graph(ComputationGraph &cg, int batch_size)
    {
        builder->start_graph(cg);
        vector<Expression> losses;
        for (int i = 0; i < batch_size; i++)
        {
            auto &data = train[(data_idx + i) % train.size()];
            builder->build_graph(data, losses);
        }
        Expression loss = dynet::sum(losses, true);
        data_idx = (data_idx + batch_size) % train.size();
        return loss;
    }

    RandomDAG::RandomDAG(
        dynet::ParameterCollection& model,
        unsigned hdim, 
        unsigned vocab_size, 
        int n_state, 
        int n_arc_type, 
        int n_state_type, 
        int n_data, 
        int in_degree, 
        int longest_dependency): model(model), output_block("out_block"){
        for (int i = 0; i < n_data; i++){
            data.push_back({});
            DAG& instance = data.back();
            for (int j = 0; j < n_state; j++){
                instance.nodes.push_back({});
                State& state = instance.nodes.back();
                state.tag = random() % n_state_type;
                if (j == 0) continue;
                for (int k = 0; k < in_degree; k++){
                    state.arcs.push_back({});
                    Arc& arc = state.arcs.back();
                    double factor = 0.1;
                    double p = random() / (RAND_MAX + 0.0) / (1 - factor);
                    double tmp = 1;
                    arc.id = 0;
                    while (p - tmp > 0 && arc.id < n_arc_type - 1){
                        tmp *= factor;
                        p -= tmp;
                        arc.id++;
                    }
                    arc.label = random() % vocab_size;
                    arc.from = j - (random() % std::min(longest_dependency, j)+1);
                }
            }
        }
        L = model.get_lookup_parameters(vocab_size, {{hdim}}, "L");
        for (int i = 0; i < n_arc_type; i++){
            blocks.push_back(new OoC::Block("block" + to_string(i)));
            WS.push_back(model.add_parameters({{hdim, hdim*2}}));
            BS.push_back(model.add_parameters({{hdim}}));
            OoC::Block & block = *blocks.back();
            Expression w = parameter(block, WS.back());
            Expression b = parameter(block, BS.back());
            block.finish_params();
            Expression h = block.placeholder({{hdim}}, "h");
            block.finish_input();
            Expression x = block.lookup(L);
            Expression o = tanh(w * concatenate({h,x}) + b);
            block.output({o});
            block.freeze();
        }

        S0 = model.add_parameters({{hdim}});

        {
            WS.push_back(model.add_parameters({{hdim, hdim}}));
            BS.push_back(model.add_parameters({{hdim}}));
            Expression w = parameter(output_block, WS.back());
            Expression b = parameter(output_block, BS.back());
            output_block.finish_params();
            Expression h = output_block.placeholder({{hdim}}, "h");
            output_block.finish_input();
            Expression o = output_block.pickneglogsoftmax(w*h+b);
            output_block.output({o});
            output_block.freeze();
        }
    }

    void RandomDAG::build_expr(ComputationGraph& cg, 
        DAG& dag, vector<Expression>& losses){
        Expression s = parameter(cg, S0);
        for (auto& state: dag.nodes) state.h = s;
        for (auto& state: dag.nodes){
            vector<Expression> hs = {state.h};
            for(auto& arc: state.arcs){
                Expression h = blocks[arc.id]->operator()(&cg, {{"h", dag.nodes[arc.from].h}}, {(unsigned)arc.label});
                hs.push_back(h);
            }
            if (hs.size() > 1)
                state.h = sum(hs);  
            losses.push_back(output_block(&cg, {{"h", state.h}}, {(unsigned)state.tag}));
        }
    }

    Expression RandomDAG::build_graph(ComputationGraph& cg, int batch_size){
        vector<Expression> losses;
        for (int i = data_idx; i < data_idx + batch_size; i++){
            build_expr(cg, data[i%data.size()], losses);
        }
        data_idx = (data_idx + batch_size) % data.size();
        return sum(losses);
    }

    

} // namespace OoCTest