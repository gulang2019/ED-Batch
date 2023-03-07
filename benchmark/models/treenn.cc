#include <vector>
#include <stdexcept>
#include <fstream>
#include <chrono>
#ifdef BOOST_REGEX
#include <boost/regex.hpp>
using namespace boost;
#else
#include <regex>
#endif

#include <dynet/training.h>
#include <dynet/expr.h>
#include <dynet/dict.h>
#include <string>
#include <dynet/param-init.h>
#include "dynet/ooc-block.h"

#include "treenn.hpp"
#include "model.hpp"

using namespace std;
using namespace std::chrono;
using namespace dynet;

namespace OoCTest
{

    Tree *Tree::from_sexpr(const string &str)
    {
        vector<string> toks = tokenize_sexpr(str);
        vector<string>::const_iterator tokit = toks.begin();
        if (*(tokit++) != "(")
            throw runtime_error("Poorly structured tree");
        return Tree::within_bracket(tokit);
    }

    vector<string> Tree::tokenize_sexpr(const string &s)
    {
        regex tokker(" +|[()]|[^ ()]+");
        vector<string> toks;
        for (auto it = sregex_iterator(s.begin(), s.end(), tokker); it != sregex_iterator(); ++it)
        {
            string m = it->str();
            if (m != " ")
                toks.push_back(m);
        }
        return toks;
    }

    Tree *Tree::within_bracket(std::vector<std::string>::const_iterator &tokit)
    {
        const string &label = *(tokit++);
        vector<Tree *> children;
        while (true)
        {
            const string &tok = *(tokit++);
            if (tok == "(")
            {
                children.push_back(within_bracket(tokit));
            }
            else if (tok == ")")
            {
                return new Tree(label, children);
            }
            else
            {
                children.push_back(new Tree(tok));
            }
        }
        throw runtime_error("Poorly structured tree");
    }

    vector<Tree *> read_dataset(const string &filename)
    {
        ifstream file(filename);
        if (!file)
            throw runtime_error("Missing file");
        string line;
        vector<Tree *> ret;
        while (getline(file, line))
            ret.push_back(Tree::from_sexpr(line));
        return ret;
    }

    TreeLSTMBuilder::TreeLSTMBuilder(dynet::ParameterCollection &model, dynet::Dict &word_vocab, unsigned wdim, unsigned hdim, bool double_typed) 
    : model(model), word_vocab(word_vocab), wdim(wdim), hdim(hdim), input_block("input_block"), double_typed(double_typed)
    , internal_block("internal_block"), internal_block1("internal_block1")
    {
        WS = {model.add_parameters({hdim, wdim}, ParameterInitConst(0.1)),     // 0: Wi
              model.add_parameters({hdim, wdim}, ParameterInitConst(0.1)),     // 1: Wo
              model.add_parameters({hdim, wdim}, ParameterInitConst(0.1)),     // 2: Wu
              model.add_parameters({hdim, 2 * hdim}, ParameterInitConst(0.1)), // 3: Ui
              model.add_parameters({hdim, 2 * hdim}, ParameterInitConst(0.1)), // 4: Uo
              model.add_parameters({hdim, 2 * hdim}, ParameterInitConst(0.1)), // 5: Uu
              model.add_parameters({hdim, hdim}, ParameterInitConst(0.1)),     // 6: UFS1
              model.add_parameters({hdim, hdim}, ParameterInitConst(0.1)),     // 7: UFS2
              model.add_parameters({hdim}, ParameterInitConst(0.1)),           // 8: Bi
              model.add_parameters({hdim}, ParameterInitConst(0.1)),           // 9: Bo
              model.add_parameters({hdim}, ParameterInitConst(0.1)),           // 10: Bu
              model.add_parameters({hdim}, ParameterInitConst(0.1)),          // 11: Bf
              // for the double typed kernel
              model.add_parameters({hdim, wdim}, ParameterInitConst(0.1)),     // 0: Wi
              model.add_parameters({hdim, wdim}, ParameterInitConst(0.1)),     // 1: Wo
              model.add_parameters({hdim, wdim}, ParameterInitConst(0.1)),     // 2: Wu
              model.add_parameters({hdim, 2 * hdim}, ParameterInitConst(0.1)), // 3: Ui
              model.add_parameters({hdim, 2 * hdim}, ParameterInitConst(0.1)), // 4: Uo
              model.add_parameters({hdim, 2 * hdim}, ParameterInitConst(0.1)), // 5: Uu
              model.add_parameters({hdim, hdim}, ParameterInitConst(0.1)),     // 6: UFS1
              model.add_parameters({hdim, hdim}, ParameterInitConst(0.1)),     // 7: UFS2
              model.add_parameters({hdim}, ParameterInitConst(0.1)),           // 8: Bi
              model.add_parameters({hdim}, ParameterInitConst(0.1)),           // 9: Bo
              model.add_parameters({hdim}, ParameterInitConst(0.1)),           // 10: Bu
              model.add_parameters({hdim}, ParameterInitConst(0.1))};          // 11: Bf
        E = model.add_lookup_parameters(std::min(word_vocab.size(), (unsigned)MAX_VOCAB_SIZE), {wdim}, ParameterInitConst(0.1));
        define_bb();
    }

    void TreeLSTMBuilder::define_bb()
    {
        {
            Expression bi, bo, bu, wi, wo, wu;
            bi = parameter(input_block, WS[8]);
            bo = parameter(input_block, WS[9]);
            bu = parameter(input_block, WS[10]);
            wi = parameter(input_block, WS[0]);
            wo = parameter(input_block, WS[1]);
            wu = parameter(input_block, WS[2]);
            input_block.finish_params();

            input_block.finish_input();

            Expression emb, i, o, u, h, c;
            emb = input_block.lookup(E);
            i = logistic(matmul(wi, emb, false) + bi);
            o = logistic(matmul(wo, emb, false) + bo);
            u = tanh(matmul(wu, emb, false) + bu);
            c = cmult(i, u);
            h = cmult(o, tanh(c));
            input_block.output({h, c});
            input_block.freeze();
        }

        {
            Expression bi, bo, bu, bf;
            bi = parameter(internal_block, WS[8]);
            bo = parameter(internal_block, WS[9]);
            bu = parameter(internal_block, WS[10]);
            bf = parameter(internal_block, WS[11]);
            Expression ui, uo, uu, ufs1, ufs2;
            ui = parameter(internal_block, WS[3]);
            uo = parameter(internal_block, WS[4]);
            uu = parameter(internal_block, WS[5]);
            ufs1 = parameter(internal_block, WS[6]);
            ufs2 = parameter(internal_block, WS[7]);
            internal_block.finish_params();
            
            Expression h1, c1, h2, c2;
            h1 = internal_block.placeholder({{hdim}}, "h1");
            c1 = internal_block.placeholder({{hdim}}, "c1");
            h2 = internal_block.placeholder({{hdim}}, "h2");
            c2 = internal_block.placeholder({{hdim}}, "c2");
            internal_block.finish_input();

            Expression e, i, o, u, f1, f2, h, c;
            e = concatenate({h1, h2});
            i = logistic(matmul(ui, e, false) + bi);
            o = logistic(matmul(uo, e, false) + bo);
            u = tanh(matmul(uu, e, false) + bu);
            f1 = logistic(matmul(ufs1, h1, false) + bf);
            f2 = logistic(matmul(ufs2, h2, false) + bf);
            c = cmult(i, u) + cmult(f1, c1) + cmult(f2, c2);
            h = cmult(o, tanh(c));
            internal_block.output({h, c});
            internal_block.freeze();
        }
        
        if(double_typed)
        {
            Expression bi, bo, bu, bf;
            bi = parameter(internal_block1, WS[8]);
            bo = parameter(internal_block1, WS[9]);
            bu = parameter(internal_block1, WS[10]);
            bf = parameter(internal_block1, WS[11]);
            Expression ui, uo, uu, ufs1, ufs2;
            ui = parameter(internal_block1, WS[3]);
            uo = parameter(internal_block1, WS[4]);
            uu = parameter(internal_block1, WS[5]);
            ufs1 = parameter(internal_block1, WS[6]);
            ufs2 = parameter(internal_block1, WS[7]);
            internal_block1.finish_params();
            Expression h1, c1, h2, c2;
            h1 = internal_block1.placeholder({{hdim}}, "h1");
            c1 = internal_block1.placeholder({{hdim}}, "c1");
            h2 = internal_block1.placeholder({{hdim}}, "h2");
            c2 = internal_block1.placeholder({{hdim}}, "c2");
            internal_block1.finish_input();
            Expression e, i, o, u, f1, f2, h, c;
            e = concatenate({h1, h2});
            i = logistic(matmul(ui, e, false) + bi);
            o = logistic(matmul(uo, e, false) + bo);
            u = tanh(matmul(uu, e, false) + bu);
            f1 = logistic(matmul(ufs1, h1, false) + bf);
            f2 = logistic(matmul(ufs2, h2, false) + bf);
            c = cmult(i, u) + cmult(f1, c1) + cmult(f2, c2);
            h = cmult(o, tanh(c));
            internal_block1.output({h, c});
            internal_block1.freeze();
        }
    }

    std::pair<dynet::Expression, dynet::Expression> TreeLSTMBuilder::expr_for_tree_impl(Tree *tree, bool decorate)
    {
        assert(tree && !tree->isleaf());
        if (visited.count(tree)) return visited[tree];
        pair<Expression, Expression> hc_ret;
        if (tree->children.size() == 1)
        {
            assert(tree->children[0]->isleaf());
            if (dynet::blocked)
            {
                Expression hc = input_block(cg, {}, {(unsigned)word_vocab.convert(tree->children[0]->label) % MAX_VOCAB_SIZE});
                hc_ret = make_pair(hc[0], hc[1]);
            }
            else
            {
                Expression emb, i, o, u, c, expr;
                emb = lookup(*cg, E, word_vocab.convert(tree->children[0]->label) % MAX_VOCAB_SIZE);
                i = logistic(affine_transform({cg_WS[8], cg_WS[0], emb}));
                o = logistic(affine_transform({cg_WS[9], cg_WS[1], emb}));
                u = tanh(affine_transform({cg_WS[10], cg_WS[2], emb}));
                hc_ret.second = cmult(i, u);
                hc_ret.first = cmult(o, tanh(hc_ret.second));
            }
        }
        else
        {
            assert(tree->children.size() == 2);
            pair<Expression, Expression> hc1, hc2;
            hc1 = expr_for_tree_impl(tree->children[0], decorate);
            hc2 = expr_for_tree_impl(tree->children[1], decorate);
            if (dynet::blocked)
            {
                Expression hc;
                if ((random() & 1) || !double_typed)
                    hc = internal_block(cg, {{"h1", hc1.first}, {"c1", hc1.second}, {"h2", hc2.first}, {"c2", hc2.second}}, {});
                else hc = internal_block1(cg, {{"h1", hc1.first}, {"c1", hc1.second}, {"h2", hc2.first}, {"c2", hc2.second}}, {});
                hc_ret = make_pair(hc[0], hc[1]);
            }
            else
            {
                int base = ((random()&1)||!double_typed)?0:12;
                Expression e, i, o, u, f1, f2, c, expr;
                e = concatenate({hc1.first, hc2.first});
                i = logistic(affine_transform({cg_WS[base + 8], cg_WS[base + 3], e}));
                o = logistic(affine_transform({cg_WS[base + 9], cg_WS[base + 4], e}));
                u = tanh(affine_transform({cg_WS[base + 10], cg_WS[base + 5], e}));
                f1 = logistic(affine_transform({cg_WS[base + 11], cg_WS[base + 6], hc1.first}));
                f2 = logistic(affine_transform({cg_WS[base + 11], cg_WS[base + 7], hc2.first}));
                hc_ret.second = cmult(i, u) + cmult(f1, hc1.second) + cmult(f2, hc2.second);
                hc_ret.first = cmult(o, tanh(hc_ret.second));
            }
        }
        visited[tree] = hc_ret;
        if (decorate)
        {
            tree->expr = hc_ret.first;
        }
        return hc_ret;
    }

    Expression Treenn::build_graph(ComputationGraph &cg, int batch_size)
    {
        builder->start_graph(cg);
        Expression W = parameter(cg, W_param);
        vector<Expression> losses;
        for (size_t j2 = 0; j2 < batch_size; ++j2)
        {
            auto &tree = train[(data_idx + j2) % train.size()];
            builder->reset();
            builder->expr_for_tree(tree, true);
            // vector<Tree *> nodes;
            // tree->nonterms(nodes);
            vector<Tree*> visited;
            builder->get_visited(visited);
            for (auto tree : visited){
                losses.push_back(pickneglogsoftmax(W * tree->expr, nonterm_voc.convert(tree->label)));
            }
        }
        Expression loss_exp = sum(losses);
        data_idx = (data_idx + batch_size) % train.size();
        return loss_exp;
    }

    TreeGRUBuilder::TreeGRUBuilder(
        dynet::ParameterCollection &model, 
        dynet::Dict &word_vocab, 
        unsigned wdim, 
        unsigned hdim) 
    : model(model), word_vocab(word_vocab), wdim(wdim), hdim(hdim), input_block("input_block")
    , internal_block("internal_block")
    {
        WS = {model.add_parameters({hdim}),           // 0: Br
              model.add_parameters({hdim}),           // 1: Bz
              model.add_parameters({hdim}),           // 2: Bc
              model.add_parameters({hdim, wdim}),     // 3: Ur
              model.add_parameters({hdim, wdim}),     // 4: Uz
              model.add_parameters({hdim, wdim}),     // 5: Uc
              model.add_parameters({hdim, hdim}), // 6: Ur1
              model.add_parameters({hdim, hdim}), // 7: Uz2
              model.add_parameters({hdim, hdim}), // 8: Uc3
              model.add_parameters({hdim, hdim}), // 9: Ur1
              model.add_parameters({hdim, hdim}), // 10: Uz2
              model.add_parameters({hdim, hdim})}; // 11: Uc3
        E = model.add_lookup_parameters(std::min(word_vocab.size(), (unsigned)MAX_VOCAB_SIZE), {wdim}, ParameterInitConst(0.1));
        define_bb();
    }

    void TreeGRUBuilder::define_bb()
    {
        {
            Expression br, bz, bc, ur, uz, uc;
            br = parameter(input_block, WS[0]);
            bz = parameter(input_block, WS[1]);
            bc = parameter(input_block, WS[2]);
            ur = parameter(input_block, WS[3]);
            uz = parameter(input_block, WS[4]);
            uc = parameter(input_block, WS[5]);
            input_block.finish_params();

            input_block.finish_input();

            Expression emb, r, z, c, h;
            emb = input_block.lookup(E);
            r = logistic(matmul(ur, emb, false) + br);
            z = logistic(matmul(uz, emb, false) + bz);
            c = tanh(matmul(uc, cmult(r, emb), false) + bc);
            h = cmult(z, emb) + cmult(1-z, c);
            input_block.output({h});
            input_block.freeze();
        }

        {
            Expression br, bz, bc;
            br = parameter(internal_block, WS[0]);
            bz = parameter(internal_block, WS[1]);
            bc = parameter(internal_block, WS[2]);
            Expression ur1, ur2, uz1, uz2, uc1, uc2;
            ur1 = parameter(internal_block, WS[6]);
            ur2 = parameter(internal_block, WS[7]);
            uc1 = parameter(internal_block, WS[8]);
            uc2 = parameter(internal_block, WS[9]);
            uz1 = parameter(internal_block, WS[10]);
            uz2 = parameter(internal_block, WS[11]);
            internal_block.finish_params();
            Expression h1, h2;
            h1 = internal_block.placeholder({{hdim}}, "h1");
            h2 = internal_block.placeholder({{hdim}}, "h2");
            internal_block.finish_input();
            Expression r1, r2, c1, c2, z1, z2, h;
            r1 = logistic(matmul(ur1, h1, false) + br);
            r2 = logistic(matmul(ur2, h2, false) + br);
            z1 = logistic(matmul(uz1, h1, false) + bz);
            z2 = logistic(matmul(uz2, h2, false) + bz);
            c1 = tanh(matmul(uc1, cmult(r1, h1), false) + bc);
            c2 = tanh(matmul(uc2, cmult(r2, h2), false) + bc);
            h = ((cmult(z1, h1) + cmult(z2, h2)) + (cmult(1-z1, c1) + cmult(1-z2, c2))) / 2;

            internal_block.output({h});
            internal_block.freeze();
        }
    }

    dynet::Expression TreeGRUBuilder::expr_for_tree_impl(Tree *tree, bool decorate)
    {
        assert(tree && !tree->isleaf());
        if (visited.count(tree)) return visited[tree];
        Expression h;
        if (tree->children.size() == 1)
        {
            assert(tree->children[0]->isleaf());
            unsigned label = (unsigned)word_vocab.convert(tree->children[0]->label) % MAX_VOCAB_SIZE;
            if (dynet::blocked) {
                h = input_block(cg, {}, {label});
            }
            else
            {
                Expression emb, r, z, c, h;
                emb = lookup(*cg, E, label);
                r = logistic(affine_transform({cg_WS[0], cg_WS[3], emb}));
                z = logistic(affine_transform({cg_WS[1], cg_WS[4], emb}));
                c = tanh(affine_transform({cg_WS[2], cg_WS[5], cmult(r, emb)}));
                h = cmult(z, emb) + cmult(1-z, c);
            }
        }
        else
        {
            assert(tree->children.size() == 2);
            Expression h1, h2;
            h1 = expr_for_tree_impl(tree->children[0], decorate);
            h2 = expr_for_tree_impl(tree->children[1], decorate);
            if (dynet::blocked) {
                h = internal_block(cg, {{"h1", h1}, {"h2", h2}}, {});
            }
            else
            {
                Expression r1, r2, c1, c2, z1, z2, h;
                r1 = logistic(affine_transform({cg_WS[0], cg_WS[6], h1}));
                r2 = logistic(affine_transform({cg_WS[0], cg_WS[7], h2}));
                z1 = logistic(affine_transform({cg_WS[1], cg_WS[8], h1}));
                z2 = logistic(affine_transform({cg_WS[1], cg_WS[9], h2}));
                c1 = tanh(affine_transform({cg_WS[2], cg_WS[10], cmult(r1, h1)}));
                c2 = tanh(affine_transform({cg_WS[2], cg_WS[11], cmult(r2, h2)}));
                h = ((cmult(z1, h1) + cmult(z2, h2)) + (cmult(1-z1, c1) + cmult(1-z2, c2))) / 2;
            }
        }
        visited[tree] = h;
        if (decorate)
        {
            tree->expr = h;
        }
        return h;
    }

    Treenn::~Treenn()
    {
        for (auto tree : train)
            delete tree;
        delete builder;
    }

    std::string random_string( std::size_t length )
    {
        static const std::string alphabet = "abcdefghijklmnopqrstuvwxyz" ;
        static std::default_random_engine rng( std::time(nullptr) ) ;
        static std::uniform_int_distribution<std::size_t> distribution( 0, alphabet.size() - 1 ) ;

        std::string str ;
        while( str.size() < length ) str += alphabet[ distribution(rng) ] ;
        return str ;
    }

    std::string construct_perfect_tree(int depth, int max_depth){
        if (depth == max_depth) {
            return "(" + to_string((random()&0x11) + 1) + " " + random_string(5) + ")";    
        }
        std::string ret = "(" + to_string((random() &0x11) + 1) + " "; 
        ret += construct_perfect_tree(depth+1, max_depth);
        ret += construct_perfect_tree(depth+1, max_depth);
        ret += ") ";
        return ret;
    }

    Tree* construct_grid_tree(int depth){
        assert(depth >= 1);
        vector<Tree*> layer;
        for (int i = 0; i < depth; i++)
            layer.push_back(new Tree(to_string(random()&0x11 + 1), {new Tree(random_string(5))}));
        for (int i = depth-1; i > 0; --i){
            vector<Tree*> new_layer;
            for (int j = 0; j < i; j++){
                new_layer.push_back(new Tree(to_string(random()&0x11 + 1), {layer[j], layer[j+1]}));
            }
            layer = new_layer;
        }
        return layer.front();
    }

    Treenn::Treenn(dynet::ParameterCollection& model, 
        int wembed_size, 
        int hidden_size, 
        Treenn::type_t type, 
        int depth_min, int depth_max): model(model){
        if (type == NORMAL || type == DOUBLETYPE || type == GRU)
            train = read_dataset("./data/trees/train.txt");
        else if (type == PERFECT){
            for (int depth = depth_min; depth < depth_max; ++depth){
                train.push_back(Tree::from_sexpr(construct_perfect_tree(0, depth)));
            }
        }
        else if (type == GRID) {
            for (int depth = depth_min; depth < depth_max; ++depth){
                train.push_back(construct_grid_tree(depth));
            }
        }
        for (auto tree : train)
            tree->make_vocab(nonterm_voc, term_voc);
        nonterm_voc.freeze();
        term_voc.convert("<unk>");
        term_voc.freeze();
        term_voc.set_unk("<unk>");
        if (type == GRU)
            builder = new TreeGRUBuilder(model, term_voc, wembed_size, hidden_size);
        else builder = new TreeLSTMBuilder(model, term_voc, wembed_size, hidden_size, type == DOUBLETYPE);
        W_param = model.add_parameters({nonterm_voc.size(), (unsigned)hidden_size}, ParameterInitConst(0.5));
    }

    MVRNN::MVRNN(dynet::ParameterCollection& model, unsigned hdim)
    :model(model), block("mvrnn_internal")
    {
        train = read_dataset("./data/trees/train.txt");
        for (auto tree : train)
            tree->make_vocab(nonterm_voc, term_voc);
        nonterm_voc.freeze();
        term_voc.convert("<unk>");
        term_voc.freeze();
        term_voc.set_unk("<unk>");

        cout << "[MVRNN]: term_voc.size() " <<term_voc.size() << ", nonterm_voc.size() " << nonterm_voc.size() << endl;

        WS = {
            model.add_parameters({hdim, 2 * hdim}), // 0: Wv
            model.add_parameters({hdim}),           // 1: Bv
            model.add_parameters({hdim, 2 * hdim}), // 2: WM
            model.add_parameters({hdim}),           // 3: BM
        };

        LWS = {
            model.add_lookup_parameters(std::min((int)term_voc.size(), MAX_VOCAB_SIZE), {hdim, hdim}), // 0: M0
            model.add_lookup_parameters(std::min((int)term_voc.size(), MAX_VOCAB_SIZE), {hdim})        // 1: V0
        };

        {
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

        WO = model.add_parameters({nonterm_voc.size(), hdim});
    }

    std::pair<dynet::Expression, dynet::Expression>
    MVRNN::expr_for_tree(Tree &tree, bool decorate)
    {
        Expression m;
        Expression v;
        if (tree.children.size() == 1)
        {
            m = lookup(*_cg, LWS[0], term_voc.convert(tree.children[0]->label) % MAX_VOCAB_SIZE);
            v = lookup(*_cg, LWS[1], term_voc.convert(tree.children[0]->label) % MAX_VOCAB_SIZE);
        }
        else
        {
            auto pl = expr_for_tree(*tree.children[0], decorate);
            auto pr = expr_for_tree(*tree.children[1], decorate);
            if (dynet::blocked) {
                Expression o = block(_cg, {{"ml", pl.first}, {"mr", pr.first}, {"vl", pl.second}, {"vr", pr.second}}, {});
                v = o[0], m = o[1];
            }
            else {
                Expression v_comp = concatenate({matmul(pl.first, pr.second), matmul(pr.first, pl.second)});
                Expression m_comp = concatenate({pl.first, pr.first});

                v = tanh(affine_transform({ws[1], ws[0], v_comp}));
                m = affine_transform({ws[3], ws[2], m_comp});
            }
        }
        if (decorate)
        {
            tree.expr = v;
        }
        return std::make_pair(m, v);
    }

    Expression MVRNN::build_graph(ComputationGraph &cg, int batch_size)
    {
        _cg = &cg;
        vector<Expression> losses;
        Expression W = parameter(cg, WO);
        if (!dynet::blocked) {
            ws.resize(WS.size());
            for (size_t i = 0; i < WS.size(); ++i) {
                ws[i] = parameter(cg, WS[i]);
            }
        }
        for (size_t i = 0; i < batch_size; i++)
        {
            auto &tree = train[(data_idx + i) % train.size()];
            pair<Expression, Expression> output = expr_for_tree(*tree, true);
            vector<Tree *> nodes;
            tree->nonterms(nodes);
            for (auto nt : nodes)
            {
                auto y = nonterm_voc.convert(nt->label);
                losses.push_back(pickneglogsoftmax(W * nt->expr, nonterm_voc.convert(nt->label)));
            }
        }
        Expression loss = sum(losses, true);
        data_idx += batch_size;
        return loss;
    }
} // namespace OoCTest
/*
(1(2(3 I)(4)) (()()))
*/