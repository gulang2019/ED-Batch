#ifdef BOOST_REGEX
  #include <boost/regex.hpp>
  using namespace boost;
#else
  #include <regex>
#endif

#include "bilstm-tagger.hpp"

using namespace dynet;
using namespace std;

namespace OoCTest
{

    vector<pair<vector<string>, vector<string>>> read(const string &fname)
    {
        ifstream fh(fname);
        if (!fh)
            throw std::runtime_error("Could not open file");
        string str;
        regex re("[ |]");
        vector<pair<vector<string>, vector<string>>> sents;
        while (getline(fh, str))
        {
            pair<vector<string>, vector<string>> word_tags;
            sregex_token_iterator first{str.begin(), str.end(), re, -1}, last;
            while (first != last)
            {
                word_tags.first.push_back(*first++);
                assert(first != last);
                word_tags.second.push_back(*first++);
            }
            sents.push_back(word_tags);
        }
        return sents;
    }

    BilstmTagger::BilstmTagger(
        ParameterCollection& model, unsigned layers, 
        unsigned cembed_dim, unsigned wembed_dim, 
        unsigned hidden_dim, unsigned mlp_dim, bool withchar)
        : model(model), layers(layers), calc_loss_block("calc_loss"), withchar(withchar)
    {
        train = read("./data/tags/train.txt");
        for (auto &sent : train)
        {
            for (auto &w : sent.first)
            {
                wv.convert(w);
                wc[w]++;
                for (size_t i = 0; i < w.size(); ++i)
                    cv.convert(w.substr(i, 1));
            }
            for (auto &t : sent.second)
                tv.convert(t);
        }
        tv.freeze();
        wv.convert("<unk>");
        wv.freeze();
        wv.set_unk("<unk>");
        cv.convert("<*>");
        cv.freeze();

        unsigned nwords = wv.size();
        unsigned ntags = tv.size();
        unsigned nchars = cv.size();
        word_lookup = model.get_lookup_parameters(nwords, {wembed_dim}, "WordLuT");
        char_lookup = model.get_lookup_parameters(nchars, {cembed_dim}, "CharLuT");

        // MLP on top of biLSTM outputs 100 -> mlp_dim -> ntags
        pH = model.add_parameters({mlp_dim, hidden_dim * 2});
        pO = model.add_parameters({ntags, mlp_dim});

        // word-level LSTMs
        fwdRNN = VanillaLSTMBuilder(layers, wembed_dim, hidden_dim, model, false, 1.0F, true, "fwdRNN"); // layers, in-dim, out-dim, model
        bwdRNN = VanillaLSTMBuilder(layers, wembed_dim, hidden_dim, model, false, 1.0F, true, "bwdRNN");

        // char-level LSTMs
        cFwdRNN = VanillaLSTMBuilder(layers, cembed_dim, wembed_dim / 2, model, false, 1.0F, true, "cFwdRNN");
        cBwdRNN = VanillaLSTMBuilder(layers, cembed_dim, wembed_dim / 2, model, false, 1.0F, true, "cBwdRNN");

        for (int i = 0; i < layers * 2; i++)
            h0_c_p.push_back(model.add_parameters({{wembed_dim / 2}}));
        for (int i = 0; i < layers * 2; i++)
            h0_w_p.push_back(model.add_parameters({{hidden_dim}}));

        {
            Expression o = parameter(calc_loss_block, pO);
            Expression h = parameter(calc_loss_block, pH);
            calc_loss_block.finish_params();
            Expression fwd = calc_loss_block.placeholder({{hidden_dim}}, "fwd");
            Expression bwd = calc_loss_block.placeholder({{hidden_dim}}, "bwd");
            calc_loss_block.finish_input();
            calc_loss_block.output({calc_loss_block.pickneglogsoftmax(o * tanh(h * concatenate({fwd, bwd})))});
            calc_loss_block.freeze();
        }
    }

    Expression BilstmTagger::word_rep(ComputationGraph &cg, const string &w)
    {
        if ((!withchar) || wc[w] >= 5)
        {
            auto ret = lookup(cg, word_lookup, wv.convert(w));
            // cg.mark_basic_block(false);
            return ret;
        }
        else
        {
            Expression ret;
            Expression pad = lookup(cg, char_lookup, cv.convert("<*>"));
            // cg.mark_basic_block(false);
            vector<Expression> cembs(w.size() + 2, pad);
            for (size_t i = 0; i < w.size(); ++i)
            {
                cembs[i + 1] = lookup(cg, char_lookup, cv.convert(w.substr(i, 1)));
                cg.mark_basic_block(false);
            }
            cFwdRNN.start_new_sequence(h0_c);
            cBwdRNN.start_new_sequence(h0_c);
            for (size_t i = 0; i < cembs.size(); ++i)
            {
                cFwdRNN.add_input(cembs[i]);
                cBwdRNN.add_input(cembs[cembs.size() - i - 1]);
            }
            ret = concatenate({cFwdRNN.back(), cBwdRNN.back()});

            return ret;
        }
    }

    pair<vector<Expression>, vector<Expression>> BilstmTagger::build_tagging_graph(ComputationGraph &cg, const vector<string> &words)
    {
        // parameters -> expressions
        // get the word vectors. word_rep(...) returns a 128-dim vector expression for each word.
        vector<Expression> wembs(words.size()), fwds(words.size()), bwds(words.size()), fbwds(words.size());
        for (size_t i = 0; i < words.size(); ++i)
            wembs[i] = word_rep(cg, words[i]);

        // feed word vectors into biLSTM
        fwdRNN.start_new_sequence(h0_w);
        bwdRNN.start_new_sequence(h0_w);
        for (size_t i = 0; i < wembs.size(); ++i)
        {
            fwds[i] = fwdRNN.add_input(wembs[i]);
            bwds[wembs.size() - 1 - i] = bwdRNN.add_input(wembs[wembs.size() - i - 1]);
        }

        return {fwds, bwds};
    }

    void BilstmTagger::new_graph(ComputationGraph &cg)
    {
        fwdRNN.new_graph(cg);
        bwdRNN.new_graph(cg);
        cFwdRNN.new_graph(cg);
        cBwdRNN.new_graph(cg);
        h0_c.clear();
        h0_w.clear();
        for (auto &param : h0_c_p)
            h0_c.push_back(parameter(cg, param));
        for (auto &param : h0_w_p)
            h0_w.push_back(parameter(cg, param));
    }

    Expression BilstmTagger::build_graph(dynet::ComputationGraph &cg,
                                   int batch_size)
    {
        new_graph(cg);
        vector<Expression> losses;
        if (!dynet::blocked) {
            O = parameter(cg, pO);
            H = parameter(cg, pH);
        }
        for (int i = data_idx; i < data_idx + batch_size; i++)
        {
            auto &s = train[i % train.size()];
            sent_loss(cg, s.first, s.second, losses);
        }
        Expression loss_exp = sum(losses, true);
        data_idx = (data_idx + batch_size) % train.size();
        return loss_exp;
    }

    void BilstmTagger::sent_loss(ComputationGraph &cg, vector<string> &words, vector<string> &tags, vector<Expression> &losses)
    {
        vector<Expression> fwds, bwds;
        std::tie(fwds, bwds) = build_tagging_graph(cg, words);
        if (dynet::blocked)
            for (size_t i = 0; i < tags.size(); ++i)
                losses.push_back(calc_loss_block(&cg, {{"fwd", fwds[i]}, {"bwd", bwds[i]}}, {(unsigned)tv.convert(tags[i])}));
        else {
            for (size_t i = 0; i < tags.size(); ++i) {
                losses.push_back(pickneglogsoftmax(O * tanh(H * concatenate({fwds[i], bwds[i]})), (unsigned)tv.convert(tags[i])));
            }
        }
        return;
    }
} // namespace OoCTest