#include <vector>
#include <dynet/dict.h>

#include "rnn_inference.hpp"

using namespace std;
using namespace dynet;

namespace OoCTest {
    vector<vector<unsigned> > read(const string & fname, Dict & vw) {
        ifstream fh(fname);
        if(!fh) throw std::runtime_error("Could not open file");
        string str; 
        vector<vector<unsigned> > sents;
        while(getline(fh, str)) {
            istringstream iss(str);
            vector<unsigned> tokens;
            while(iss >> str)
            tokens.push_back(vw.convert(str));
            tokens.push_back(vw.convert("<s>"));
            sents.push_back(tokens);
        }
        return sents;
    }


    WordIdCorpus read_corpus(const string &filename
        , dynet::Dict* sd, dynet::Dict* td
        , bool cid
        , unsigned slen, bool r2l_target
        , bool swap)
    {
        int kSRC_SOS = sd->convert("<s>");
        int kSRC_EOS = sd->convert("</s>");
        int kTGT_SOS = td->convert("<s>");
        int kTGT_EOS = td->convert("</s>");

        ifstream in(filename);
        assert(in);

        WordIdCorpus corpus;

        string line;
        int lc = 0, stoks = 0, ttoks = 0;
        unsigned int max_src_len = 0, max_tgt_len = 0;
        while (getline(in, line)) {
            WordIdSentence source, target;

            if (!swap)
                read_sentence_pair(line, source, *sd, target, *td);
            else read_sentence_pair(line, source, *td, target, *sd);

            // reverse the target if required
            if (r2l_target) 
                std::reverse(target.begin() + 1/*BOS*/,target.end() - 1/*EOS*/);

            // constrain sentence length(s)
            if (cid/*train only*/ && slen > 0/*length limit*/){
                if (source.size() > slen || target.size() > slen)
                    continue;// ignore this sentence
            }

            if ((source.front() != kSRC_SOS && source.back() != kSRC_EOS) ||
                    (target.front() != kTGT_SOS && target.back() != kTGT_EOS)) {
                stringstream ss;
                ss << "Sentence in " << filename << ":" << lc << " didn't start or end with <s>, </s>\n";
                assert(ss.str().c_str());

                abort();
            }

            if (source.size() < 3 || target.size() < 3){ // ignore empty sentences, e.g., <s> </s>
                continue;
            }

            corpus.push_back(WordIdSentencePair(source, target));

            max_src_len = std::max(max_src_len, (unsigned int)source.size());
            max_tgt_len = std::max(max_tgt_len, (unsigned int)target.size());

            stoks += source.size();
            ttoks += target.size();

            ++lc;
    }

	// print stats
	if (cid)
		cerr << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t), " << "max length (s & t): " << max_src_len << " & " << max_tgt_len << ", " << sd->size() << " & " << td->size() << " types" << endl;
	else 
		cerr << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t), " << "max length (s & t): " << max_src_len << " & " << max_tgt_len << endl;

	return corpus;
}


    LSTMNMT::LSTMNMT(ParameterCollection& model, unsigned hidden_size, unsigned embed_dim):
    hdim(hidden_size), wdim(embed_dim) {
        corpus = read_corpus("./data/iwslt15/train.en-vi.capped", &sd, &td, 0, 512, false, false);
        s_embed = model.add_lookup_parameters(std::min(sd.size(), (unsigned)MAX_VOCAB_SIZE), {{embed_dim}});
        d_embed = model.add_lookup_parameters(std::min(td.size(), (unsigned)MAX_VOCAB_SIZE), {{embed_dim}});

        W = model.add_parameters({{std::min(td.size(), (unsigned)MAX_VOCAB_SIZE), hdim}});
        H0 = model.add_parameters({{hdim}});
        C0 = model.add_parameters({{hdim}});

        for (int i = 0; i < 2; i++){
            WS.push_back({
                model.add_parameters({hdim, hdim}), // 0: Uf
                model.add_parameters({hdim, hdim}), // 1: Uc
                model.add_parameters({hdim, hdim}), // 2: Ui
                model.add_parameters({hdim, hdim}), // 3: Uo
                model.add_parameters({hdim, hdim}), // 4: Wf
                model.add_parameters({hdim, hdim}), // 5: Wc
                model.add_parameters({hdim, hdim}), // 6: Wi
                model.add_parameters({hdim, hdim}), // 7: Wo
                model.add_parameters({hdim}),       // 8: Bf
                model.add_parameters({hdim}),       // 9: Bc
                model.add_parameters({hdim}),       // 10: Bi
                model.add_parameters({hdim}),        // 11: Bo
                model.add_parameters({{hdim, wdim}}), // 12:  E2H
                model.add_parameters({{hdim}}),  // 13: E2H bias
            });
        }

        for (int base = 0; base < 2; base++)
        {
            OoC::Block& block = base==0? encoder:decoder;
            vector<Expression> params;
            for (auto & p: WS[base])
                params.push_back(parameter(block, p));
            block.finish_params();
            Expression h_in = block.placeholder({{hdim}}, "h");
            Expression c_in = block.placeholder({{hdim}}, "c");
            block.finish_input();
            Expression x;
            if (base == 0) x = block.lookup(s_embed);
            else x = block.lookup(d_embed);
            x = tanh(matmul(params[12], x, false) + params[13]);
            Expression f, c_, i, o;
            f = logistic(params[8] + matmul(params[0], x, false) + matmul(params[4], h_in, false));
            c_ = tanh(params[9] + matmul(params[1], x, false) + matmul(params[5], h_in, false));
            i = logistic(params[10] + matmul(params[2], x, false) + matmul(params[6], h_in, false));
            o = logistic(params[11] + matmul(params[3], x, false) + matmul(params[7], h_in, false));

            Expression h_out, c_out;
            c_out = cmult(i, c_) + cmult(f, identity(c_in));
            h_out = cmult(o, tanh(c_out));
            block.output({h_out, c_out});
            block.freeze();
        }
    }

    Expression LSTMNMT::build_graph(
        dynet::ComputationGraph& cg, 
        int batch_size){

        Expression w = parameter(cg, W);
        int n_progressing = batch_size;
        int iter = 0;
        vector<Expression> losses;
        stack<Expression> S;
        vector<Expression> params;
        if(dynet::blocked == false)
            for (auto & p: WS[0])
                params.push_back(parameter(cg, p));

        // encoder 
        Expression h0 = parameter(cg, H0);
        Expression c0 = parameter(cg, C0);
        vector<Expression> hs(batch_size), cs(batch_size);
        for (int i = 0; i < batch_size; i ++){
            int idx = (index + i) % corpus.size();
            auto & sent = std::get<0>(corpus[idx]);
            Expression& h = hs[i], &c = cs[i];
            h = h0, c = c0;
            for (int j = 0; j < sent.size(); ++j){
                if (dynet::blocked) {
                    Expression hc = encoder(&cg, {{"h", h}, {"c", c}}, {(unsigned)sent[j] % MAX_VOCAB_SIZE});
                    h = hc[0], c = hc[1];                    
                }
                else {
                    Expression x = lookup(cg, s_embed, sent[j] % MAX_VOCAB_SIZE);
                    x = tanh(matmul(params[12], x, false) + params[13]);
                    Expression f, c_, i_, o;
                    f = logistic(params[8] + matmul(params[0], x, false) + matmul(params[4], h, false));
                    c_ = tanh(params[9] + matmul(params[1], x, false) + matmul(params[5], h, false));
                    i_ = logistic(params[10] + matmul(params[2], x, false) + matmul(params[6], h, false));
                    o = logistic(params[11] + matmul(params[3], x, false) + matmul(params[7], h, false));
                    c = cmult(i_, c_) + cmult(f, identity(c));
                    h = cmult(o, tanh(c));
                }
            }
        }


        if (dynet::blocked == false){
            params.clear();
            for (auto & p: WS[1]) 
                params.push_back(parameter(cg, p));
        }

        // decoder
        while (n_progressing) {
            for (int i = 0; i < batch_size; i ++) {
                int idx = (index + i) % corpus.size();
                auto & sent = std::get<1>(corpus[idx]);
                if (sent.size() <= iter) continue;
                int len = sent.size(); 
                auto & h = hs[i], & c = cs[i];
                if (dynet::blocked) {
                    Expression hc = decoder(&cg, {{"h", h}, {"c", c}}, {(unsigned)sent[idx] % MAX_VOCAB_SIZE});
                    h = hc[0];
                    c = hc[1];
                }
                else {
                    Expression f, c_, i_, o, x;
                    x = lookup(cg, d_embed, sent[iter] % MAX_VOCAB_SIZE);
                    x = tanh(matmul(params[12], x, false) + params[13]);
                    f = logistic(params[8] + matmul(params[0], x, false) + matmul(params[4], h, false));
                    c_ = tanh(params[9] + matmul(params[1], x, false) + matmul(params[5], h, false));
                    i_ = logistic(params[10] + matmul(params[2], x, false) + matmul(params[6], h, false));
                    o = logistic(params[11] + matmul(params[3], x, false) + matmul(params[7], h, false));

                    c = cmult(i_, c_) + cmult(f, c);
                    h = cmult(o, tanh(c));
                }
                Expression l = pickneglogsoftmax(w * hs[i], sent[iter] % MAX_VOCAB_SIZE);
                S.push(l);
                losses.push_back(l);
            }
            for (int i = batch_size - 1; i >= 0; --i){
                int idx = (index + i) % corpus.size();
                auto & sent = std::get<1>(corpus[idx]);
                if (sent.size() <= iter) continue;
                Expression l = S.top(); S.pop();
                global_timer.stop("construction");
                global_timer.start("forward");
                auto t = cg.forward(l);
                float v = as_scalar(t); // mimic tensor dependent, because we do not have actual parameters
                global_timer.stop("forward");
                global_timer.start("construction");
                if (iter == sent.size()-1) {
                    n_progressing --;
                }
            }
            iter++;
        }

        index = (index + batch_size) % corpus.size();
        return sum(losses);
    }

    void LSTMNMT::reset() {index = 0;}

} // namespace OoCTest 