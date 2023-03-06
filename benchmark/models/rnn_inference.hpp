#pragma once
#include <string>
#include <vector>
#include <tuple>

#include "model.hpp"

using std::vector;
using std::tuple;
using dynet::Dict;
using dynet::Parameter;

typedef int WordId;// word Id
typedef std::vector<WordId> WordIdSentence;// word Id sentence
typedef std::vector<WordIdSentence> WordIdSentences;// batches of sentences
typedef tuple<WordIdSentence, WordIdSentence> WordIdSentencePair; // Note: can be extended to include additional information (e.g., document ID)
typedef vector<WordIdSentencePair> WordIdCorpus;

namespace OoCTest{
    class LSTMNMT: public Model {
    public:
        LSTMNMT(dynet::ParameterCollection& model, unsigned hidden_size, unsigned embed_dim); 
        dynet::Expression build_graph (dynet::ComputationGraph & cg, 
            int batch_size) override;
        void reset() override;

    private: 
        WordIdCorpus corpus;
        Dict sd, td;
        int index = 0;
        OoC::Block encoder, decoder;
        vector<vector<Parameter> > WS;
        unsigned hdim, wdim;
        dynet::LookupParameter s_embed, d_embed; 
        Parameter W, H0, C0;
    };
} // 