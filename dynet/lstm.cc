#include "dynet/lstm.h"

#include "dynet/param-init.h"
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

using namespace std;

namespace dynet {

enum { X2I, H2I, C2I, BI, X2O, H2O, C2O, BO, X2C, H2C, BC };

CoupledLSTMBuilder::CoupledLSTMBuilder(unsigned layers,
                         unsigned input_dim,
                         unsigned hidden_dim,
                         ParameterCollection& model) : layers(layers), input_dim(input_dim), hid(hidden_dim), dropout_masks_valid(false) {
  unsigned layer_input_dim = input_dim;
  local_model = model.add_subcollection("lstm-builder");
  for (unsigned i = 0; i < layers; ++i) {
    // i
    Parameter p_x2i = local_model.add_parameters({hidden_dim, layer_input_dim});
    Parameter p_h2i = local_model.add_parameters({hidden_dim, hidden_dim});
    Parameter p_c2i = local_model.add_parameters({hidden_dim, hidden_dim});
    Parameter p_bi = local_model.add_parameters({hidden_dim}, ParameterInitConst(0.f));

    // o
    Parameter p_x2o = local_model.add_parameters({hidden_dim, layer_input_dim});
    Parameter p_h2o = local_model.add_parameters({hidden_dim, hidden_dim});
    Parameter p_c2o = local_model.add_parameters({hidden_dim, hidden_dim});
    Parameter p_bo = local_model.add_parameters({hidden_dim}, ParameterInitConst(0.f));

    // c
    Parameter p_x2c = local_model.add_parameters({hidden_dim, layer_input_dim});
    Parameter p_h2c = local_model.add_parameters({hidden_dim, hidden_dim});
    Parameter p_bc = local_model.add_parameters({hidden_dim}, ParameterInitConst(0.f));

    layer_input_dim = hidden_dim;  // output (hidden) from 1st layer is input to next

    vector<Parameter> ps = {p_x2i, p_h2i, p_c2i, p_bi, p_x2o, p_h2o, p_c2o, p_bo, p_x2c, p_h2c, p_bc};
    params.push_back(ps);
  }  // layers
  dropout_rate = 0.f;
  dropout_rate_h = 0.f;
  dropout_rate_c = 0.f;
}

void CoupledLSTMBuilder::new_graph_impl(ComputationGraph& cg, bool update) {
  param_vars.clear();

  for (unsigned i = 0; i < layers; ++i) {
    auto& p = params[i];

    //i
    Expression i_x2i = update ? parameter(cg, p[X2I]) : const_parameter(cg, p[X2I]);
    Expression i_h2i = update ? parameter(cg, p[H2I]) : const_parameter(cg, p[H2I]);
    Expression i_c2i = update ? parameter(cg, p[C2I]) : const_parameter(cg, p[C2I]);
    Expression i_bi = update ? parameter(cg, p[BI]) : const_parameter(cg, p[BI]);
    //o
    Expression i_x2o = update ? parameter(cg, p[X2O]) : const_parameter(cg, p[X2O]);
    Expression i_h2o = update ? parameter(cg, p[H2O]) : const_parameter(cg, p[H2O]);
    Expression i_c2o = update ? parameter(cg, p[C2O]) : const_parameter(cg, p[C2O]);
    Expression i_bo = update ? parameter(cg, p[BO]) : const_parameter(cg, p[BO]);
    //c
    Expression i_x2c = update ? parameter(cg, p[X2C]) : const_parameter(cg, p[X2C]);
    Expression i_h2c = update ? parameter(cg, p[H2C]) : const_parameter(cg, p[H2C]);
    Expression i_bc = update ? parameter(cg, p[BC]) : const_parameter(cg, p[BC]);

    vector<Expression> vars = {i_x2i, i_h2i, i_c2i, i_bi, i_x2o, i_h2o, i_c2o, i_bo, i_x2c, i_h2c, i_bc};
    param_vars.push_back(vars);
  }
  _cg = &cg;
}

// layout: 0..layers = c
//         layers+1..2*layers = h
void CoupledLSTMBuilder::start_new_sequence_impl(const vector<Expression>& hinit) {
  // Check input dim and hidden dim
  if (input_dim != params[0][X2I].dim()[1]) {
    cerr << "Warning : LSTMBuilder input dimension " << input_dim
         << " doesn't match with parameter dimension " << params[0][X2I].dim()[1]
         << ". Setting input_dim to " << params[0][X2I].dim()[1] << endl;
    input_dim = params[0][X2I].dim()[1];
  }
  if (hid != params[0][X2I].dim()[0]) {
    cerr << "Warning : LSTMBuilder hidden dimension " << hid
         << " doesn't match with parameter dimension " << params[0][X2I].dim()[0]
         << ". Setting hid to " << params[0][X2I].dim()[0] << endl;
    hid = params[0][X2I].dim()[0];
  }

  h.clear();
  c.clear();
  if (hinit.size() > 0) {
    DYNET_ARG_CHECK(layers * 2 == hinit.size(),
                            "LSTMBuilder must be initialized with 2 times as many expressions as layers "
                            "(hidden state and cell for each layer). However, for " << layers << " layers, "
                            << hinit.size() << " expressions were passed in");
    h0.resize(layers);
    c0.resize(layers);
    for (unsigned i = 0; i < layers; ++i) {
      c0[i] = hinit[i];
      h0[i] = hinit[i + layers];
    }
    has_initial_state = true;
  } else {
    has_initial_state = false;
  }

  dropout_masks_valid = false;
}

void CoupledLSTMBuilder::set_dropout_masks(unsigned batch_size) {
  masks.clear();
  for (unsigned i = 0; i < layers; ++i) {
    std::vector<Expression> masks_i;
    unsigned idim = (i == 0) ? input_dim : hid;
    if (dropout_rate > 0.f || dropout_rate_h > 0.f || dropout_rate_c > 0.f) {
      float retention_rate = 1.f - dropout_rate;
      float retention_rate_h = 1.f - dropout_rate_h;
      float retention_rate_c = 1.f - dropout_rate_c;
      float scale = 1.f / retention_rate;
      float scale_h = 1.f / retention_rate_h;
      float scale_c = 1.f / retention_rate_c;
      // in
      masks_i.push_back(random_bernoulli(*_cg, Dim({ idim}, batch_size), retention_rate, scale));
      // h
      masks_i.push_back(random_bernoulli(*_cg, Dim({ hid}, batch_size), retention_rate_h, scale_h));
      // c
      masks_i.push_back(random_bernoulli(*_cg, Dim({ hid}, batch_size), retention_rate_c, scale_c));
      masks.push_back(masks_i);
    }
  }
  dropout_masks_valid = true;
}

ParameterCollection & CoupledLSTMBuilder::get_parameter_collection() { return local_model; }

// TO DO - Make this correct
// Copied c from the previous step (otherwise c.size()< h.size())
// Also is creating a new step something we want?
// wouldn't overwriting the current one be better?
Expression CoupledLSTMBuilder::set_h_impl(int prev, const vector<Expression>& h_new) {
  DYNET_ARG_CHECK(h_new.empty() || h_new.size() == layers,
                          "LSTMBuilder::set_h expects as many inputs as layers, but got " << h_new.size() << " inputs for " << layers << " layers");
  const unsigned t = h.size();
  h.push_back(vector<Expression>(layers));
  c.push_back(vector<Expression>(layers));
  for (unsigned i = 0; i < layers; ++i) {
    Expression h_i = h_new[i];
    Expression c_i;
      if (t == 0) {
          c_i = dynet::zeros(*(h_new[i].pg), Dim({this->hid}));
      } else {
          c_i = c[t - 1][i];
      }
    h[t][i] = h_i;
    c[t][i] = c_i;
  }
  return h[t].back();
}
// Current implementation : s_new is either {new_c[0],...,new_c[n]}
// or {new_c[0],...,new_c[n],new_h[0],...,new_h[n]}
Expression CoupledLSTMBuilder::set_s_impl(int prev, const std::vector<Expression>& s_new) {
  DYNET_ARG_CHECK(s_new.size() == layers || s_new.size() == 2 * layers,
                          "CoupledLSTMBuilder::set_s expects either as many inputs or twice as many inputs as layers, but got " << s_new.size() << " inputs for " << layers << " layers");
  bool only_c = s_new.size() == layers;
  const unsigned t = c.size();
  h.push_back(vector<Expression>(layers));
  c.push_back(vector<Expression>(layers));
  for (unsigned i = 0; i < layers; ++i) {
    // Initialize h_i
    Expression h_i;
    if (only_c) {
      // If we're not initializing h_i, copy from the previous timestep
      // (or set to 0 if this is the first timestep)
      if (t == 0) {
        h_i = dynet::zeros(*(s_new[i].pg), Dim({this->hid}));
      } else {
        h_i = h[t - 1][i];
      }
    } else {
      // Otherwise set h_i to the given value
      h_i = s_new[i + layers];
    }
    // Initialize c_i
    Expression c_i = s_new[i];
    h[t][i] = h_i;
    c[t][i] = c_i;
  }
  return h[t].back();
}

Expression CoupledLSTMBuilder::add_input_impl(int prev, const Expression& x) {
  h.push_back(vector<Expression>(layers));
  c.push_back(vector<Expression>(layers));
  vector<Expression>& ht = h.back();
  vector<Expression>& ct = c.back();
  Expression in = x;
  if ((dropout_rate > 0.f || dropout_rate_h > 0.f) && !dropout_masks_valid) set_dropout_masks(x.dim().bd);
  for (unsigned i = 0; i < layers; ++i) {
    const vector<Expression>& vars = param_vars[i];
    Expression i_h_tm1, i_c_tm1;
    bool has_prev_state = (prev >= 0 || has_initial_state);
    if (prev < 0) {
      if (has_initial_state) {
        // intial value for h and c at timestep 0 in layer i
        // defaults to zero matrix input if not set in add_parameter_edges
        i_h_tm1 = h0[i];
        i_c_tm1 = c0[i];
      }
    } else {  // t > 0
      i_h_tm1 = h[prev][i];
      i_c_tm1 = c[prev][i];
    }

    // apply dropout according to https://arxiv.org/abs/1512.05287 (tied weights)
    // x
    if (dropout_rate > 0.f) {
      in = cmult(in, masks[i][0]);
    }

    // h
    if (has_prev_state && dropout_rate_h > 0.f)
      i_h_tm1 = cmult(i_h_tm1, masks[i][1]);
    // For c, create another variable since we still need full i_c_tm1 for the componentwise mult
    Expression i_dropped_c_tm1;
    if (has_prev_state) {
      i_dropped_c_tm1 = i_c_tm1;
      if (dropout_rate_c > 0.f)
        i_dropped_c_tm1 = cmult(i_dropped_c_tm1, masks[i][2]);
    }

    // input
    Expression i_ait;
    if (has_prev_state)
      i_ait = affine_transform({vars[BI], vars[X2I], in, vars[H2I], i_h_tm1, vars[C2I], i_dropped_c_tm1});
    else
      i_ait = affine_transform({vars[BI], vars[X2I], in});
    Expression i_it = logistic(i_ait);
    // forget
    Expression i_ft = 1.f - i_it;
    // write memory cell
    Expression i_awt;
    if (has_prev_state)
      i_awt = affine_transform({vars[BC], vars[X2C], in, vars[H2C], i_h_tm1});
    else
      i_awt = affine_transform({vars[BC], vars[X2C], in});
    Expression i_wt = tanh(i_awt);
    // output
    if (has_prev_state) {
      Expression i_nwt = cmult(i_it, i_wt);
      Expression i_crt = cmult(i_ft, i_c_tm1);
      ct[i] = i_crt + i_nwt;
    } else {
      ct[i] = cmult(i_it, i_wt);
    }

    Expression i_aot;
    // Drop c. Uses the same mask as c_tm1. is this justified?
    Expression dropped_c = ct[i];
    if (dropout_rate_c > 0.f)
      dropped_c = cmult(dropped_c, masks[i][2]);
    if (has_prev_state)
      i_aot = affine_transform({vars[BO], vars[X2O], in, vars[H2O], i_h_tm1, vars[C2O], dropped_c});
    else
      i_aot = affine_transform({vars[BO], vars[X2O], in, vars[C2O], dropped_c});
    Expression i_ot = logistic(i_aot);
    Expression ph_t = tanh(ct[i]);
    in = ht[i] = cmult(i_ot, ph_t);
  }
  return ht.back();
}

void CoupledLSTMBuilder::copy(const RNNBuilder & rnn) {
  const CoupledLSTMBuilder & rnn_lstm = (const CoupledLSTMBuilder&)rnn;
  DYNET_ARG_CHECK(params.size() == rnn_lstm.params.size(),
                          "Attempt to copy LSTMBuilder with different number of parameters "
                          "(" << params.size() << " != " << rnn_lstm.params.size() << ")");
  for (size_t i = 0; i < params.size(); ++i)
    for (size_t j = 0; j < params[i].size(); ++j)
      params[i][j] = rnn_lstm.params[i][j];
}

void CoupledLSTMBuilder::set_dropout(float d) {
  DYNET_ARG_CHECK(d >= 0.f && d <= 1.f,
                          "dropout rate must be a probability (>=0 and <=1)");
  dropout_rate = d;
  dropout_rate_h = d;
  dropout_rate_c = d;
}

void CoupledLSTMBuilder::set_dropout(float d, float d_h, float d_c) {
  DYNET_ARG_CHECK(d >= 0.f && d <= 1.f && d_h >= 0.f && d_h <= 1.f && d_c >= 0.f && d_c <= 1.f,
                          "dropout rate must be a probability (>=0 and <=1)");
  dropout_rate = d;
  dropout_rate_h = d_h;
  dropout_rate_c = d_c;
}

void CoupledLSTMBuilder::disable_dropout() {
  dropout_rate = 0.f;
  dropout_rate_h = 0.f;
  dropout_rate_c = 0.f;
}

// Vanilla LSTM

//enum { _X2I, _H2I, _C2I, _BI, _X2F, _H2F, _C2F, _BF, _X2O, _H2O, _C2O, _BO, _X2G, _H2G, _C2G, _BG };
enum { _X2I, _H2I, _BI, _X2F, _H2F, _BF, _X2O, _H2O, _BO, _X2G, _H2G, _BG };
enum { LN_GH, LN_BH, LN_GX, LN_BX, LN_GC, LN_BC};


VanillaLSTMBuilder::VanillaLSTMBuilder() : has_initial_state(false), layers(0), input_dim(0), hid(0), dropout_rate_h(0), ln_lstm(false), forget_bias(1.f), dropout_masks_valid(false) { }

unordered_set<string> VanillaLSTMBuilder::names;

VanillaLSTMBuilder::VanillaLSTMBuilder(unsigned layers,
                                       unsigned input_dim,
                                       unsigned hidden_dim,
                                       ParameterCollection& model,
                                       bool ln_lstm, float forget_bias,
                                       bool blocked, string name) 
: layers(layers), input_dim(input_dim), hid(hidden_dim), ln_lstm(ln_lstm), forget_bias(forget_bias), dropout_masks_valid(false), blocked(blocked), name(name) {
  unsigned layer_input_dim = input_dim;
  local_model = model.add_subcollection("vanilla-lstm-builder");
  for (unsigned i = 0; i < layers; ++i) {
    // i
    Parameter p_x2i = local_model.add_parameters({hidden_dim * 4, layer_input_dim});
    Parameter p_h2i = local_model.add_parameters({hidden_dim * 4, hidden_dim});
    //Parameter p_c2i = model.add_parameters({hidden_dim, hidden_dim});
    Parameter p_bi = local_model.add_parameters({hidden_dim * 4}, ParameterInitConst(0.f));

    layer_input_dim = hidden_dim;  // output (hidden) from 1st layer is input to next

    vector<Parameter> ps = {p_x2i, p_h2i, /*p_c2i,*/ p_bi};
    params.push_back(ps);

    if (ln_lstm){
      Parameter p_gh = model.add_parameters({hidden_dim * 4}, ParameterInitConst(1.f));
      Parameter p_bh = model.add_parameters({hidden_dim * 4}, ParameterInitConst(0.f));
      Parameter p_gx = model.add_parameters({hidden_dim * 4}, ParameterInitConst(1.f));
      Parameter p_bx = model.add_parameters({hidden_dim * 4}, ParameterInitConst(0.f));
      Parameter p_gc = model.add_parameters({hidden_dim}, ParameterInitConst(1.f));
      Parameter p_bc = model.add_parameters({hidden_dim}, ParameterInitConst(0.f));
      vector<Parameter> ln_ps = {p_gh, p_bh, p_gx, p_bx, p_gc, p_bc};
      ln_params.push_back(ln_ps);
    }
  }  // layers
  dropout_rate = 0.f;
  dropout_rate_h = 0.f;


  if (names.count(name))
    cerr << "[WARNING]: define a duplicate VanillaLSTMBuilder named " << name << endl; 
  names.insert(name);

  if (blocked) {
    if (ln_lstm){
      throw std::runtime_error("not implemented!");
    }
    for (bool has_prev_state: {false,true})
    {
      blocks.push_back({});
      for (unsigned layer = 0; layer < layers; layer++){
        params_t param({layer? hidden_dim:input_dim, hidden_dim, ln_lstm, has_prev_state, layer});
        if (bb_gates.count(param) == 0){
          cout << "[VanillaLSTM]: build instance for ";
          cout << "input_dim: " << input_dim << ", hidden_dim:" << hidden_dim << "ln_lstm:" << ln_lstm << ",has_prev_state:" << has_prev_state << ",layer:" << layer << endl;  
          bb_gates[param]= make_shared<OoC::Block>("lstm_update-"+ name 
                                                  + "-" + to_string(has_prev_state)
                                                  + "-" + to_string(layer)
                                                  + "-" + to_string(bb_gates.size()));
          OoC::Block& block = *bb_gates[param];
          vector<Expression> vars, ln_vars;
          for (auto& w: params[layer]) 
            vars.push_back(parameter(block, w));
          if (ln_lstm){
            auto& ln_p = ln_params[layer];
            for (unsigned j = 0; j < ln_p.size(); ++j) { 
              ln_vars.push_back(const_parameter(block, ln_p[j])); 
            }
          }
          block.finish_params();

          Expression i_h_tm1, i_c_tm1, in;
          in = block.placeholder({layer? hidden_dim:input_dim}, "x");
          i_h_tm1 = block.placeholder({hidden_dim}, "h");
          i_c_tm1 = block.placeholder({hidden_dim}, "c");
          block.finish_input();
          // input
          Expression tmp;
          if (ln_lstm){
            if (has_prev_state)
              tmp = vars[_BI] + layer_norm(vars[_X2I] * in, ln_vars[LN_GX], ln_vars[LN_BX]) 
                + layer_norm(vars[_H2I] * i_h_tm1, ln_vars[LN_GH], ln_vars[LN_BH]);
            else
              tmp = vars[_BI] + layer_norm(vars[_X2I] * in, ln_vars[LN_GX], ln_vars[LN_BX]);
          }else{
            if (has_prev_state)
              // tmp = affine_transform({vars[_BI], vars[_X2I], in, vars[_H2I], i_h_tm1});
              tmp = vars[_BI] + matmul(vars[_X2I], in, false) + matmul(vars[_H2I], i_h_tm1, false);
            else
              // tmp = affine_transform({vars[_BI], vars[_X2I], in});
              tmp = vars[_BI] + matmul(vars[_X2I], in, false);
          }          

          Expression i_ait, i_aft, i_aot, i_agt;
          i_ait = pick_range(tmp, 0, hid);
          i_aft = pick_range(tmp, hid, hid * 2);
          i_aot = pick_range(tmp, hid * 2, hid * 3);
          i_agt = pick_range(tmp, hid * 3, hid * 4);
          
          Expression i_it = logistic(i_ait);
          if (forget_bias != 0.0)
              tmp = logistic(i_aft + forget_bias);
          else
              tmp= logistic(i_aft);


          Expression i_ft = tmp;
          Expression i_ot = logistic(i_aot);
          Expression i_gt = tanh(i_agt);

          Expression ct =  has_prev_state ? (cmult(i_ft, i_c_tm1) + cmult(i_it, i_gt)) :  cmult(i_it, i_gt);
          Expression ht;
          if (ln_lstm) {            
            ht = cmult(i_ot, tanh(layer_norm(ct, ln_vars[LN_GC], ln_vars[LN_BC])));
          }
          else ht = cmult(i_ot, tanh(ct));

          block.output({ht, ct});
          block.freeze();
          if (profiling_flag > 1) cout << block.as_string(true);
        }
        blocks.back().push_back(bb_gates[param]);
      }
    }
  }
}


void VanillaLSTMBuilder::new_graph_impl(ComputationGraph& cg, bool update) {
  param_vars.clear();
  if (ln_lstm)ln_param_vars.clear();
  for (unsigned i = 0; i < layers; ++i) {
    auto& p = params[i];
    vector<Expression> vars;
    for (unsigned j = 0; j < p.size(); ++j) { vars.push_back(update ? parameter(cg, p[j]) : const_parameter(cg, p[j])); }
    param_vars.push_back(vars);
    if (ln_lstm){
      auto& ln_p = ln_params[i];
      vector<Expression> ln_vars;
      for (unsigned j = 0; j < ln_p.size(); ++j) { ln_vars.push_back(update ? parameter(cg, ln_p[j]) : const_parameter(cg, ln_p[j])); }
      ln_param_vars.push_back(ln_vars);
    }
  }

  _cg = &cg;
}
// layout: 0..layers = c
//         layers+1..2*layers = h
void VanillaLSTMBuilder::start_new_sequence_impl(const vector<Expression>& hinit) {
  h.clear();
  c.clear();

  if (hinit.size() > 0) {
    DYNET_ARG_CHECK(layers * 2 == hinit.size(),
                            "VanillaLSTMBuilder must be initialized with 2 times as many expressions as layers "
                            "(hidden state, and cell for each layer). However, for " << layers << " layers, " <<
                            hinit.size() << " expressions were passed in");
    h0.resize(layers);
    c0.resize(layers);
    for (unsigned i = 0; i < layers; ++i) {
      c0[i] = hinit[i];
      h0[i] = hinit[i + layers];
    }
    has_initial_state = true;
  } else {
    has_initial_state = false;
  }

  dropout_masks_valid = false;
}

void VanillaLSTMBuilder::set_dropout_masks(unsigned batch_size) {
  masks.clear();
  for (unsigned i = 0; i < layers; ++i) {
    std::vector<Expression> masks_i;
    unsigned idim = (i == 0) ? input_dim : hid;
    if (dropout_rate > 0.f || dropout_rate_h > 0.f) {
      float retention_rate = 1.f - dropout_rate;
      float retention_rate_h = 1.f - dropout_rate_h;
      float scale = 1.f / retention_rate;
      float scale_h = 1.f / retention_rate_h;
      // in
      masks_i.push_back(random_bernoulli(*_cg, Dim({ idim}, batch_size), retention_rate, scale));
      // h
      masks_i.push_back(random_bernoulli(*_cg, Dim({ hid}, batch_size), retention_rate_h, scale_h));
      masks.push_back(masks_i);
    }
  }
  dropout_masks_valid = true;
}

ParameterCollection & VanillaLSTMBuilder::get_parameter_collection() {
  return local_model;
}

// TODO - Make this correct
// Copied c from the previous step (otherwise c.size()< h.size())
// Also is creating a new step something we want?
// wouldn't overwriting the current one be better?
Expression VanillaLSTMBuilder::set_h_impl(int prev, const vector<Expression>& h_new) {
  DYNET_ARG_CHECK(h_new.empty() || h_new.size() == layers,
                          "VanillaLSTMBuilder::set_h expects as many inputs as layers, but got " <<
                          h_new.size() << " inputs for " << layers << " layers");
  const unsigned t = h.size();
  h.push_back(vector<Expression>(layers));
  c.push_back(vector<Expression>(layers));
  for (unsigned i = 0; i < layers; ++i) {
    Expression h_i = h_new[i];
    Expression c_i;
      if (t == 0) {
          c_i = dynet::zeros(*(h_new[i].pg), Dim({this->hid}));
      } else {
          c_i = c[t - 1][i];
      }
    h[t][i] = h_i;
    c[t][i] = c_i;
  }
  return h[t].back();
}
// Current implementation : s_new is either {new_c[0],...,new_c[n]}
// or {new_c[0],...,new_c[n],new_h[0],...,new_h[n]}
Expression VanillaLSTMBuilder::set_s_impl(int prev, const std::vector<Expression>& s_new) {
  DYNET_ARG_CHECK(s_new.size() == layers || s_new.size() == 2 * layers,
                          "VanillaLSTMBuilder::set_s expects either as many inputs or twice as many inputs as layers, but got " << s_new.size() << " inputs for " << layers << " layers");
  bool only_c = s_new.size() == layers;
  const unsigned t = c.size();
  h.push_back(vector<Expression>(layers));
  c.push_back(vector<Expression>(layers));
  for (unsigned i = 0; i < layers; ++i) {
    // Initialize h_i
    Expression h_i;
    if (only_c) {
      // If we're not initializing h_i, copy from the previous timestep
      // (or set to 0 if this is the first timestep)
      if (t == 0) {
        h_i = dynet::zeros(*(s_new[i].pg), Dim({this->hid}));
      } else {
        h_i = h[t - 1][i];
      }
    } else {
      // Otherwise set h_i to the given value
      h_i = s_new[i + layers];
    }
    // Initialize c_i
    Expression c_i = s_new[i];
    h[t][i] = h_i;
    c[t][i] = c_i;
  }
  return h[t].back();
}

Expression VanillaLSTMBuilder::add_input_impl(int prev, const Expression& x) {
  h.push_back(vector<Expression>(layers));
  c.push_back(vector<Expression>(layers));
  vector<Expression>& ht = h.back();
  vector<Expression>& ct = c.back();
  Expression in = x;
  if ((dropout_rate > 0.f || dropout_rate_h > 0.f) && !dropout_masks_valid) set_dropout_masks(x.dim().bd);
  for (unsigned i = 0; i < layers; ++i) {
    Expression i_h_tm1, i_c_tm1;
    bool has_prev_state = (prev >= 0 || has_initial_state);
    if (prev < 0) {
      if (has_initial_state) {
        // intial value for h and c at timestep 0 in layer i
        // defaults to zero matrix input if not set in add_parameter_edges
        i_h_tm1 = h0[i];
        i_c_tm1 = c0[i];
      }
    } else {  // t > 0
      i_h_tm1 = h[prev][i];
      i_c_tm1 = c[prev][i];
    }
    // apply dropout according to https://arxiv.org/abs/1512.05287 (tied weights)
    if (dropout_rate > 0.f) {
      assert(false);
      in = cmult(in, masks[i][0]);
    }
    if (has_prev_state && dropout_rate_h > 0.f){
      assert(false);
      i_h_tm1 = cmult(i_h_tm1, masks[i][1]);
    }
    
    // input
    
    if (dynet::blocked){
      Expression o = blocks[has_prev_state][i]->operator()(
        _cg, {{"x", in}, {"h", i_h_tm1}, {"c", i_c_tm1}}, {});
      in = ht[i] = o[0], ct[i] = o[1];
    }
    else {
      const vector<Expression>& vars = param_vars[i];
      Expression tmp;
      if (ln_lstm){
        const vector<Expression>& ln_vars = ln_param_vars[i];
        if (has_prev_state)
          tmp = vars[_BI] + layer_norm(vars[_X2I] * in, ln_vars[LN_GX], ln_vars[LN_BX]) + layer_norm(vars[_H2I] * i_h_tm1, ln_vars[LN_GH], ln_vars[LN_BH]);
        else
          tmp = vars[_BI] + layer_norm(vars[_X2I] * in, ln_vars[LN_GX], ln_vars[LN_BX]);
      }else{
        if (has_prev_state)
          // tmp = affine_transform({vars[_BI], vars[_X2I], in, vars[_H2I], i_h_tm1});
          tmp = vars[_BI] + matmul(vars[_X2I], in) + matmul(vars[_H2I], i_h_tm1);
        else
          // tmp = affine_transform({vars[_BI], vars[_X2I], in});
          tmp = vars[_BI] + matmul(vars[_X2I], in);
      }
      Expression i_ait;
      Expression i_aft;
      Expression i_aot;
      Expression i_agt;
      i_ait = pick_range(tmp, 0, hid);
      i_aft = pick_range(tmp, hid, hid * 2);
      i_aot = pick_range(tmp, hid * 2, hid * 3);
      i_agt = pick_range(tmp, hid * 3, hid * 4);
      
      Expression i_it = logistic(i_ait);
      if (forget_bias != 0.0)
          tmp = logistic(i_aft + forget_bias);
      else
          tmp= logistic(i_aft);


      Expression i_ft = tmp;
      Expression i_ot = logistic(i_aot);
      Expression i_gt = tanh(i_agt);

      ct[i] = has_prev_state ? (cmult(i_ft, i_c_tm1) + cmult(i_it, i_gt)) :  cmult(i_it, i_gt);
      if (ln_lstm) {
        const vector<Expression>& ln_vars = ln_param_vars[i];
        in = ht[i] = cmult(i_ot, tanh(layer_norm(ct[i], ln_vars[LN_GC], ln_vars[LN_BC])));
      } else
        in = ht[i] = cmult(i_ot, tanh(ct[i]));
    }
  }
  return ht.back();
}

void VanillaLSTMBuilder::copy(const RNNBuilder & rnn) {
  const VanillaLSTMBuilder & rnn_lstm = (const VanillaLSTMBuilder&)rnn;
  DYNET_ARG_CHECK(params.size() == rnn_lstm.params.size(),
                          "Attempt to copy VanillaLSTMBuilder with different number of parameters "
                          "(" << params.size() << " != " << rnn_lstm.params.size() << ")");
  for (size_t i = 0; i < params.size(); ++i)
    for (size_t j = 0; j < params[i].size(); ++j)
      params[i][j] = rnn_lstm.params[i][j];
  for (size_t i = 0; i < ln_params.size(); ++i)
    for (size_t j = 0; j < ln_params[i].size(); ++j)
      ln_params[i][j] = rnn_lstm.ln_params[i][j];
}

void VanillaLSTMBuilder::set_dropout(float d) {
  DYNET_ARG_CHECK(d >= 0.f && d <= 1.f,
                          "dropout rate must be a probability (>=0 and <=1)");
  dropout_rate = d;
  dropout_rate_h = d;
}

void VanillaLSTMBuilder::set_dropout(float d, float d_h) {
  DYNET_ARG_CHECK(d >= 0.f && d <= 1.f && d_h >= 0.f && d_h <= 1.f,
                          "dropout rate must be a probability (>=0 and <=1)");
  dropout_rate = d;
  dropout_rate_h = d_h;
}

void VanillaLSTMBuilder::disable_dropout() {
  dropout_rate = 0.f;
  dropout_rate_h = 0.f;
}


SparseLSTMBuilder::SparseLSTMBuilder() : has_initial_state(false), layers(0), input_dim(0), hid(0), dropout_rate_h(0), ln_lstm(false), forget_bias(1.f), dropout_masks_valid(false) { }

SparseLSTMBuilder::SparseLSTMBuilder(unsigned layers,
                                       unsigned input_dim,
                                       unsigned hidden_dim,
                                       ParameterCollection& model,
                                       bool ln_lstm, float forget_bias) : layers(layers), input_dim(input_dim), hid(hidden_dim), ln_lstm(ln_lstm), forget_bias(forget_bias), dropout_masks_valid(false) {
  unsigned layer_input_dim = input_dim;
  local_model = model.add_subcollection("vanilla-lstm-builder");
  for (unsigned i = 0; i < layers; ++i) {
    // i
    Parameter p_x2i = local_model.add_parameters({hidden_dim * 4, layer_input_dim});
    Parameter p_h2i = local_model.add_parameters({hidden_dim * 4, hidden_dim});
    Parameter p_x2i_mask = local_model.add_parameters({hidden_dim * 4, layer_input_dim}, ParameterInitConst(1.0f));
    Parameter p_h2i_mask = local_model.add_parameters({hidden_dim * 4, hidden_dim}, ParameterInitConst(1.0f));
    //Parameter p_c2i = model.add_parameters({hidden_dim, hidden_dim});
    Parameter p_bi = local_model.add_parameters({hidden_dim * 4}, ParameterInitConst(0.f));

    layer_input_dim = hidden_dim;  // output (hidden) from 1st layer is input to next

    vector<Parameter> ps = {p_x2i, p_h2i, /*p_c2i,*/ p_bi, p_x2i_mask, p_h2i_mask};
    params.push_back(ps);

    if (ln_lstm){
      Parameter p_gh = model.add_parameters({hidden_dim * 4}, ParameterInitConst(1.f));
      Parameter p_bh = model.add_parameters({hidden_dim * 4}, ParameterInitConst(0.f));
      Parameter p_gx = model.add_parameters({hidden_dim * 4}, ParameterInitConst(1.f));
      Parameter p_bx = model.add_parameters({hidden_dim * 4}, ParameterInitConst(0.f));
      Parameter p_gc = model.add_parameters({hidden_dim}, ParameterInitConst(1.f));
      Parameter p_bc = model.add_parameters({hidden_dim}, ParameterInitConst(0.f));
      vector<Parameter> ln_ps = {p_gh, p_bh, p_gx, p_bx, p_gc, p_bc};
      ln_params.push_back(ln_ps);
    }
  }  // layers
  dropout_rate = 0.f;
  dropout_rate_h = 0.f;
}

struct WEIGHT_MAGNITUDE{
  float value;
  int layer_index=0;
  int params_ofs=0;
  int index=0;
} ;

bool mag_compare(const WEIGHT_MAGNITUDE &a, const WEIGHT_MAGNITUDE &b) { return a.value < b.value; }

void SparseLSTMBuilder::set_sparsity(float percent){
    cout<<"Setting sparsity level at "<<percent<<"%\n";
    int total_parameters=0;
    for (unsigned i = 0; i < layers; ++i) {
      const vector<Parameter>& vars = params[i];
      total_parameters+=vars[_BI+1].dim().cols()*vars[_BI+1].dim().rows();
      total_parameters+=vars[_BI+2].dim().cols()*vars[_BI+2].dim().rows();
    }
    cout<<"\tTotal number of parameters is "<<total_parameters<<"\n";
    int prune_count=(int)(percent*total_parameters);
    cout<<"\tDesired number of parameters is "<<total_parameters-prune_count<<"\n";

    vector<WEIGHT_MAGNITUDE> magnitudes;

    vector<vector<float>> new_mask_1;
    vector<vector<float>> new_mask_2;

    for (unsigned i = 0; i < layers; ++i) {

      vector<float> tmp_mask_1;
      vector<float> tmp_mask_2;

      vector<Parameter>& vars = params[i];
      for (int ofs=0; ofs<2; ofs++){
        Tensor *weight_values_tensor=vars[_X2I+ofs].values();
        Tensor *mask_values_tensor=vars[_BI+ofs+1].values();
        vector<float> weight_values=as_vector(weight_values_tensor[0]);
        vector<float> mask_values=as_vector(mask_values_tensor[0]);

        WEIGHT_MAGNITUDE wm;
        int size=vars[_BI+ofs+1].dim().cols()*vars[_BI+1+ofs].dim().rows();
        for (int ii=0;ii<size;ii++){
          wm.value=std::abs(weight_values[ii]*mask_values[ii]);
          wm.layer_index=i;
          wm.params_ofs=ofs;
          wm.index=ii;
          magnitudes.push_back(wm);
          if (ofs==0){
            tmp_mask_1.push_back(1.0f);
          }else{
            tmp_mask_2.push_back(1.0f);
          }
        }
      }
      new_mask_1.push_back(tmp_mask_1);
      new_mask_2.push_back(tmp_mask_2);
    }

    std::sort(magnitudes.begin(), magnitudes.end(), mag_compare);

    for (int ii=0;ii<prune_count;ii++){
      if (magnitudes[ii].params_ofs ==0){
        new_mask_1[magnitudes[ii].layer_index][magnitudes[ii].index]=0.0f;
      }else{
        new_mask_2[magnitudes[ii].layer_index][magnitudes[ii].index]=0.0f;
      }
    }

    for (int i=0;i<layers;i++){
      vector<Parameter>& vars = params[i];
      vars[_BI+1].set_value(new_mask_1[i]);
      vars[_BI+2].set_value(new_mask_2[i]);
    }
}

void SparseLSTMBuilder::new_graph_impl(ComputationGraph& cg, bool update) {
  param_vars.clear();
  if (ln_lstm)ln_param_vars.clear();
  for (unsigned i = 0; i < layers; ++i) {
    auto& p = params[i];
    vector<Expression> vars;
    vars.push_back(update ? cmult(parameter(cg, p[0]), const_parameter(cg, p[3])) : cmult(const_parameter(cg, p[0]), const_parameter(cg, p[3])));
    vars.push_back(update ? cmult(parameter(cg, p[1]), const_parameter(cg, p[4])) : cmult(const_parameter(cg, p[1]), const_parameter(cg, p[4])));
    vars.push_back(update ? parameter(cg, p[2]): const_parameter(cg, p[2]));
    param_vars.push_back(vars);
    if (ln_lstm){
      auto& ln_p = ln_params[i];
      vector<Expression> ln_vars;
      for (unsigned j = 0; j < ln_p.size(); ++j) { ln_vars.push_back(update ? parameter(cg, ln_p[j]) : const_parameter(cg, ln_p[j])); }
      ln_param_vars.push_back(ln_vars);
    }
  }

  _cg = &cg;
}
// layout: 0..layers = c
//         layers+1..2*layers = h
void SparseLSTMBuilder::start_new_sequence_impl(const vector<Expression>& hinit) {
  h.clear();
  c.clear();

  if (hinit.size() > 0) {
    DYNET_ARG_CHECK(layers * 2 == hinit.size(),
                            "SparseLSTMBuilder must be initialized with 2 times as many expressions as layers "
                            "(hidden state, and cell for each layer). However, for " << layers << " layers, " <<
                            hinit.size() << " expressions were passed in");
    h0.resize(layers);
    c0.resize(layers);
    for (unsigned i = 0; i < layers; ++i) {
      c0[i] = hinit[i];
      h0[i] = hinit[i + layers];
    }
    has_initial_state = true;
  } else {
    has_initial_state = false;
  }

  dropout_masks_valid = false;
}

void SparseLSTMBuilder::set_dropout_masks(unsigned batch_size) {
  masks.clear();
  for (unsigned i = 0; i < layers; ++i) {
    std::vector<Expression> masks_i;
    unsigned idim = (i == 0) ? input_dim : hid;
    if (dropout_rate > 0.f || dropout_rate_h > 0.f) {
      float retention_rate = 1.f - dropout_rate;
      float retention_rate_h = 1.f - dropout_rate_h;
      float scale = 1.f / retention_rate;
      float scale_h = 1.f / retention_rate_h;
      // in
      masks_i.push_back(random_bernoulli(*_cg, Dim({ idim}, batch_size), retention_rate, scale));
      // h
      masks_i.push_back(random_bernoulli(*_cg, Dim({ hid}, batch_size), retention_rate_h, scale_h));
      masks.push_back(masks_i);
    }
  }
  dropout_masks_valid = true;
}

ParameterCollection & SparseLSTMBuilder::get_parameter_collection() {
  return local_model;
}

// TODO - Make this correct
// Copied c from the previous step (otherwise c.size()< h.size())
// Also is creating a new step something we want?
// wouldn't overwriting the current one be better?
Expression SparseLSTMBuilder::set_h_impl(int prev, const vector<Expression>& h_new) {
  DYNET_ARG_CHECK(h_new.empty() || h_new.size() == layers,
                          "SparseLSTMBuilder::set_h expects as many inputs as layers, but got " <<
                          h_new.size() << " inputs for " << layers << " layers");
  const unsigned t = h.size();
  h.push_back(vector<Expression>(layers));
  c.push_back(vector<Expression>(layers));
  for (unsigned i = 0; i < layers; ++i) {
    Expression h_i = h_new[i];
    Expression c_i = c[t - 1][i];
    h[t][i] = h_i;
    c[t][i] = c_i;
  }
  return h[t].back();
}
// Current implementation : s_new is either {new_c[0],...,new_c[n]}
// or {new_c[0],...,new_c[n],new_h[0],...,new_h[n]}
Expression SparseLSTMBuilder::set_s_impl(int prev, const std::vector<Expression>& s_new) {
  DYNET_ARG_CHECK(s_new.size() == layers || s_new.size() == 2 * layers,
                          "SparseLSTMBuilder::set_s expects either as many inputs or twice as many inputs as layers, but got " << s_new.size() << " inputs for " << layers << " layers");
  bool only_c = s_new.size() == layers;
  const unsigned t = c.size();
  h.push_back(vector<Expression>(layers));
  c.push_back(vector<Expression>(layers));
  for (unsigned i = 0; i < layers; ++i) {
    Expression h_i = only_c ? h[t - 1][i] : s_new[i + layers];
    Expression c_i = s_new[i];
    h[t][i] = h_i;
    c[t][i] = c_i;
  }
  return h[t].back();
}

Expression SparseLSTMBuilder::add_input_impl(int prev, const Expression& x) {
  h.push_back(vector<Expression>(layers));
  c.push_back(vector<Expression>(layers));
  vector<Expression>& ht = h.back();
  vector<Expression>& ct = c.back();
  Expression in = x;
  if ((dropout_rate > 0.f || dropout_rate_h > 0.f) && !dropout_masks_valid) set_dropout_masks(x.dim().bd);
  for (unsigned i = 0; i < layers; ++i) {
    const vector<Expression>& vars = param_vars[i];

    Expression i_h_tm1, i_c_tm1;
    bool has_prev_state = (prev >= 0 || has_initial_state);
    if (prev < 0) {
      if (has_initial_state) {
        // intial value for h and c at timestep 0 in layer i
        // defaults to zero matrix input if not set in add_parameter_edges
        i_h_tm1 = h0[i];
        i_c_tm1 = c0[i];
      }
    } else {  // t > 0
      i_h_tm1 = h[prev][i];
      i_c_tm1 = c[prev][i];
    }
    // apply dropout according to https://arxiv.org/abs/1512.05287 (tied weights)
    if (dropout_rate > 0.f) {
      in = cmult(in, masks[i][0]);
    }
    if (has_prev_state && dropout_rate_h > 0.f)
      i_h_tm1 = cmult(i_h_tm1, masks[i][1]);
    // input
    Expression tmp;
    Expression i_ait;
    Expression i_aft;
    Expression i_aot;
    Expression i_agt;
    if (ln_lstm){
      const vector<Expression>& ln_vars = ln_param_vars[i];
      if (has_prev_state)
        tmp = vars[_BI] + layer_norm(vars[_X2I] *in, ln_vars[LN_GX], ln_vars[LN_BX]) + layer_norm(vars[_H2I] * i_h_tm1, ln_vars[LN_GH], ln_vars[LN_BH]);
      else
        tmp = vars[_BI] + layer_norm(vars[_X2I] * in, ln_vars[LN_GX], ln_vars[LN_BX]);
    }else{
      if (has_prev_state){
        tmp = affine_transform({vars[_BI], vars[_X2I], in, vars[_H2I], i_h_tm1});
      }else{
        tmp = affine_transform({vars[_BI], vars[_X2I], in});
      }
    }
    i_ait = pick_range(tmp, 0, hid);
    i_aft = pick_range(tmp, hid, hid * 2);
    i_aot = pick_range(tmp, hid * 2, hid * 3);
    i_agt = pick_range(tmp, hid * 3, hid * 4);
    Expression i_it = logistic(i_ait);
    if (forget_bias != 0.0)
        tmp = logistic(i_aft + forget_bias);
    else
        tmp= logistic(i_aft);

    Expression i_ft = tmp;
    Expression i_ot = logistic(i_aot);
    Expression i_gt = tanh(i_agt);

    ct[i] = has_prev_state ? (cmult(i_ft, i_c_tm1) + cmult(i_it, i_gt)) :  cmult(i_it, i_gt);
    if (ln_lstm) {
      const vector<Expression>& ln_vars = ln_param_vars[i];
      in = ht[i] = cmult(i_ot, tanh(layer_norm(ct[i], ln_vars[LN_GC], ln_vars[LN_BC])));
    } else
      in = ht[i] = cmult(i_ot, tanh(ct[i]));
  }
  return ht.back();
}

void SparseLSTMBuilder::copy(const RNNBuilder & rnn) {
  const SparseLSTMBuilder & rnn_lstm = (const SparseLSTMBuilder&)rnn;
  DYNET_ARG_CHECK(params.size() == rnn_lstm.params.size(),
                          "Attempt to copy SparseLSTMBuilder with different number of parameters "
                          "(" << params.size() << " != " << rnn_lstm.params.size() << ")");
  for (size_t i = 0; i < params.size(); ++i)
    for (size_t j = 0; j < params[i].size(); ++j)
      params[i][j] = rnn_lstm.params[i][j];
  for (size_t i = 0; i < ln_params.size(); ++i)
    for (size_t j = 0; j < ln_params[i].size(); ++j)
      ln_params[i][j] = rnn_lstm.ln_params[i][j];
}

void SparseLSTMBuilder::set_dropout(float d) {
  DYNET_ARG_CHECK(d >= 0.f && d <= 1.f,
                          "dropout rate must be a probability (>=0 and <=1)");
  dropout_rate = d;
  dropout_rate_h = d;
}

void SparseLSTMBuilder::set_dropout(float d, float d_h) {
  DYNET_ARG_CHECK(d >= 0.f && d <= 1.f && d_h >= 0.f && d_h <= 1.f,
                          "dropout rate must be a probability (>=0 and <=1)");
  dropout_rate = d;
  dropout_rate_h = d_h;
}

void SparseLSTMBuilder::disable_dropout() {
  dropout_rate = 0.f;
  dropout_rate_h = 0.f;
}

CompactVanillaLSTMBuilder::CompactVanillaLSTMBuilder() : has_initial_state(false), layers(0), input_dim(0), hid(0), dropout_rate_h(0), weightnoise_std(0), dropout_masks_valid(false) { }

CompactVanillaLSTMBuilder::CompactVanillaLSTMBuilder(unsigned layers,
                 unsigned input_dim,
                 unsigned hidden_dim,
                 ParameterCollection& model)
      : layers(layers), input_dim(input_dim), hid(hidden_dim), weightnoise_std(0), dropout_masks_valid(false){
  unsigned layer_input_dim = input_dim;
  local_model = model.add_subcollection("compact-vanilla-lstm-builder");
  for (unsigned i = 0; i < layers; ++i) {
    // i
    Parameter p_Wx = local_model.add_parameters({hidden_dim * 4, layer_input_dim});
    Parameter p_Wh = local_model.add_parameters({hidden_dim * 4, hidden_dim});
    Parameter p_b = local_model.add_parameters({hidden_dim * 4}, ParameterInitConst(0.f));

    layer_input_dim = hidden_dim;  // output (hidden) from 1st layer is input to next

    vector<Parameter> ps = {p_Wx, p_Wh, p_b};
    params.push_back(ps);

  }  // layers
  dropout_rate = 0.f;
  dropout_rate_h = 0.f;
}

void CompactVanillaLSTMBuilder::new_graph_impl(ComputationGraph& cg, bool update) {
  param_vars.clear();
  for (unsigned i = 0; i < layers; ++i) {
    auto& p = params[i];
    vector<Expression> vars;
    for (unsigned j = 0; j < p.size(); ++j) { vars.push_back(update ? parameter(cg, p[j]) : const_parameter(cg, p[j])); }
    param_vars.push_back(vars);
  }

  _cg = &cg;
}
// layout: 0..layers = c
//         layers+1..2*layers = h
void CompactVanillaLSTMBuilder::start_new_sequence_impl(const vector<Expression>& hinit) {
  h.clear();
  c.clear();

  if (hinit.size() > 0) {
    DYNET_ARG_CHECK(layers * 2 == hinit.size(),
                            "CompactVanillaLSTMBuilder must be initialized with 2 times as many expressions as layers "
                            "(hidden state, and cell for each layer). However, for " << layers << " layers, " <<
                            hinit.size() << " expressions were passed in");
    h0.resize(layers);
    c0.resize(layers);
    for (unsigned i = 0; i < layers; ++i) {
      c0[i] = hinit[i];
      h0[i] = hinit[i + layers];
    }
    has_initial_state = true;
  } else {
    has_initial_state = false;
  }

  dropout_masks_valid = false;
}

void CompactVanillaLSTMBuilder::set_dropout_masks(unsigned batch_size) {
  masks.clear();
  for (unsigned i = 0; i < layers; ++i) {
    std::vector<Expression> masks_i;
    unsigned idim = (i == 0) ? input_dim : hid;
    if (dropout_rate > 0.f || dropout_rate_h > 0.f) {
      float retention_rate = 1.f - dropout_rate;
      float retention_rate_h = 1.f - dropout_rate_h;
      float scale = 1.f / retention_rate;
      float scale_h = 1.f / retention_rate_h;
      // in
      masks_i.push_back(random_bernoulli(*_cg, Dim({ idim}, batch_size), retention_rate, scale));
      // h
      masks_i.push_back(random_bernoulli(*_cg, Dim({ hid}, batch_size), retention_rate_h, scale_h));
      masks.push_back(masks_i);
    }
  }
  dropout_masks_valid = true;
}

ParameterCollection & CompactVanillaLSTMBuilder::get_parameter_collection() {
  return local_model;
}

// TODO - Make this correct
// Copied c from the previous step (otherwise c.size()< h.size())
// Also is creating a new step something we want?
// wouldn't overwriting the current one be better?
Expression CompactVanillaLSTMBuilder::set_h_impl(int prev, const vector<Expression>& h_new) {
  DYNET_ARG_CHECK(h_new.empty() || h_new.size() == layers,
                          "CompactVanillaLSTMBuilder::set_h expects as many inputs as layers, but got " <<
                          h_new.size() << " inputs for " << layers << " layers");
  const unsigned t = h.size();
  h.push_back(vector<Expression>(layers));
  c.push_back(vector<Expression>(layers));
  for (unsigned i = 0; i < layers; ++i) {
    Expression h_i = h_new[i];
    Expression c_i;
      if (t == 0) {
          c_i = dynet::zeros(*(h_new[i].pg), Dim({this->hid}));
      } else {
          c_i = c[t - 1][i];
      }
    h[t][i] = h_i;
    c[t][i] = c_i;
  }
  return h[t].back();
}
// Current implementation : s_new is either {new_c[0],...,new_c[n]}
// or {new_c[0],...,new_c[n],new_h[0],...,new_h[n]}
Expression CompactVanillaLSTMBuilder::set_s_impl(int prev, const std::vector<Expression>& s_new) {
  DYNET_ARG_CHECK(s_new.size() == layers || s_new.size() == 2 * layers,
                          "CompactVanillaLSTMBuilder::set_s expects either as many inputs or twice as many inputs as layers, but got " << s_new.size() << " inputs for " << layers << " layers");
  bool only_c = s_new.size() == layers;
  const unsigned t = c.size();
  h.push_back(vector<Expression>(layers));
  c.push_back(vector<Expression>(layers));
  for (unsigned i = 0; i < layers; ++i) {
    // Initialize h_i
    Expression h_i;
    if (only_c) {
      // If we're not initializing h_i, copy from the previous timestep
      // (or set to 0 if this is the first timestep)
      if (t == 0) {
        h_i = dynet::zeros(*(s_new[i].pg), Dim({this->hid}));
      } else {
        h_i = h[t - 1][i];
      }
    } else {
      // Otherwise set h_i to the given value
      h_i = s_new[i + layers];
    }
    // Initialize c_i
    Expression c_i = s_new[i];
    h[t][i] = h_i;
    c[t][i] = c_i;
  }
  return h[t].back();
}

Expression CompactVanillaLSTMBuilder::add_input_impl(int prev, const Expression& x) {
  h.push_back(vector<Expression>(layers));
  c.push_back(vector<Expression>(layers));
  vector<Expression>& ht = h.back();
  vector<Expression>& ct = c.back();
  Expression in = x;
  if ((dropout_rate > 0.f || dropout_rate_h > 0.f) && !dropout_masks_valid) set_dropout_masks(x.dim().bd);
  for (unsigned i = 0; i < layers; ++i) {
    const vector<Expression>& vars = param_vars[i];
    Expression i_h_tm1, i_c_tm1;
    if (prev < 0) {
      if (has_initial_state) {
        // initial value for h and c at timestep 0 in layer i
        // defaults to zero matrix input if not set in add_parameter_edges
        i_h_tm1 = h0[i];
        i_c_tm1 = c0[i];
      } else {
  i_h_tm1 = zeros(*_cg, Dim({vars[_BI].dim()[0]/4}, x.dim().bd));
  i_c_tm1 = i_h_tm1;
      }
    } else {  // t > 0
      i_h_tm1 = h[prev][i];
      i_c_tm1 = c[prev][i];
    }
    if (dropout_rate > 0.f || dropout_rate_h > 0.f){
      // apply dropout according to https://arxiv.org/abs/1512.05287 (tied weights)
      Expression gates_t = vanilla_lstm_gates_dropout({in}, i_h_tm1, vars[_X2I], vars[_H2I], vars[_BI], masks[i][0], masks[i][1], weightnoise_std);
      ct[i] = vanilla_lstm_c(i_c_tm1, gates_t);
      in = ht[i] = vanilla_lstm_h(ct[i], gates_t);
    } else {
      Expression gates_t = vanilla_lstm_gates({in}, i_h_tm1, vars[_X2I], vars[_H2I], vars[_BI], weightnoise_std);
      ct[i] = vanilla_lstm_c(i_c_tm1, gates_t);
      in = ht[i] = vanilla_lstm_h(ct[i], gates_t);
    }
  }
  return ht.back();
}

void CompactVanillaLSTMBuilder::copy(const RNNBuilder & rnn) {
  const CompactVanillaLSTMBuilder & rnn_lstm = (const CompactVanillaLSTMBuilder&)rnn;
  DYNET_ARG_CHECK(params.size() == rnn_lstm.params.size(),
                          "Attempt to copy CompactVanillaLSTMBuilder with different number of parameters "
                          "(" << params.size() << " != " << rnn_lstm.params.size() << ")");
  for (size_t i = 0; i < params.size(); ++i)
    for (size_t j = 0; j < params[i].size(); ++j)
      params[i][j] = rnn_lstm.params[i][j];
}

void CompactVanillaLSTMBuilder::set_dropout(float d) {
  DYNET_ARG_CHECK(d >= 0.f && d <= 1.f,
                          "dropout rate must be a probability (>=0 and <=1)");
  dropout_rate = d;
  dropout_rate_h = d;
}

void CompactVanillaLSTMBuilder::set_dropout(float d, float d_h) {
  DYNET_ARG_CHECK(d >= 0.f && d <= 1.f && d_h >= 0.f && d_h <= 1.f,
                          "dropout rate must be a probability (>=0 and <=1)");
  dropout_rate = d;
  dropout_rate_h = d_h;
}

void CompactVanillaLSTMBuilder::disable_dropout() {
  dropout_rate = 0.f;
  dropout_rate_h = 0.f;
}
void CompactVanillaLSTMBuilder::set_weightnoise(float std) {
  DYNET_ARG_CHECK(std >= 0.f, "weight noise must have standard deviation >=0");
  weightnoise_std = std;
}

} // namespace dynet
