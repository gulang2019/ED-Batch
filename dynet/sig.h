#ifndef DYNET_SIG_H
#define DYNET_SIG_H

#define DYNET_MAX_SIG 100

#include <vector>
#include <algorithm>
#include <unordered_map>
#include <map>
#include <cstddef>
#include <string>
#include <cstring>
#include <iostream>

#include <dynet/dim.h>

namespace dynet {

  namespace nt {
    // TODO: synchronize with type2name
    enum NodeType {
      tanh=1, sqrt, abs, erf, square, cube, exp, logsigmoid, loggamma, log, nobackprop, scalegradient, identity, negate, rectify, logistic, softsign, silu, round, ceiling, floor,
      sinh, cosh, asinh, acosh, atanh, sin, cos, tan, asin, acos, atan, plus_const, minus_const, concat, cmult, csum, sum, reduce, squared_distance, softmax, pnls, pickrange, scalar_mult, dropout,
      input, scalar_input, lookup, select, argmax_index,
      COMPLEX,
      affine, matmul, transpose,
      vanilla_lstm_gates, vanilla_lstm_h, vanilla_lstm_c,
      conv2d, block, get
    };

    extern std::unordered_map<int, std::string> type2name;
  }

struct SigYoav {
  SigYoav(short which) : which(which), nn(0), nd(0) { }
  SigYoav() : which(0), nn(0), nd(0) { }
  const unsigned short which;
  unsigned short nn;
  unsigned short nd;
  Dim dims[10];
  unsigned node_ids[10];

  void add_node(unsigned i) { node_ids[nn++]=i; }
  // TODO add_dim is NOT SAME as dim.print_profile(oss)
  void add_dim(const Dim &d)           { dims[nd++]=d;     }
};

inline bool operator==(const SigYoav& a, const SigYoav& b) {
  if (a.which == b.which && a.nn == b.nn && a.nd == b.nd) {
    for (int i = 0; i < a.nn; ++i) { if (a.node_ids[i] != b.node_ids[i]) return false; }
    for (int i = 0; i < a.nd; ++i) { if (a.dims[i] != b.dims[i]) return false; }
    return true;
  } else {
    return false;
  }
}

struct SigString {
  SigString(int which) : which(which), tail(data) { }
  SigString() : which(0), tail(data) { }
  int which;
  int data[DYNET_MAX_SIG];
  int* tail;

  void add_node(unsigned i) { *(tail++) = -(int)i; }
  void add_dim(const Dim &d) {
    *(tail++) = -(int)d.nd;
    memcpy(tail, d.d, d.nd * sizeof(unsigned int));
    tail += d.nd; /* * sizeof(unsigned int) / sizeof(int) */
  }
};

inline bool operator<(const SigString& a, const SigString& b) {
  if(a.which != b.which) return a.which < b.which;
  ptrdiff_t a_size = (ptrdiff_t)(a.tail - a.data), b_size = (ptrdiff_t)(b.tail - b.data);
  if(a_size != b_size) return a_size < b_size;
  return memcmp(a.data, b.data, a_size * sizeof(int)) < 0;
}

inline bool operator==(const SigString& a, const SigString& b) {
  if(a.which != b.which) return false;
  ptrdiff_t a_size = a.tail - a.data;
  if(a_size != (ptrdiff_t)(b.tail - b.data)) return false;
  return memcmp(a.data, b.data, a_size * sizeof(int)) == 0;
}

inline bool operator!=(const SigString& a, const SigString& b) { return !(a == b); }

// returns the binary representation (in an int) of a float
static inline int float_contents_as_int(float x) {
  union {
    int bin;
    float f;
  };
  f = x;
  return bin;
}
static_assert(sizeof(int) >= sizeof(float),
    "float_contents_as_int needs float to be the same size or smaller than int");

struct SigHash {
  SigHash(int which) : hash((int)0xcc9e2d51 ^ which), which(which) { }
  SigHash() : hash((int)0xcc9e2d51a), which(0) { }
  int hash;
  int which;

  // sbdm hash
  inline void add_int(int i) {
    hash = i + (hash << 6) + (hash << 16) - hash;
  }
  inline void add_float(float f) {
    add_int(float_contents_as_int(f));
  }
  inline void add_which(int _which){hash ^= which ^ _which; which = _which;}
  void add_node(unsigned i) { add_int((int)i); }
  void add_dim(const Dim &d) {
    add_int(-(int)d.nd);
    for(size_t i = 0; i < d.nd; ++i)
      add_int((int)d.d[i]);
  }
};

inline bool operator<(const SigHash& a, const SigHash& b) {
  return a.hash < b.hash;
}
inline bool operator==(const SigHash& a, const SigHash& b) {
  return a.hash == b.hash;
}
inline bool operator!=(const SigHash& a, const SigHash& b) {
  return a.hash != b.hash;
}

template <class Sig>
struct SigLinearMap {
  SigLinearMap() { sigs.reserve(50); whiches.reserve(50); Sig s; sigs.push_back(s); whiches.push_back(s.which); }
  int get_idx(Sig &s) {
    for (unsigned i=0; i<sigs.size(); ++i) {
      if (sigs[i]==s)
          return i;
    }
    sigs.push_back(s);
    whiches.push_back(s.which);
    return sigs.size()-1;
  }
  int sig2type(int sig) { return whiches[sig]; }
  int size() { return sigs.size(); }
  std::vector<Sig> sigs;
  std::vector<int> whiches;
};

template <class Sig>
struct SigLinearSortedMap {
  SigLinearSortedMap() : sorted(false), found(0) { sigs.reserve(50); whiches.reserve(50); Sig s; sigs.push_back(std::pair<Sig,int>(s,0)); whiches.push_back(s.which); }
  int get_idx(Sig &s) {
    if (sorted) {
      // search and return
      auto loc = std::lower_bound(sigs.begin(), sigs.end(), std::pair<Sig, int>(s,0), [](const std::pair<Sig, int> &s1, const std::pair<Sig, int> &s2) { return s1.first<s2.first; });
      if (loc != sigs.end() && loc->first==s) {
        return loc->second;
      }
      // not found, continue to add.
    } else { // linear scan
      for (unsigned i=0; i<sigs.size(); ++i) {
        if (sigs[i].first==s) {
          const int res=sigs[i].second;
          found++;
          if (found > 50 && !sorted) sort();
          return res;
        }
      }
    }
    found=0;
    sorted=false;
    int next_sig;
    if (valid_sigs.size()){
      next_sig = valid_sigs.back();
      valid_sigs.pop_back();
      whiches[next_sig] = s.which;
    }
    else{
      assert(whiches.size() == sigs.size());
      next_sig = (int)(sigs.size());
      whiches.push_back(s.which);
    }
    sigs.push_back(std::pair<Sig, int>(s, next_sig));
    return next_sig;
  }
  void clear() {
    sigs.clear(); whiches.clear(); sorted=false;
  }
  void sort() {
    if (sorted) { return; }
    std::sort(sigs.begin(), sigs.end(), [](std::pair<Sig, int> s1, std::pair<Sig, int> s2) {return s1.first < s2.first;});
    sorted=true;
  }
  
  std::vector<int> valid_sigs; 
  
  void invalidate (nt::NodeType type){
    // return;
    // std::cout << "[sigmap::invalidate]:";
    // std::cout << "before:"; show();
    auto iter = sigs.begin();
    while (iter != sigs.end()){
      if (whiches[iter->second] == type){
        valid_sigs.push_back(iter->second);
        iter = sigs.erase(iter);
      }
      else iter++;
    } 
    // std::cout << "after:"; show();
  }

  void show(){
    std::cout << "sig:";
    for (auto& sig: sigs) std::cout << "(" << nt::type2name[whiches[sig.second]] << ","  << sig.second << "),";
    std::cout << "erased:";
    for (auto& t: valid_sigs) std::cout << t << ",";
    std::cout << std::endl;
  }

  int sig2type(int sig) { return whiches[sig]; }
  int size() { return sigs.size(); }
  std::vector<std::pair<Sig,int> > sigs;
  std::vector<int> whiches;
  bool sorted;
  int found;
};

struct SigHasher {
  size_t operator()(const SigHash& k) const { return k.hash; }
};

template <class Sig>
struct SigTreeMap {
  SigTreeMap() { }
  int get_idx(Sig &s) {
    auto it = sigs.find(s);
    if(it != sigs.end()) return it->second;
    sigs.insert(std::make_pair(s, (int)sigs.size()));
    return sigs.size()-1;
  }
  int size() { return sigs.size(); }
  std::map<Sig, int> sigs;
};

template <class Sig>
struct SigHashMap {
  SigHashMap() {Sig s; sigs[s.hash] = {s.which, 0};}
  int get_idx(Sig &s) {
    if (sigs.count(s.hash) == 0) 
      sigs[s.hash] = {s.which, sigs.size()};
    return sigs[s.hash].second;
  }
  int size() { return sigs.size(); }
  std::unordered_map<int, std::pair<int, int> > sigs;
  int sig2type (int sig){
    for (auto& kv: sigs) if (kv.second.second == sig) return kv.second.first;
    return 0;
  }
};

typedef SigHash Sig;
//typedef SigLinearMap<Sig> SigMap;
typedef SigLinearSortedMap<Sig> SigMap;
// typedef SigHashMap<Sig> SigMap;

} // namespace dynet

#endif
