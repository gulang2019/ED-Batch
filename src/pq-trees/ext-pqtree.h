#ifndef EXT_PQTREE_H
#define EXT_PQTREE_H

#include "pqtree.h"
#include <assert.h>
#include <unordered_set>
#include <unordered_map>

struct Pattern {
    int subset_size;
    std::vector<std::vector<int> > subsets;
    std::unordered_map<int, int> ele2isoid;
    bool valid = true;
    Pattern(std::vector<std::vector<int> > & subsets_){
        for (auto & subset: subsets_)
            subsets.push_back(subset);
        assert(subsets.size());
        subset_size = subsets[0].size();
        init();
    }
    Pattern(std::vector<std::vector<int> > && subsets_){
        for (auto & subset: subsets_)
            subsets.push_back(move(subset));
        assert(subsets.size());
        subset_size = subsets[0].size();
        init();
    }
    void init(){
        for (auto& subset: subsets){
            assert((int)subset.size() == subset_size);
            for (int i = 0; i < subset_size; i++){
                ele2isoid[subset[i]] = i;
            }
        }
    }
    void transform(const std::set<std::set<int> >& in, std::set<std::set<int> >& out) const{
        for (auto& s: in){
            std::set<int> s_;
            for (auto ele: s) {
                if (ele2isoid.count(ele) == 0) {
                    std::cout << "ERROR: " << ele << " not in ";
                    show();
                    std::cout << std::endl;
                }
                s_.insert(ele2isoid.at(ele));
            }
            out.insert(move(s_));
        }
    }
    std::set<int> transform(const std::set<int>& in) const{
        std::set<int> s_;
        for (auto ele: in) s_.insert(ele2isoid.at(ele));
        return move(s_);
    }
    void reverseTransform(const std::set<std::set<int> >& in, std::set<std::set<int> >&out) const{
        for (auto & subset: subsets){
            for (auto & s: in){
                std::set<int> s_;
                for (auto ele: s) s_.insert(subset[ele]);
                out.insert(move(s_));
            }
        }
    }

    int count(const int& v) const{
        return ele2isoid.count(v);
    }

    int getSubsetIdx(const std::set<int>& subset) const{
        int idx = 0;
        for (auto & subset_key:  subsets){
            std::set<int> tmp(subset_key.begin(), subset_key.end());
            if (tmp == subset)
                return idx;
            idx++;
        }
        return -1;
    }

    void show() const {
        fprintf(stdout, "{", subset_size);
        for (auto &subset: subsets) {
            fprintf(stdout, "{");
            for (auto ele: subset)  
                fprintf(stdout, "%d, ", ele);
            fprintf(stdout, "}, ");
        }
        fprintf(stdout, "},");
    }
};

struct PatternInfo{
    std::set<std::set<int> > consecutive_constraints;
    void show() const {
        fprintf(stdout, "consecutive_constraints: ");
        for (auto & cons: consecutive_constraints) {
            fprintf(stdout, "(");
            for (auto ele: cons) fprintf(stdout, "%d, ", ele); 
            fprintf(stdout, "), ");
        }
        fprintf(stdout, "\n");
    }
};

class Visitor{
public:
  static std::set<int> ExtractCanonicalSubsets(PQNode* node, std::set<std::set<int> >& canonical_subsets);
  static void GetPatternInfoBFS(const std::set<int>& subset, std::set<std::set<int> > & consecutive_constraints);
  static bool InTree(PQNode * node, const PQNode * target);
  static void ReplaceBinaryPNodes(PQNode * node); 
};

class IsoPQTree {
    
public:
    IsoPQTree(PQTree* tree_in): tree(tree_in){
        assert(tree_in != nullptr);
    }
    IsoPQTree(const std::set<int>&S) {
        tree = new PQTree(S);
    }
    IsoPQTree(int N) {
        std::set<int> S;
        for (int i = 0; i < N; i++)
            S.insert(i);
        tree = new PQTree(S);
    }
    template<typename Iterator>
    IsoPQTree(Iterator begin, Iterator end){
        tree = new PQTree(begin, end);
    }
    void PQTree2Collection(std::list<std::set<int>> & collections);
    // bool ReduceAll(const std::list<std::set<int> > & reduction_sets);
    bool ReduceAll(const std::set<std::set<int> > & reduction_sets);
    bool ReduceAll(const std::list<std::set<int> > & reduction_sets);
    bool ReduceAll(const std::vector<std::set<int> > & reduction_sets);
    bool Reduce(const std::set<int> & reduction_set);
    bool IsoReduce(
        // [pattern_id, subset_id, seq_id] -> element_id, require [pid, *, *] to be iso
        Pattern & isoPattern, PatternInfo & pattern_info, bool & hasUpdate
    );
    bool IsoReduceAll(
        const std::vector<std::vector<std::vector<int> > > &isoPatterns
    );
    bool IsoReduceAll(
        std::vector<Pattern> &isoPatterns
    );
    bool GetEquivalentClass(
        Pattern & pattern
    );
    bool Check(const std::vector<std::vector<std::vector<int> > >& patterns, bool verbose = false);
    std::string Print();
    std::list<int> Frontier();
private:
    PQTree * tree;
    // 0: don't change current order
    // 1: change current order
    std::vector<std::map<PQNode*, bool> > qnodes2dir;
    std::unordered_map<PQNode*, bool> qnode2dir;
    std::unordered_set<PQNode*> fixed_pnodes;
    void FindFrontier(PQNode * node, std::list<int> & ordering);
    // {} normal pnode; <>: fixed pqnode; () fixed qnode; [] normal qnode
    void Print(PQNode* node, std::string * str) const;
    bool GetPatternInfo(const std::set<int>& subset, std::set<std::set<int> >& consecutive_constraints);
    bool DecideQNodeDir();
    bool Serialize(const std::vector<int>& subset, std::vector<PQNode*>& nodes,
    std::unordered_map<PQNode*, std::set<int> >& node2leave);
};

#endif