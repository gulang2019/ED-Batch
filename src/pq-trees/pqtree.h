/*
PQ-Tree class based on the paper "Testing for the Consecutive Onces Property,
Interval Graphs, and Graph Planarity Using PQ-Tree Algorithms" by Kellog S.
Booth and George S. Lueker in the Journal of Computer and System Sciences 13,
335-379 (1976)
*/

// This file is part of the PQ Tree library.
//
// The PQ Tree library is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version.
//
// The PQ Tree Library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
// or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// for more details.
//
// You should have received a copy of the GNU General Public License along
// with the PQ Tree Library.  If not, see <http://www.gnu.org/licenses/>.
#include <assert.h>
#include <list>
#include <set>
#include <vector>
#include <queue>
#include <map>
#include <iostream>
#include <unordered_set>
#include "pqnode.h"
#include "set_methods.h"


#ifndef PQTREE_H
#define PQTREE_H

class PQTree {
  friend class IsoPQTree;
  private:

  // Root node of the PQTree
  PQNode *root_;

  // The number of blocks of blocked nodes during the 1st pass
  int block_count_;

  // The number of blocked nodes during the 1st pass
  int blocked_nodes_;

  // A variable (0 or 1) which is a count of the number of virtual nodes which
  // are imagined to be in the queue during the bubbling up.
  int off_the_top_;

  // Keeps track of all reductions performed on this tree in order
  std::list<std::set<int> > reductions_;

  // Keeps a pointer to the leaf containing a particular value this map actually
  // increases the time complexity of the algorithm.  To fix, you can create an
  // array of items so that each item hashes to its leaf address in constant
  // time, but this is a tradeoff to conserve space.
  std::map<int, PQNode*> leaf_address_;

  // A reference to a pseudonode that cannot be reached through the root
  // of the tree.  The pseudonode is a temporary node designed to handle
  // a special case in the first bubbling up pass it only exists during the
  // scope of the reduce operation
  PQNode* pseudonode_;

  // true if a non-safe reduce has failed, tree is useless.
  bool invalid_;

  // store the modified node to speedup reset
  std::unordered_set<PQNode*> dirty_nodes_;
  inline PQNode * new_node(){
    PQNode * ret = new PQNode;
    dirty_nodes_.insert(ret);
    // fprintf(stdout, "[insert_dirty %s, %d]: %p\n", __FILE__, __LINE__, ret);
    return ret;
  }
  inline void delete_node(PQNode* node){
    dirty_nodes_.erase(node);
    delete node;
  }
  void Reset();

  // Loops through the consecutive blocked siblings of an unblocked node
  // recursively unblocking the siblings.
  // Args:
  //   sibling: next sibling node to unblock (if already unblocked, this call
  //            is a no-op).
  //   parent: Node to std::set as the parent of all unblocked siblings
  int UnblockSiblings(PQNode* sibling);

  // All of the templates for matching a reduce are below.  The template has a
  // letter describing which type of node it refers to and a number indicating
  // the index of the template for that letter.  These are the same indices in
  // the Booth & Lueker paper.  The return value indicates whether or not the
  // pattern accurately matches the template
  bool TemplateL1(PQNode* candidate_node);
  bool TemplateQ1(PQNode* candidate_node);
  bool TemplateQ2(PQNode* candidate_node);
  bool TemplateQ3(PQNode* candidate_node);
  bool TemplateP1(PQNode* candidate_node, bool is_reduction_root);
  bool TemplateP2(PQNode* candidate_node);
  bool TemplateP3(PQNode* candidate_node);
  bool TemplateP4(PQNode* candidate_node);
  bool TemplateP5(PQNode* candidate_node);
  bool TemplateP6(PQNode* candidate_node);

  // This procedure is the first pass of the Booth&Leuker PQTree algorithm
  // It processes the pertinent subtree of the PQ-Tree to determine the mark
  // of every node in that subtree
  // the pseudonode, if created, is returned so that it can be deleted at
  // the end of the reduce step
  // whether to use a pseudonode for the interior Qnode case
  bool Bubble(const std::set<int>& S, bool create_pseudonode = true);

  bool ReduceStep(const std::set<int>& S, PQNode ** root = nullptr);

 public:
  // Default constructor - constructs a tree using a std::set
  // Only reductions using elements of that std::set will succeed
  PQTree(std::set<int> S);
  template<typename Iterator> 
  PQTree(const Iterator &begin, const Iterator & end) {
    // Set up the root node as a P-Node initially.
    root_ = new PQNode;
    root_->type_ = PQNode::pnode;
    invalid_ = false;
    pseudonode_ = NULL;
    block_count_ = 0;
    blocked_nodes_ = 0;
    off_the_top_ = 0;
    for (auto i = begin; i != end; i++) {
      PQNode *new_node;
      new_node = new PQNode(*i);
      leaf_address_[*i] = new_node;
      new_node->parent_ = root_;
      new_node->type_ = PQNode::leaf;
      root_->circular_link_.push_back(new_node);
    }
  }
  PQTree(const PQTree& to_copy);
  ~PQTree();

  // Returns the root PQNode used for exploring the tree.
  PQNode* Root();

  // Mostly for debugging purposes, Prints the tree to standard out
  std::string Print() const;

  // Cleans up pointer mess caused by having a pseudonode
  void CleanPseudo();

  // Reduces the tree but protects if from becoming invalid if the reduction
  // fails, takes more time.
  bool SafeReduce(const std::set<int>&);
  bool SafeReduceAll(const std::list<std::set<int> >&);

  //reduces the tree - tree can become invalid, making all further
  //reductions fail
  bool Reduce(const std::set<int>& S);
  bool ReduceAll(const std::list<std::set<int> >& L);
  bool ReduceAll(const std::set<std::set<int> >& L);
  bool ReduceAll(const std::vector<std::set<int> >& L);

  // add some new nodes 
  void add_nodes(const std::set<int>& S);

  // Returns 1 possible frontier, or ordering preserving the reductions
  std::list<int> Frontier();

  // Assignment operator
  PQTree& operator=(const PQTree& to_copy);

  // Copies to_copy.  Used for the copy constructor and assignment operator.
  void CopyFrom(const PQTree& to_copy);


  // Returns a frontier not including leaves that were not a part of any
  // reduction
  std::list<int> ReducedFrontier();

  // Returns the reductions that have been performed on this tree.
  std::list<std::set<int> > GetReductions();

  // Returns the std::set of all elements on which a reduction was performed.
  std::set<int> GetContained();
};

void ReduceBy(const std::set<int>& reduce_set, PQTree* tree); 

#endif
