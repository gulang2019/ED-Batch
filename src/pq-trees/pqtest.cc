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

#include <iostream>
#include <set>
#include "pqnode.h"
#include "pqtree.h"
#include "ext-pqtree.h"

using namespace std;

string ReadableType(PQNode::PQNode_types type) {
  if (type == PQNode::leaf) {
    return "leaf";
  } else if (type == PQNode::pnode) {
    return "P-Node";
  } else if (type == PQNode::qnode) {
    return "Q-Node";
  }
  return "unknown";
}

void ReduceBy(const set<int>& reduce_set, IsoPQTree* tree) {
  cout << "Reducing by set { ";
  for (set<int>::iterator i = reduce_set.begin(); i != reduce_set.end(); ++i)
    cout << *i << " ";
  cout << "}" << endl;

  assert (tree->Reduce(reduce_set));
  cout << tree->Print() << endl;
}
// Subgraph1 
void TestBed1() {
  set<int> S;
  for (int i = 0; i < 13; i++)
    S.insert(i);
  PQTree tree(S);

  cout << "PQ Tree with 8 elements and no reductions" << endl;
  cout << tree.Print() << endl;


  ReduceBy(set<int>({1,4,7}), &tree);
  ReduceBy(set<int>({2,5,8}), &tree);
  ReduceBy(set<int>({6,9}), &tree);
  ReduceBy(set<int>({5,8}), &tree);

  // Lets actually explore the tree manually
  // cout << endl;
  // PQNode* root = tree.Root();
  // cout << "Root Type: " << ReadableType(root->Type()) << endl;
  // vector<PQNode*> children;
  // root->Children(&children);
  // for (int i = 0; i < children.size(); ++i) {
  //   PQNode* child = children[i];
  //   cout << "Child " << i + 1 << " Type: " << ReadableType(child->Type());
  //   if (child->Type() == PQNode::leaf) {
  //     cout << " Value: " << child->LeafValue() << endl;
  //   } else {
  //     cout << endl;
  //     vector<PQNode*> grandchildren;
  //     child->Children(&grandchildren);
  //     for (int j = 0; j < grandchildren.size(); ++j) {
  //       PQNode* grandchild = grandchildren[j];
  //       cout << "GrandChild " << j + 1 << " Type: "
  //            << ReadableType(grandchild->Type());
  //       if (grandchild->Type() == PQNode::leaf)
  //         cout << " Value: " << grandchild->LeafValue();
  //         cout << endl;
  //     }
  //   }
  // }
  // cout << endl;

  // Now, we perform a reduction that will fail.
  // cout << "Reducing by set {5, 3} - will fail" << endl;
  // S.clear();
  // S.insert(5);
  // S.insert(3);
  // cout << tree.Reduce(S) << endl;
  // cout << tree.Print() << endl;
}

void TestBed2() {
  set<int> S;
  for (int i = 0; i < 6; i++)
    S.insert(i);
  PQTree tree(S);

  S.clear();
  S.insert(4);
  S.insert(1);
  ReduceBy(S, &tree);

  S.clear();
  S.insert(3);
  S.insert(0);
  S.insert(2);
  S.insert(5);
  S.insert(4);
  ReduceBy(S, &tree);

  S.clear();
  S.insert(0);
  S.insert(2);
  S.insert(5);
  S.insert(4);
  ReduceBy(S, &tree);

  S.clear();
  S.insert(2);
  S.insert(5);
  ReduceBy(S, &tree);

  S.clear();
  S.insert(0);
  S.insert(2);
  ReduceBy(S, &tree);
}

void TestInteractive(){
  int n;
  cin >> n;
  set<int> S;
  for (int i = 1; i <= n; i++)
    S.insert(i);
  fprintf(stdout, "Set(%d)\n", n);
  PQTree tree(S);
  while(cin >> n){
    if (n < 0) return;
    set<int> S;
    bool failed = false;
    for (int i = 0; i < n; i++) {
      int t;
      cin >> t;
      S.insert(t);
      if (t < 0) {
        failed = true;
        break;
      }
    }
    if (failed) continue;
    ReduceBy(S, &tree);
  }
}

void TestTree2Collection(){
  set<int> S;
  for(int i = 0; i < 13; i++) S.insert(i);
  PQTree tree(S);

  cout << "PQ Tree with 8 elements and no reductions" << endl;
  cout << tree.Print() << endl;


  ReduceBy(set<int>({1,4,7}), &tree);
  ReduceBy(set<int>({2,5,8}), &tree);
  ReduceBy(set<int>({6,9}), &tree);
  ReduceBy(set<int>({5,8}), &tree);
  ReduceBy(set<int>({6,8}), &tree);

  IsoPQTree iso_pqtree(&tree);
  list<set<int> > collections;
  iso_pqtree.PQTree2Collection(collections);
  fprintf(stdout, "---------------------test---------------------\n");
  int idx = 0;
  for (auto& collection: collections){
    fprintf(stdout, "collection %d: ", idx++);
    for (auto node: collection) 
      fprintf(stdout, "%d, ", node);
    fprintf(stdout, "\n");
  }

  for (auto& collection: collections){
    ReduceBy(collection, &tree);
  }
  fprintf(stdout, "----------test:tree2collection passed!---------\n");
}

void TestIsoReductionAll1(){
  IsoPQTree tree(10);
  ReduceBy({0,1,2}, &tree);
  ReduceBy({3,4,5}, &tree);
  ReduceBy({7,8}, &tree);
  ReduceBy({0,1}, &tree);
  ReduceBy({4,5}, &tree);

  fprintf(stdout, "----------------------test----------------------\n");
  fprintf(stdout, "tree before: %s\n", tree.Print().c_str());
  assert(tree.IsoReduceAll({{
    {3,4,5},
    {0,1,2}
  },{
    {0,1},
    {4,5},
    {7,8}
  }
  }));
  fprintf(stdout, "tree: %s\n", tree.Print().c_str());
  assert(tree.Check({{
    {3,4,5},
    {0,1,2}
  },{
    {0,1},
    {4,5},
    {7,8}
  }
  }));
  fprintf(stdout, "----------iso reduction 1 test passed-------------\n");
}

void TestIsoReductionAll2(){
  IsoPQTree tree(10);

  fprintf(stdout, "----------------------test----------------------\n");
  fprintf(stdout, "tree before: %s\n", tree.Print().c_str());
  assert(tree.IsoReduceAll({{
    {4,5,6},
    {1,2,3}
  },{
    {1,2},
    {5,6},
    {8,9}
  }
  }));
  fprintf(stdout, "tree: %s\n", tree.Print().c_str());
  assert(tree.Check({{
    {4,5,6},
    {1,2,3}
  },{
    {1,2},
    {5,6},
    {8,9}
  }
  }));
  fprintf(stdout, "----------iso reduction 2 test passed-------------\n");
}

void TestIsoReductionAll3(){
  IsoPQTree tree(15);

  fprintf(stdout, "----------------------test----------------------\n");
  fprintf(stdout, "tree before: %s\n", tree.Print().c_str());
  assert(tree.IsoReduceAll({{
    {5,8,10,11},
    {0,2,3,4}
  },{
    {12,13,14},
    {5,8,10},
    {6,7,9}
  }
  }));
  fprintf(stdout, "tree: %s\n", tree.Print().c_str());
  assert(tree.Check({{
    {5,8,10,11},
    {0,2,3,4}
  },{
    {12,13,14},
    {5,8,10},
    {6,7,9}
  }
  }));
  fprintf(stdout, "----------iso reduction 3 test passed-------------\n");
}

void TestIsoReductionAll4(){
  IsoPQTree tree(20);

  fprintf(stdout, "----------------------test----------------------\n");
  fprintf(stdout, "tree before: %s\n", tree.Print().c_str());
  assert(tree.IsoReduceAll({{
    {0,1,2,3,4},{5,6,7,8,9}
  },{
    {10,11,12,13},{5,6,8,9},
  },{
    {15,16,17},{11,12,13},{14,18,19}
  }
  }));
  fprintf(stdout, "tree: %s\n", tree.Print().c_str());
  assert(tree.Check({{
    {0,1,2,3,4},{5,6,7,8,9}
  },{
    {10,11,12,13},{5,6,8,9},
  },{
    {15,16,17},{11,12,13},{14,18,19}
  }
  }));
  fprintf(stdout, "----------iso reduction 4 test passed-------------\n");
}

void TestIsoReductionAll5(){
  IsoPQTree tree(13);

  fprintf(stdout, "----------------------test----------------------\n");
  fprintf(stdout, "tree before: %s\n", tree.Print().c_str());
  assert(tree.IsoReduceAll({{
    {0,8},{1,9}
  },{
    {2,6},{3,7},
  },{
    {1,7,9},{10,11,12}
  }
  }));
  fprintf(stdout, "tree: %s\n", tree.Print().c_str());
  assert(tree.Check({{
    {0,8},{1,9}
  },{
    {2,6},{3,7},
  },{
    {1,7,9},{10,11,12}
  }
  }));
  fprintf(stdout, "----------iso reduction 5 test passed-------------\n");
}

void TestFrontier(){
  IsoPQTree tree(20);
  
  fprintf(stdout, "----------------------test----------------------\n");
  fprintf(stdout, "tree before: %s\n", tree.Print().c_str());
  assert(tree.IsoReduceAll({{
    {0,1,2,3,4},{5,6,7,8,9}
  },{
    {10,11,12,13},{5,6,8,9},
  },{
    {15,16,17},{11,12,13},{14,18,19}
  }
  }));
  fprintf(stdout, "tree: %s\n", tree.Print().c_str());
  list<int> frontier = tree.Frontier();
  fprintf(stdout, "frontier: ");
  for (auto ele: frontier) fprintf(stdout, "%d, ", ele);
  fprintf(stdout, "\n");
  assert(tree.Check({{
    {0,1,2,3,4},{5,6,7,8,9}
  },{
    {10,11,12,13},{5,6,8,9},
  },{
    {15,16,17},{11,12,13},{14,18,19}
  }
  }));
  fprintf(stdout, "----------frontier test passed!-------------\n");
}


void TestIsoReductionAll6(){
  set<int> S =  {1226, 1225, 1218, 1217, 1216, 1214, 1215, 1219, 1220, 1221, 1222, 1223, 1224};
  IsoPQTree tree(S);
  vector<vector<vector<int>>> patterns = {
    {{1221, 1218, 1215, },},
    {{1221, 1218, 1215, },{1222, 1219, 1216, },},
    {{1219, 1216, },{1220, 1217, },},
  };

  fprintf(stdout, "----------------------test----------------------\n");
  tree.IsoReduceAll(patterns);
  list<int> mem_ids = tree.Frontier();
  fprintf(stdout, "frontier: ");
  for (auto id: mem_ids){
    fprintf(stdout, "%d, ", id);
  }
  fprintf(stdout, "\n");
  fprintf(stdout, "tree: %s\n", tree.Print().c_str());
  assert(tree.Check(patterns));
  fprintf(stdout, "----------iso reduction 6 test passed-------------\n");
}

void TestIsoReductionAll7(){
  IsoPQTree tree(14);
  vector<vector<vector<int>>> patterns = {
    {{2,3}},
    {{4,5,6}, {1,2,3}},
    {{8, 9},{1, 2}, {5,6}},
    {{11, 12, 13},{8, 9, 1},},
  };

  fprintf(stdout, "----------------------test----------------------\n");
  tree.IsoReduceAll(patterns);
  list<int> mem_ids = tree.Frontier();
  fprintf(stdout, "frontier: ");
  for (auto id: mem_ids){
    fprintf(stdout, "%d, ", id);
  }
  fprintf(stdout, "\n");
  fprintf(stdout, "tree: %s\n", tree.Print().c_str());
  assert(tree.Check(patterns));
  fprintf(stdout, "----------iso reduction 7 test passed-------------\n");
}


void TestIsoReductionAll8(){
  IsoPQTree tree(18);
  vector<vector<vector<int>>> patterns = {
    {{2,4,8,10}, {1,3,7,9}},
    {{12,11,13},{8,10,2},}
  };

  fprintf(stdout, "----------------------test----------------------\n");
  tree.IsoReduceAll(patterns);
  list<int> mem_ids = tree.Frontier();
  fprintf(stdout, "frontier: ");
  for (auto id: mem_ids){
    fprintf(stdout, "%d, ", id);
  }
  fprintf(stdout, "\n");
  fprintf(stdout, "tree: %s\n", tree.Print().c_str());
  assert(tree.Check(patterns));
  fprintf(stdout, "----------iso reduction 8 test passed-------------\n");
}

void TestIsoReductionAll9(){
  IsoPQTree tree(13);
  vector<vector<vector<int>>> patterns = {
    {{1,2}, {3,5}, {4,6}},
    {{3,4}, {7,9},{8,10}}
  };

  fprintf(stdout, "----------------------test----------------------\n");
  tree.IsoReduceAll(patterns);
  list<int> mem_ids = tree.Frontier();
  fprintf(stdout, "frontier: ");
  for (auto id: mem_ids){
    fprintf(stdout, "%d, ", id);
  }
  fprintf(stdout, "\n");
  fprintf(stdout, "tree: %s\n", tree.Print().c_str());
  assert(tree.Check(patterns));
  fprintf(stdout, "----------iso reduction 9 test passed-------------\n");
}

void TestIsoReductionAll10(){
  IsoPQTree tree(13);
  vector<vector<vector<int>>> patterns = {
    {{3,4,6}, {7,9,11}, {2,8,10}},
    {{1,2}, {3,5}, {4,6}},
  };

  fprintf(stdout, "----------------------test----------------------\n");
  tree.IsoReduceAll(patterns);
  list<int> mem_ids = tree.Frontier();
  fprintf(stdout, "frontier: ");
  for (auto id: mem_ids){
    fprintf(stdout, "%d, ", id);
  }
  fprintf(stdout, "\n");
  fprintf(stdout, "tree: %s\n", tree.Print().c_str());
  assert(tree.Check(patterns));
  fprintf(stdout, "----------iso reduction 10 test passed-------------\n");
}

void TestIsoReductionAll11(){
  IsoPQTree tree(350);
  vector<vector<vector<int>>> patterns = {
{{0, }, },
{{172, 169, 163, 160, 151, 148, 142, 139, 127, 124, 118, 115, 106, 103, 97, 94, 79, 76, 70, 67, 58, 55, 49, 46, 34, 31, 25, 22, 13, 10, 4, 1, }, },
{{173, 170, 164, 161, 152, 149, 143, 140, 128, 125, 119, 116, 107, 104, 98, 95, 80, 77, 71, 68, 59, 56, 50, 47, 35, 32, 26, 23, 14, 11, 5, 2, }, {172, 169, 163, 160, 151, 148, 142, 139, 127, 124, 118, 115, 106, 103, 97, 94, 79, 76, 70, 67, 58, 55, 49, 46, 34, 31, 25, 22, 13, 10, 4, 1, }, },
{{174, 171, 165, 162, 153, 150, 144, 141, 129, 126, 120, 117, 108, 105, 99, 96, 81, 78, 72, 69, 60, 57, 51, 48, 36, 33, 27, 24, 15, 12, 6, 3, }, {172, 169, 163, 160, 151, 148, 142, 139, 127, 124, 118, 115, 106, 103, 97, 94, 79, 76, 70, 67, 58, 55, 49, 46, 34, 31, 25, 22, 13, 10, 4, 1, }, },
{{7, 16, 28, 37, 52, 61, 73, 82, 100, 109, 121, 130, 145, 154, 166, 175, }, {2, 11, 23, 32, 47, 56, 68, 77, 95, 104, 116, 125, 140, 149, 161, 170, }, {3, 12, 24, 33, 48, 57, 69, 78, 96, 105, 117, 126, 141, 150, 162, 171, }, {5, 14, 26, 35, 50, 59, 71, 80, 98, 107, 119, 128, 143, 152, 164, 173, }, {6, 15, 27, 36, 51, 60, 72, 81, 99, 108, 120, 129, 144, 153, 165, 174, }, },
{{8, 17, 29, 38, 53, 62, 74, 83, 101, 110, 122, 131, 146, 155, 167, 176, }, {7, 16, 28, 37, 52, 61, 73, 82, 100, 109, 121, 130, 145, 154, 166, 175, }, },
{{9, 18, 30, 39, 54, 63, 75, 84, 102, 111, 123, 132, 147, 156, 168, 177, }, {7, 16, 28, 37, 52, 61, 73, 82, 100, 109, 121, 130, 145, 154, 166, 175, }, },
{{178, 157, 133, 112, 85, 64, 40, 19, }, {167, 146, 122, 101, 74, 53, 29, 8, }, {168, 147, 123, 102, 75, 54, 30, 9, }, {176, 155, 131, 110, 83, 62, 38, 17, }, {177, 156, 132, 111, 84, 63, 39, 18, }, },
{{179, 158, 134, 113, 86, 65, 41, 20, }, {178, 157, 133, 112, 85, 64, 40, 19, }, },
{{180, 159, 135, 114, 87, 66, 42, 21, }, {178, 157, 133, 112, 85, 64, 40, 19, }, },
{{43, 88, 136, 181, }, {20, 65, 113, 158, }, {21, 66, 114, 159, }, {41, 86, 134, 179, }, {42, 87, 135, 180, }, },
{{44, 89, 137, 182, }, {43, 88, 136, 181, }, },
{{45, 90, 138, 183, }, {43, 88, 136, 181, }, },
{{184, 91, }, {137, 44, }, {138, 45, }, {182, 89, }, {183, 90, }, },
{{185, 92, }, {184, 91, }, },
{{186, 93, }, {184, 91, }, },
{{187, }, {92, }, {93, }, {185, }, {186, }, },
{{188, }, {187, }, },
{{189, }, {187, }, },
{{190, 192, 254, 286, 256, 224, 194, 196, 210, 226, 240, 258, 272, 288, 302, 310, 304, 296, 290, 280, 274, 266, 260, 248, 242, 234, 228, 218, 212, 204, 198, 200, 202, 206, 208, 214, 216, 220, 222, 230, 232, 236, 238, 244, 246, 250, 252, 262, 264, 268, 270, 276, 278, 282, 284, 292, 294, 298, 300, 306, 308, 312, 314, }, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, }, {188, 92, 185, 182, 137, 89, 44, 20, 41, 65, 86, 113, 134, 158, 179, 176, 167, 155, 146, 131, 122, 110, 101, 83, 74, 62, 53, 38, 29, 17, 8, 2, 5, 11, 14, 23, 26, 32, 35, 47, 50, 56, 59, 68, 71, 77, 80, 95, 98, 104, 107, 116, 119, 125, 128, 140, 143, 149, 152, 161, 164, 170, 173, }, },
{{315, 313, 309, 307, 301, 299, 295, 293, 285, 283, 279, 277, 271, 269, 265, 263, 253, 251, 247, 245, 239, 237, 233, 231, 223, 221, 217, 215, 209, 207, 203, 201, 199, 205, 213, 219, 229, 235, 243, 249, 261, 267, 275, 281, 291, 297, 305, 311, 303, 289, 273, 259, 241, 227, 211, 197, 195, 225, 257, 287, 255, 193, 191, }, {314, 312, 308, 306, 300, 298, 294, 292, 284, 282, 278, 276, 270, 268, 264, 262, 252, 250, 246, 244, 238, 236, 232, 230, 222, 220, 216, 214, 208, 206, 202, 200, 198, 204, 212, 218, 228, 234, 242, 248, 260, 266, 274, 280, 290, 296, 304, 310, 302, 288, 272, 258, 240, 226, 210, 196, 194, 224, 256, 286, 254, 192, 190, }, },
{{316, }, {191, }, {193, }, {195, }, {197, }, {199, }, {201, }, {203, }, {205, }, {207, }, {209, }, {211, }, {213, }, {215, }, {217, }, {219, }, {221, }, {223, }, {225, }, {227, }, {229, }, {231, }, {233, }, {235, }, {237, }, {239, }, {241, }, {243, }, {245, }, {247, }, {249, }, {251, }, {253, }, {255, }, {257, }, {259, }, {261, }, {263, }, {265, }, {267, }, {269, }, {271, }, {273, }, {275, }, {277, }, {279, }, {281, }, {283, }, {285, }, {287, }, {289, }, {291, }, {293, }, {295, }, {297, }, {299, }, {301, }, {303, }, {305, }, {307, }, {309, }, {311, }, {313, }, {315, }, },
{{317, }, {316, }, },
};

  fprintf(stdout, "----------------------test----------------------\n");
  tree.IsoReduceAll(patterns);
  list<int> mem_ids = tree.Frontier();
  fprintf(stdout, "frontier: ");
  for (auto id: mem_ids){
    fprintf(stdout, "%d, ", id);
  }
  fprintf(stdout, "\n");
  fprintf(stdout, "tree: %s\n", tree.Print().c_str());
  assert(tree.Check(patterns));
  fprintf(stdout, "----------iso reduction 11 test passed-------------\n");
}


void TestIsoReductionAll12(){
  IsoPQTree tree(18);
  vector<vector<vector<int>>> patterns = {
    {{1,2,3}, {1,3,7,9}},
    {{12,11,13},{8,10,2},}
  };

  fprintf(stdout, "----------------------test----------------------\n");
  tree.IsoReduceAll(patterns);
  list<int> mem_ids = tree.Frontier();
  fprintf(stdout, "frontier: ");
  for (auto id: mem_ids){
    fprintf(stdout, "%d, ", id);
  }
  fprintf(stdout, "\n");
  fprintf(stdout, "tree: %s\n", tree.Print().c_str());
  assert(tree.Check(patterns));
  fprintf(stdout, "----------iso reduction 8 test passed-------------\n");
}


void TestIsoReductionAll13(){
  IsoPQTree tree(8);
  vector<vector<vector<int>>> patterns = {
    {{{0,1,2}, {3,5,6}},
    {{0,1,2}, {3,6,7}}}
  };

  fprintf(stdout, "----------------------test----------------------\n");
  tree.IsoReduceAll(patterns);
  list<int> mem_ids = tree.Frontier();
  fprintf(stdout, "frontier: ");
  for (auto id: mem_ids){
    fprintf(stdout, "%d, ", id);
  }
  fprintf(stdout, "\n");
  fprintf(stdout, "tree: %s\n", tree.Print().c_str());
  assert(tree.Check(patterns));
  fprintf(stdout, "----------iso reduction 13 test passed-------------\n");
}



int main(int argc, char **argv) {
  // TestInteractive();
  // TestTree2Collection(); 
  TestIsoReductionAll1();
  TestIsoReductionAll2();
  TestIsoReductionAll3();
  TestIsoReductionAll4();
  TestIsoReductionAll5();
  TestIsoReductionAll6();
  TestIsoReductionAll7();
  TestIsoReductionAll8();
  TestIsoReductionAll10();
  TestIsoReductionAll11();
  TestIsoReductionAll13();
  // TestFrontier();
}
