// Bounded beam search for 19 soups from the original fixed Coordination Ring
// starts. This is a heuristic search, NOT an infeasibility/optimality proof.
// Compile with a C++17 compiler. Args: beam width, cooking bonus, deadline
// decay. Stdout is the 400 joint actions as JSON; stderr reports progress.
// State: two (position, facing, inventory) triples, then reachable counter/pot
// contents. Both movement and interactions follow the pinned simulator; chef 0
// interacts first.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
using namespace std;
struct State {
  array<unsigned char, 24> v{};
  bool operator==(State const &o) const { return v == o.v; }
};
struct Hash {
  size_t operator()(State const &s) const {
    size_t h = 0;
    for (auto v : s.v)
      h = (h ^ v) * 1099511628211ULL;
    return h;
  }
};
struct Node {
  State s;
  int delivered = 0, parent = -1, a = 0;
  double score = 0;
};
int xy[8][2] = {{1, 1}, {2, 1}, {3, 1}, {3, 2}, {3, 3}, {2, 3}, {1, 3}, {1, 2}};
int vec[4][2] = {{0, -1}, {0, 1}, {1, 0}, {-1, 0}},
    dirs[6] = {2, 1, 3, 0, -1, -1};
vector<pair<int, int>> cells;
string grid[5] = {"WWWPW", "W   P", "B W W", "O   W", "WOXWW"};
int flooridx(int x, int y) {
  for (int i = 0; i < 8; i++)
    if (xy[i][0] == x && xy[i][1] == y)
      return i;
  return -1;
}
int cellidx(int x, int y) {
  for (int i = 0; i < (int)cells.size(); i++)
    if (cells[i] == make_pair(x, y))
      return i;
  return -1;
}
Node step(Node const &n, int action) {
  Node z = n;
  z.a = action;
  int acts[2] = {action / 6, action % 6}, old[2] = {n.s.v[0], n.s.v[3]},
      dest[2] = {old[0], old[1]};
  for (int a = 0; a < 2; a++) {
    int base = a * 3;
    if (acts[a] < 4) {
      int d = dirs[acts[a]];
      z.s.v[base + 1] = d;
      int p = flooridx(xy[old[a]][0] + vec[d][0], xy[old[a]][1] + vec[d][1]);
      if (p >= 0)
        dest[a] = p;
    }
  }
  if (dest[0] == dest[1] || (dest[0] == old[1] && dest[1] == old[0]))
    dest[0] = old[0], dest[1] = old[1];
  for (int a = 0; a < 2; a++)
    z.s.v[a * 3] = dest[a];
  for (int a = 0; a < 2; a++)
    if (acts[a] == 5) {
      int b = a * 3, p = dest[a], d = z.s.v[b + 1];
      int c = cellidx(xy[p][0] + vec[d][0], xy[p][1] + vec[d][1]);
      if (c < 0)
        continue;
      int inv = z.s.v[b + 2], val = z.s.v[6 + c];
      char k = grid[cells[c].second][cells[c].first];
      if (k == 'O' && inv == 0)
        z.s.v[b + 2] = 1;
      else if (k == 'B' && inv == 0)
        z.s.v[b + 2] = 2;
      else if (k == 'X' && inv == 3) {
        z.s.v[b + 2] = 0;
        z.delivered++;
      } else if (k == 'W') {
        if (!inv && val) {
          z.s.v[b + 2] = val;
          z.s.v[6 + c] = 0;
        } else if (inv && !val) {
          z.s.v[b + 2] = 0;
          z.s.v[6 + c] = inv;
        }
      } else if (k == 'P') {
        if (inv == 1 && val <= 2) {
          z.s.v[b + 2] = 0;
          z.s.v[6 + c] = val < 2 ? val + 1 : 23;
        } else if (inv == 2 && val == 3) {
          z.s.v[b + 2] = 3;
          z.s.v[6 + c] = 0;
        }
      }
    }
  for (int c = 0; c < (int)cells.size(); c++)
    if (grid[cells[c].second][cells[c].first] == 'P' && z.s.v[6 + c] >= 4)
      z.s.v[6 + c]--;
  return z;
}
// Dual potentials (scaled by 41) from theoretical_maximum_bound.py.
// A legal joint step increases potential by at most 2. Initial potential is 4.
int PHI[128] = {2, 7, 6, 39, 2, 7, 6, 39, 2, 7, 6, 39, 2, 7, 6, 39, 1, 8, 7, 38,
                2, 8, 7, 39, 1, 8, 7, 38, 1, 8, 7, 38, 0, 9, 8, 39, 0, 8, 7, 39,
                0, 9, 8, 39, 0, 8, 7, 39, 1, 8, 7, 40, 1, 8, 7, 40, 1, 8, 7, 40,
                2, 8, 7, 40, 2, 7, 6, 41, 2, 7, 6, 41, 3, 7, 6, 41, 2, 7, 6, 41,
                3, 7, 6, 42, 3, 6, 5, 43, 3, 6, 5, 42, 3, 6, 5, 42, 3, 5, 4, 41,
                4, 5, 4, 41, 3, 5, 4, 41, 4, 5, 4, 41, 3, 6, 5, 40, 3, 6, 5, 40,
                3, 7, 6, 40, 4, 6, 5, 40};
int WEIGHT[13][4] = {{0, 4, 3, 36}, {0, 0, 0, 0},  {0, 0, 0, 0},  {0, 6, 3, 36},
                     {0, 0, 0, 0},  {0, 6, 5, 36}, {0, 5, 4, 38}, {0, 0, 0, 0},
                     {0, 0, 0, 0},  {0, 4, 3, 38}, {0, 0, 0, 0},  {0, 6, 5, 38},
                     {0, 3, 2, 37}};
int current_time = 0;
double potential(Node const &n) {
  double v = n.delivered * 41;
  for (int a = 0; a < 2; a++)
    v += PHI[n.s.v[3 * a] * 16 + n.s.v[3 * a + 1] * 4 + n.s.v[3 * a + 2]];
  for (int c = 0; c < (int)cells.size(); c++) {
    int val = n.s.v[6 + c];
    char k = grid[cells[c].second][cells[c].first];
    if (k == 'P')
      v += 10 * (val < 3 ? val : 3);
    else if (k == 'W')
      v += WEIGHT[c][val];
  }
  return v;
}
double score(Node const &n, double bonus, double decay_time) {
  double base = potential(n), stock = base - n.delivered * 41;
  double decay = min(1.0, (400 - current_time) / decay_time);
  double ready = 0;
  for (int c = 0; c < (int)cells.size(); c++)
    if (grid[cells[c].second][cells[c].first] == 'P') {
      int v = n.s.v[6 + c];
      if (v == 3)
        ready += 20;
      else if (v >= 4)
        ready += 23 - v;
    }
  return n.delivered * 41 + stock * decay + ready * bonus;
}
int main(int argc, char **argv) {
  int width = argc > 1 ? atoi(argv[1]) : 5000;
  double ow = argc > 2 ? atof(argv[2]) : .02,
         pw = argc > 3 ? atof(argv[3]) : 40;
  if (width <= 0 || pw <= 0) {
    cerr << "Width and decay must be positive" << endl;
    return 1;
  }
  for (int x = 0; x < 5; x++)
    for (int y = 0; y < 5; y++)
      if (flooridx(x, y) < 0) {
        bool adj = false;
        for (auto &v : vec)
          if (flooridx(x + v[0], y + v[1]) >= 0)
            adj = true;
        if (adj)
          cells.push_back({x, y});
      }
  cerr << "cells=" << cells.size() << endl;
  vector<vector<Node>> history(401);
  Node start;
  start.s.v[0] = 1;
  start.s.v[3] = 7;
  history[0].push_back(start);
  for (int t = 1; t <= 400; t++) {
    current_time = t;
    unordered_map<State, Node, Hash> candidates;
    candidates.reserve(width * 20);
    auto &prev = history[t - 1];
    for (int i = 0; i < (int)prev.size(); i++)
      for (int act = 0; act < 36; act++) {
        Node n = step(prev[i], act);
        n.parent = i; // 19 deliveries require potential >= 19*41 = 779.
        if (potential(n) + 2 * (400 - t) < 779)
          continue;
        n.score = score(n, ow, pw);
        auto it = candidates.find(n.s);
        if (it == candidates.end())
          candidates.emplace(n.s, n);
        else if (n.delivered > it->second.delivered)
          it->second = n;
      }
    if (candidates.empty()) {
      cerr << "No retained path can reach 19 at " << t << endl;
      return 2;
    }
    auto &next = history[t];
    for (auto &kv : candidates)
      next.push_back(kv.second);
    if ((int)next.size() > width) {
      nth_element(
          next.begin(), next.begin() + width, next.end(),
          [](Node const &a, Node const &b) { return a.score > b.score; });
      next.resize(width);
    }
    if (t % 50 == 0)
      cerr << t << " "
           << max_element(
                  next.begin(), next.end(),
                  [](auto &a, auto &b) { return a.delivered < b.delivered; })
                  ->delivered
           << endl;
  }
  auto &last = history[400];
  int idx =
      max_element(last.begin(), last.end(),
                  [](auto &a, auto &b) {
                    return a.delivered < b.delivered ||
                           (a.delivered == b.delivered && a.score < b.score);
                  }) -
      last.begin();
  cerr << "deliveries=" << last[idx].delivered << endl;
  array<int, 400> actions;
  for (int t = 400; t > 0; t--) {
    auto &n = history[t][idx];
    actions[t - 1] = n.a;
    idx = n.parent;
  }
  cout << "[";
  for (int t = 0; t < 400; t++) {
    if (t)
      cout << ",";
    cout << "[" << actions[t] / 6 << "," << actions[t] % 6 << "]";
  }
  cout << "]\n";
}
