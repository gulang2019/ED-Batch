#include "utils.h"
#include <vector>
#include <algorithm>
using namespace std;

namespace OoC
{
  Timer::Timer(const char *t) : type(t)
  {
    scope_tag = "default";
    scopes[scope_tag] = {};
  }
  void Timer::start(std::string key)
  {
    if (locked)
      return;
    auto &scope = scopes[scope_tag];
    if (scope.start_times.count(key) == 0)
      scope.start_times[key] = std::chrono::high_resolution_clock::now();
  }

  void Timer::stop(std::string key)
  {
    auto &scope = scopes[scope_tag];
    if (locked)
      return;
    if (!scope.start_times.count(key))
      return;
    double elapsed = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - scope.start_times[key]).count();
    if (type == "DEFAULT")
    {
      scope.values[key] = elapsed;
    }
    else if (type == "ADD")
    {
      if (scope.values.count(key) == 0)
        scope.values[key] = 0.0;
      scope.values[key] += elapsed;
    }
    scope.start_times.erase(key);
  }

  void Timer::lock() { locked = true; }
  void Timer::unlock() { locked = false; }

  // void Timer::show(std::string label, std::initializer_list<std::string> scope_tags, int repeat)
  // {
  //   show(label, std::vector<std::string>(scope_tags), repeat);
  // }

  void Timer::show(std::string label, std::vector<std::string> scope_tags, int repeat)
  {
    fprintf(stdout, "%s\n", label.c_str());
    for (auto scope_name : scope_tags)
    {
      auto &scope = scopes[scope_name];
      fprintf(stdout, "scope: %s\n", scope_name.c_str());
      vector<pair<string, double>> times;
      for (auto kv : scope.values)
        times.push_back(kv);
      sort(times.begin(), times.end(), [](pair<string, double> &v1, pair<string, double> &v2)
           { return v1.second < v2.second; });
      for (auto kv : times)
      {
        fprintf(stdout, "\t%s:\t%fms\n", kv.first.c_str(), kv.second / repeat);
      }
      for (auto kv : scope.int_values)
      {
        fprintf(stdout, "\t%s:\t%d\n", kv.first.c_str(), kv.second / repeat);
      }
      for (auto kv: scope.log_values) {
        fprintf(stdout, "\t%s:", kv.first.c_str());
        for (auto v:kv.second) fprintf(stdout, "%d,", v);
        fprintf(stdout, "\n");
      }
    }
  }
  void Timer::clear()
  {
    auto &scope = scopes[scope_tag];
    scope.values.clear();
    scope.start_times.clear();
    scope.log_values.clear();
    scope.int_values.clear();
  }

  void Timer::clearall(){
    scopes.clear();
    scopes["default"] = {};
    scope_tag = "default";
  }

  void Timer::cumint(std::string key, int v)
  {
    auto &scope = scopes[scope_tag];
    if (locked)
      return;
    if (scope.int_values.count(key) == 0)
    {
      scope.int_values[key] = 0;
    }
    scope.int_values[key] += v;
  }

  void Timer::log(std::string key, int v)
  {
    if (locked)
      return;
    auto &scope = scopes[scope_tag];
    if (scope.log_values.count(key) == 0)
      scope.log_values[key] = {};
    scope.log_values[key].push_back(v);
  }

  void Timer::save(std::string filename)
  {
    std::ofstream file;
    file.open(filename);
    file << "alg,metric,value" << std::endl; 
    for (auto item : scopes)
    {
      auto &name = item.first;
      auto &scope = item.second;
      for (auto kv : scope.values)
      {
        file << name << "," << kv.first << "," << kv.second << std::endl;
      }
      for (auto kv : scope.int_values)
      {
        file << name << "," << kv.first << "," << kv.second << std::endl;
      }
    }
    file.close();
  }

  double Timer::get(std::string key)
  {
    auto scope = scopes[scope_tag];
    if (scope.values.count(key) == 0)
      return 0.0f;
    return scope.values[key];
  }

  void Timer::scope(std::string key){
    scope_tag = key;
  }

} // namespace OoC