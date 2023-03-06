#pragma once

#include <string>
#include <unordered_map>
#include <chrono>
#include <functional>
#include <fstream>
#include <set>
#include <vector>
#include <map>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <cassert>
#include <stdexcept>
#include <iostream>

namespace OoC
{
  struct TimerValue
  {
    std::map<std::string, double> values;
    std::map<std::string, int> int_values;
    std::map<std::string, std::vector<int>> log_values;
    std::map<std::string, std::chrono::high_resolution_clock::time_point> start_times;
  };

  struct Timer
  {
    std::string type;
    std::string scope_tag;
    Timer(const char *t = "ADD");
    void start(std::string key);
    void stop(std::string key);
    void scope(std::string key);
    void lock();
    void unlock();

    // void show(std::string label = "Timer Show", std::initializer_list<std::string> scope_tags = {"default"}, int repeat = 1);
    void show(std::string label = "Timer Show", std::vector<std::string> scope_tags = {"default"}, int repeat = 1);
    void clear();
    void clearall();

    void cumint(std::string key, int v);

    void log(std::string key, int v);

    void save(std::string filename);

    double get(std::string key);

    std::unordered_map<std::string, TimerValue> scopes;

    bool locked = false;
  };
  template <typename T>
  struct TupleDict
  {
    struct Node
    {
      bool isLeaf = false;
      T *data = nullptr;
      std::unordered_map<int, Node *> next;
    } head;

    std::vector<T *> data;

    void operator=(TupleDict &&other)
    {
      clear();
      head = other.head;
      data = other.data;
      other.data.clear();
      other.head.next.clear();
    }

    Node* deepcopy(const Node* node, Node* new_node){
      if (new_node == nullptr) 
        new_node = new Node;
      new_node->isLeaf = node->isLeaf;
      if (node->isLeaf) {
        assert(node->data != nullptr);
        new_node->data = new T(*node->data);
        data.push_back(new_node->data);
      }
      for (auto kv:node->next){
        new_node->next[kv.first] = deepcopy(kv.second, nullptr);
      }
      return new_node;
    }

    void operator=(const TupleDict<T>& other){
      clear();
      deepcopy(&other.head, &head);
    }

    T &operator[](const std::set<int> &key)
    {
      Node *node = &head;
      for (auto v : key)
      {
        if (node->next.count(v) == 0)
          node->next[v] = new Node;
        node = node->next[v];
      }
      if (!node->isLeaf)
      {
        node->isLeaf = true;
        node->data = new T({});
        data.push_back(node->data);
      }
      return *(node->data);
    }

    T &operator[](const std::vector<int> &key)
    {
      Node *node = &head;
      for (auto v : key)
      {
        if (node->next.count(v) == 0)
          node->next[v] = new Node;
        node = node->next[v];
      }
      if (!node->isLeaf)
      {
        node->isLeaf = true;
        node->data = new T;
        data.push_back(node->data);
      }
      return *(node->data);
    }

    bool count(const std::set<int> &key) const
    {
      const Node *node = &head;
      assert(node);
      for (auto &v : key)
      {
        if (node->next.count(v) == 0)
          return false;
        node = node->next.at(v);
      }
      return node->isLeaf;
    }

    bool count(const std::vector<int> &key) const
    {
      const Node *node = &head;
      for (auto &v : key)
      {
        if (node->next.count(v) == 0)
          return false;
        node = node->next.at(v);
      }
      return node->isLeaf;
    }

    T *get(const std::set<int> &key) const
    {
      const Node *node = &head;
      for (auto &v : key)
      {
        if (node->next.count(v) == 0)
          return nullptr;
        node = node->next.at(v);
      }
      return node->data;
    }

    T *get(const std::vector<int> &key) const
    {
      const Node *node = &head;
      for (auto &v : key)
      {
        if (node->next.count(v) == 0)
          return nullptr;
        node = node->next.at(v);
      }
      return node->data;
    }

    void clear()
    {
      for (auto ptr : data)
      {
        delete ptr;
      }
      data.clear();
      std::function<void(Node *)> erase = [&](Node *node)
      {
        if (node == nullptr)
          return;
        for (auto kv : node->next)
        {
          if (kv.second)
          {
            erase(kv.second);
            delete kv.second;
          }
        }
        node->next.clear();
      };
      erase(&head);
      assert(head.next.size() == 0);
    }

    void reporter(std::function<void(T *)> &func, Node *node, std::vector<int> &vec)
    {
      if (node->isLeaf)
      {
        for (auto e : vec)
          std::cout << e << ",";
        std::cout << ":";
        func(node->data);
        std::cout << std::endl;
      }
      for (auto kv : node->next)
      {
        vec.push_back(kv.first);
        reporter(func, kv.second, vec);
        vec.pop_back();
      }
    }

    void report(std::function<void(T *)> &func)
    {
      std::cout << "-----------report------------" << std::endl;
      std::vector<int> vec;
      reporter(func, &head, vec);
      std::cout << "--------report end-----------" << std::endl;
    }

    ~TupleDict()
    {
      clear();
    }
  };
} // namespace OoC

#ifndef THREAD_POOL_H
#define THREAD_POOL_H

class ThreadPool
{
public:
  ThreadPool(size_t);
  template <class F, class... Args>
  auto enqueue(F &&f, Args &&...args)
      -> std::future<typename std::result_of<F(Args...)>::type>;
  ~ThreadPool();

private:
  // need to keep track of threads so we can join them
  std::vector<std::thread> workers;
  // the task queue
  std::queue<std::function<void()>> tasks;

  // synchronization
  std::mutex queue_mutex;
  std::condition_variable condition;
  bool stop;
};

// the constructor just launches some amount of workers
inline ThreadPool::ThreadPool(size_t threads)
    : stop(false)
{
  for (size_t i = 0; i < threads; ++i)
    workers.emplace_back(
        [this]
        {
          for (;;)
          {
            std::function<void()> task;
            {
              std::unique_lock<std::mutex> lock(this->queue_mutex);
              this->condition.wait(lock,
                                   [this]
                                   { return this->stop || !this->tasks.empty(); });
              if (this->stop && this->tasks.empty())
                return;
              task = std::move(this->tasks.front());
              this->tasks.pop();
            }
            task();
          }
        });
}

// add new work item to the pool
template <class F, class... Args>
auto ThreadPool::enqueue(F &&f, Args &&...args)
    -> std::future<typename std::result_of<F(Args...)>::type>
{
  using return_type = typename std::result_of<F(Args...)>::type;

  auto task = std::make_shared<std::packaged_task<return_type()>>(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...));

  std::future<return_type> res = task->get_future();
  {
    std::unique_lock<std::mutex> lock(queue_mutex);

    // don't allow enqueueing after stopping the pool
    if (stop)
      throw std::runtime_error("enqueue on stopped ThreadPool");

    tasks.emplace([task]()
                  { (*task)(); });
  }
  condition.notify_one();
  return res;
}

// the destructor joins all threads
inline ThreadPool::~ThreadPool()
{
  {
    std::unique_lock<std::mutex> lock(queue_mutex);
    stop = true;
  }
  condition.notify_all();
  for (std::thread &worker : workers)
    worker.join();
}

#endif
