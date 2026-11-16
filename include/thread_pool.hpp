// =====================================================================================
// thread_pool.hpp — minimal persistent pool for data-parallel loops
// =====================================================================================
// run(tasks, fn) calls fn(0..tasks-1) spread over the workers AND the calling thread and
// returns when all are done. Workers spin briefly before sleeping so back-to-back calls (one
// per image stage) do not pay a full OS wake-up each time.
// =====================================================================================
#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

class ThreadPool {
public:
    // `threads` = total parallelism including the caller (1 = no workers, runs inline).
    explicit ThreadPool(int threads) : total_(threads < 1 ? 1 : threads) {
        for (int i = 1; i < total_; i++) workers_.emplace_back([this] { workerLoop(); });
    }
    ~ThreadPool() {
        { std::lock_guard<std::mutex> lk(m_); stop_ = true; generation_.fetch_add(1); }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
    }
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int size() const { return total_; }

    void run(int tasks, const std::function<void(int)>& fn) {
        if (tasks <= 0) return;
        if (total_ == 1 || tasks == 1) { for (int i = 0; i < tasks; i++) fn(i); return; }
        {
            std::lock_guard<std::mutex> lk(m_);
            fn_ = &fn; tasks_ = tasks; next_.store(0); busy_.store(static_cast<int>(workers_.size()));
            generation_.fetch_add(1);
        }
        cv_.notify_all();
        drain();                                           // the caller works too
        while (busy_.load(std::memory_order_acquire) != 0) std::this_thread::yield();
    }

private:
    void drain() {
        for (int i; (i = next_.fetch_add(1)) < tasks_;) (*fn_)(i);
    }
    void workerLoop() {
        unsigned seen = generation_.load();
        for (;;) {
            // spin for a short while, then sleep on the condition variable
            for (int spin = 0; spin < 20000 && generation_.load(std::memory_order_acquire) == seen; spin++) std::this_thread::yield();
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return generation_.load() != seen; });
                if (stop_) return;
                seen = generation_.load();
            }
            drain();
            busy_.fetch_sub(1, std::memory_order_release);
        }
    }

    int total_;
    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cv_;
    std::atomic<unsigned> generation_{0};
    std::atomic<int> next_{0}, busy_{0};
    const std::function<void(int)>* fn_ = nullptr;
    int tasks_ = 0;
    bool stop_ = false;
};
