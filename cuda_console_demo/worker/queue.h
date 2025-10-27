#ifndef QUEUE_H
#define QUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <string>
#include <QString>

struct PersonInfo {
    int id;
    QString name;
    QString timestamp;
    int frameno;
}

using WorkItem = std::string;

class WorkQueue {
public:
    void push(const WorkItem& item);
    WorkItem pop();
    bool empty() const;
    size_t size() const;

private:
    std::queue<WorkItem> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cond_var_;
};

#endif // QUEUE_H