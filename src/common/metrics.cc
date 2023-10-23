#include "metrics.h"

using namespace s21;

Metrics::Metrics(unsigned int size) : precision(size),
    recall(size), F1(size) {}

MetricsMaker::MetricsMaker(unsigned int size) : metrics_(size),
    precision_helper_(size, std::make_pair(0, 0)),
    recall_helper_(size, std::make_pair(0, 0)),
    size_(size) {}

void MetricsMaker::operator()(int result, int expected) {
    ++(precision_helper_[result].first);
    ++(recall_helper_[expected].first);
    if (result == expected) {
        ++accuracy_helper_;
        ++(precision_helper_[result].second);
        ++(recall_helper_[result].second);
    }
    ++count_;
}
Metrics &MetricsMaker::operator()() {
    metrics_.accuracy = (double)accuracy_helper_ / (double)count_;
    auto ratio = [&] (std::pair<int, int> p) { return (double)p.second / (double)p.first; };
    for (unsigned int k = 0; k < size_; ++k) {
        metrics_.precision[k] = ratio(precision_helper_[k]);
        metrics_.recall[k] = ratio(recall_helper_[k]);
        metrics_.F1[k] = 2.0 * metrics_.precision[k] * metrics_.recall[k] / (metrics_.precision[k] + metrics_.recall[k]);
    }
    return metrics_;
}

void MetricsMaker::Clear() {
    metrics_.accuracy = 0.0;
    count_ = 0;
    accuracy_helper_ = 0;
    for (unsigned int k = 0; k < Const::min_layer; ++k) {
        metrics_.precision[k] = 0;
        metrics_.recall[k] = 0;
        metrics_.F1[k] = 0;
        precision_helper_[k] = std::make_pair(0, 0);
        recall_helper_[k] = std::make_pair(0, 0);
    }
}


MappedLettersMetrics Metrics::GetMappedLettersMetrics() {
    unsigned int size = precision.size();
    MappedLettersMetrics metrics(size);

    metrics.accuracy = accuracy;
    metrics.training_time = training_time;
    metrics.testing_time = testing_time;
    for (unsigned int k = 0; k < size; ++k) {
        metrics.precision_map[k] = std::make_pair(k + 'A', precision[k]);
        metrics.recall_map[k] = std::make_pair(k + 'A', recall[k]);
        metrics.F1_map[k] = std::make_pair(k + 'A', F1[k]);
    }
    return metrics;
}
