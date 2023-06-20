#include "metrics.h"

using namespace s21;

Metrics::Metrics() : precision(Const::min_layer),
    recall(Const::min_layer), F1(Const::min_layer) {}

MetricsMaker::MetricsMaker() : metrics_(),
    precision_helper_(Const::min_layer, std::make_pair(0, 0)),
    recall_helper_(Const::min_layer, std::make_pair(0, 0)) {}

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
    for (int k = 0; k < Const::min_layer; ++k) {
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
    for (int k = 0; k < Const::min_layer; ++k) {
        metrics_.precision[k] = 0;
        metrics_.recall[k] = 0;
        metrics_.F1[k] = 0;
        precision_helper_[k] = std::make_pair(0, 0);
        recall_helper_[k] = std::make_pair(0, 0);
    }
}
