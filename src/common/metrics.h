#pragma once

#include <vector>
#include <iostream>
#include "strategy.h"

#include "types.h"

namespace s21 {

struct Metrics {
    std::vector<fp_type> precision, recall, F1;
    fp_type accuracy = 0.0;
    Metrics();
};

class MetricsMaker {
    public:
        MetricsMaker();

        void operator()(int result, int expected);
        Metrics &operator()();
        void Clear();

    private:
        Metrics metrics_;
        int count_ = 0, accuracy_helper_ = 0;
        std::vector<std::pair<int, int>> precision_helper_;
        std::vector<std::pair<int, int>> recall_helper_;
};

} // namespace s21
