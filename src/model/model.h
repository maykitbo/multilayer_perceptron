#pragma once

#include <vector>
#include "data_manager.h"
#include "../common/metrics.h"

namespace s21 {

class Model {
    public:
        Model() {}
        void Learn(DataManager &letters, int epoch_count) {
            // MetricsMaker metrics;

            int count = 0;
            for (int k = 0; k < epoch_count; ++k) {
                letters.ForTrain([&] (data_vector &letter, int name) {
                    letter_ = &letter;
                    Forward();

                    // metrics(GetResult(), name);
                    // if (((++count) % (letters.Size() / 5)) == 0) {
                    //     std::cout << metrics().accuracy << " accuracy\n";
                    //     metrics.Clear();
                        
                    // }

                    Backward(name);
                    
                });
    
                // if (k != epoch_count - 1) {
                    MetricsMaker epoch_metrics;
                    letters.ForTest([&] (data_vector &letter, int name) {
                        letter_ = &letter;
                        Forward();
                        epoch_metrics(GetResult(), name);
                    });
                    const auto &m = epoch_metrics();
                    std::cout << m.accuracy << "; ";
                    for (auto i : m.precision) printf("%.3lf ", i);
                    std::cout << '\n';
                    letters.SetMetric(m.precision);
                    UpdateLR();
                // }
            }
            
        }
        virtual ~Model() = default;
        Metrics Test(DataManager &letters) {
            MetricsMaker metrics;

            letters.ForTest([&] (data_vector &letter, int name) {
                letter_ = &letter;
                Forward();
                metrics(GetResult(), name);
            });
            return metrics();
        }
        virtual void Forward() = 0;
        virtual void Backward(int answer) = 0;
        virtual int GetResult() = 0;
        virtual void UpdateLR() = 0;

    protected:
        data_vector *letter_;
};

} // namespace s21
