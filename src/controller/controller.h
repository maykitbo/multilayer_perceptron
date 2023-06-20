#pragma once

#include <vector>
#include "strategy.h"
#include "../model/model.h"

#include <memory>

// #define FIRST_LAYER_SIZE 784
// #define LAST_LAYER_SIZE 26

// enum LayerFunc {
//     linear, convex_up, convex_down, manual
// }

class Controller {
    private:
        // std::unique_ptr<Model> model_;
        
    
    public:
        // std::vector<int> linearFunc(int n) {
        //     std::vector<int> sizes(n);

        //     for (int i = 0; i < n; ++i) {
        //         sizes[i] = Const::max_layer - (i+1) * (Const::max_layer - Const::min_layer) / (n+1);
        //     }

        //     return sizes;
        // }

        void Test();
        void Train();
        void OneLetter();
        void Create(const std::vector<int> &layer_sizes, float learning_rate);
        void Open(const std::string &file_name);
        void Save(const std::string &file_name);

};
