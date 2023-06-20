#include "graph_model.h"

EdgeLayer::EdgeLayer(std::list<Node> *layer) {
    std::mt19937 generator{std::random_device{}()};
    std::normal_distribution<float> distribution{Const::w_mean, Const::w_sd};
    for (auto iter = layer->begin(); iter != layer->end(); ++iter) {
        // edges_.emplace_back(1.0, iter);
        edges_.emplace_back(distribution(generator), iter);
    }
}

void EdgeLayer::Signal(float value) {
    for (auto &edge : edges_) {
        edge.Signal(value);
    }
}

void Node::Signal() {
    activation_ += bias_;
    edges_.Signal(activation_);
}

void Connection::Signal(float value) {
    next_->activation_ += Func::Sigmoid(value) * weight_;
}

void GraphModel::node_clear() {
    for (auto &layer : layers_) {
        for (auto &neuron : layer) {
            neuron.activation_ = 0;
        }
    }
    for (auto &neuron : output_) {
        neuron.activation_ = 0;
    }
}

GraphModel::GraphModel(int layers, float learning_rate, int max, int min) :
                learning_rate_(learning_rate), output_(min), layers_(layers - 1) {
    std::list<Node> *layer = &output_;
    for (int k = layers - 1; k >= 1; --k) {
        int cols = Func::LayerReduction(max, min, layers, k);
        for (int g = 0; g < cols; ++g) {
            layers_[k - 1].emplace_back(layer);
        }
        layer = &layers_[k - 1];
    }
    for (int k = 0; k < max; ++k) {
        start_edges_.emplace_back(layer);
    }
}

void GraphModel::Fetch(letter &letter) {
    node_clear();
    for (int k = 0; k < start_edges_.size(); ++k) {
        start_edges_[k].Signal(letter.second[k]);
    }
    for (auto &layer : layers_) {
        for (auto &neuron : layer) {
            neuron.Signal();
        }
    }
}




void GraphModel::Print() {
    std::cout << start_edges_.size() << " ";
    for (auto &i : layers_) {
        std::cout << i.size() << " ";
    }
    std::cout << output_.size() << "\n";

    for (auto &i : output_) {
        printf("%.8f ", Func::Sigmoid(i.activation_));
    }
    std::cout << "\n";
}
