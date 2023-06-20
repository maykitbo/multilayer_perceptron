#pragma once


#include <vector>
#include <list>

#include "../common/strategy.h"

struct Node;

typedef std::pair<char, vector_f> letter;


struct Connection {
    float weight_;
    std::list<Node>::iterator next_, prev_;
    void Signal(float value);
    Connection(float weight, std::list<Node>::iterator &next) : weight_(weight), next_(next) {}
};

struct EdgeLayer {
    std::list<Connection> edges_;
    EdgeLayer() {}
    EdgeLayer(std::list<Node> *layer);
    void Signal(float value);
};

struct Node {
    float activation_, bias_;
    EdgeLayer edges_;
    Node() {}
    Node(std::list<Node> *layer) : edges_(layer) {}
    void Signal();
};

class GraphModel {
    std::vector<std::list<Node>> layers_;
    std::list<Node> output_;
    std::vector<EdgeLayer> start_edges_;
    float learning_rate_;
    void node_clear();
    public:
        GraphModel(int layers, float learning_rate = 1, int max = 784, int min = 26);
        void Fetch(letter &letter);

        void Print();
};

