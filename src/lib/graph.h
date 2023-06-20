#pragma once


template<class T>
class Node {
    T data_;
};


template<class T>
class Edge {
    T data_;
    Node *source_, *destination_;
};

