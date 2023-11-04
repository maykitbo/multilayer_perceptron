#pragma once

#include "painters/background.h"
#include "painters/painter.h"
#include "painters/colors_set.h"

#include <vector>
#include <iostream>

#include <QGuiApplication>

#include <cstdlib> // For std::system


namespace Graph {

class ImagePainter
{
    public:
        ImagePainter(const QSize &size, int argc, char *argv[]);

        Painter &AddGraph(const data_t &data,
                        const QString &name,
                        const QPen &graph_pen, bool points = false, bool lines = true);

        Painter &AddGraph(Graph::data_t &&data,
                        const QString &name,
                        const QPen &graph_pen, bool points = false, bool lines = true);
        
        void AddGraph(const Painter &graph);
        void AddGraph(Painter &&graph);

        void ReserveGraphs(unsigned size);
        
        Painter &AddGraph(const Graph::data_t &data, const QString &name, bool points = false, bool lines = true);
        Painter &AddGraph(Graph::data_t &&data, const QString &name, bool points = false, bool lines = true);

        void SavePng(const QString &path);
        
        Background &GetBackground();
        Painter &operator[](unsigned index);
    
    private:
        QGuiApplication app_;
        Parameters p_;

        Background background_;
        std::vector<Painter> graphs_;

        void OpenPng(const QString &path);

        ColorsSet<std::vector<QColor>> colors_;      
};

} // namespace Graph
