#pragma once

#include <QWidget>
#include <QGraphicsScene>
// #include <QCheckBox>
// #include <QPushButton>

#include "./ui_widget.h"
// #include "../build/mlp_autogen/include/ui_widget.h"

#include <functional>

#include <controller.h>

#include <iostream>

// #define START_LAYER_COUNT 5


QT_BEGIN_NAMESPACE
namespace Ui { class Widget; }
QT_END_NAMESPACE

class Widget : public QWidget {
    Q_OBJECT

    public:
        Widget(QWidget *parent = nullptr);
        ~Widget();

    private:
        Ui::Widget *ui;
        Controller control_;
        QGraphicsScene *scene_;
        QPen layer_pen_ = QPen(Qt::red);

        int l_width_;
        int l_height_;
        int l_indent_;
        int layer_count_;

        void setLRect();

        void addLine(int x, int y);

        std::function<int(int)> layer_func_;

        int layerFunc(std::function<double(double, double, double, double)> func, int k) {
            int size = func(layer_count_ + 1, k + 1, ui->layer_k1->value(), ui->layer_k2->value());
            auto object = static_cast<QSpinBox*>(ui->layers_layout->itemAt(k)->widget());
            object->blockSignals(true);
            object->setValue(size);
            object->blockSignals(false);
            return size;
        }

        void drawLayers();

        void showEvent(QShowEvent* event) override;
        void resizeEvent(QResizeEvent *event) override;

        int manualLayers(int k);

    private slots:
        void addLayer();
        void removeLayer();
};
