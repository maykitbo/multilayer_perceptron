#include "widget.h"


Widget::Widget(QWidget *parent) : QWidget(parent), ui(new Ui::Widget), layer_count_(0) {
    ui->setupUi(this);

    layer_pen_.setWidth(2);
    ui->layers_view->scale(1, -1);

    double double_num = 123456.78;
    // auto str = QString::number(double_num, 'f')

    layer_func_ = [&] (int k) { return k; };

    scene_ = new QGraphicsScene(ui->layers_view);

    for (int k = 0; k < Const::start_layer_count; ++k) {
        addLayer();
    }
    connect(ui->add_layer, &QPushButton::clicked, this, &Widget::addLayer);
    connect(ui->remove_layer, &QPushButton::clicked, this, &Widget::removeLayer);
    connect(ui->liner_func, &QCheckBox::clicked, [&] {
        layer_func_ = [&] (int k) {
            return layerFunc(Func::linearLayerReduction, k);
        };
    });
    connect(ui->manual_settings, &QCheckBox::clicked, [&] {
        layer_func_ = [&] (int k) { return manualLayers(k); };
    });
    connect(ui->layer_k1, &QDoubleSpinBox::valueChanged, [&] (double val) {
        drawLayers();
    });
    connect(ui->layer_k2, &QDoubleSpinBox::valueChanged, [&] (double val) {
        drawLayers();
    });
    connect(ui->func_1, &QCheckBox::clicked, [&] {
        layer_func_ = [&] (int k) {
            return layerFunc(Func::convexLayerReduction1, k);
        };
    });
    connect(ui->func_2, &QCheckBox::clicked, [&] {
        layer_func_ = [&] (int k) {
            return layerFunc(Func::convexLayerReduction2, k);
        };
    });
    connect(ui->layer_func_group, &QButtonGroup::idClicked, [&] (int k) {
        drawLayers();
    });

    ui->liner_func->click();
}

// int Widget::layerFunc(std::function<int(int, int)> func, int n, int k);


void Widget::showEvent(QShowEvent* event) {
    setLRect();
    drawLayers();
}

void Widget::addLayer() {
    if (layer_count_ >= Const::max_layer_count) return;

    ++layer_count_;

    auto spinbox = new QSpinBox(this);
    spinbox->setMaximum(Const::max_layer);
    connect(spinbox, &QSpinBox::valueChanged, this, [&] (int n) { drawLayers(); });
    ui->layers_layout->addWidget(spinbox);
    drawLayers();
}

void Widget::removeLayer() {
    if (layer_count_ <= Const::min_layer_count) return;

    --layer_count_;

    delete ui->layers_layout->itemAt(layer_count_ - 1)->widget();
    drawLayers();
}

void Widget::resizeEvent(QResizeEvent *event) {
    setLRect();
}

void Widget::setLRect() {
    l_width_ = ui->layers_view->width();
    l_height_ = ui->layers_view->height() - 6;
    l_indent_ = ui->layer_in_size->width() / 2;
}

void Widget::drawLayers() {
    scene_->clear();

    // auto add_line = [&] (int x, int y) { scene_->addLine(x, 0, x, y, layer_pen_); };

    addLine(l_indent_, l_height_);
    for (int k = 0; k < layer_count_; ++k) {
        addLine((k + 1) * l_width_ / (layer_count_ + 1),
                l_height_ * layer_func_(k) / Const::max_layer);
    }
    addLine(l_width_ - l_indent_,
            l_height_ * Const::min_layer / Const::max_layer);

    ui->layers_view->setScene(scene_);
}

void Widget::addLine(int x, int y) {
    scene_->addLine(x, 0, x, y, layer_pen_);
}

int Widget::manualLayers(int k) {
    return static_cast<QSpinBox*>(ui->layers_layout->itemAt(k)->widget())->value();
}


Widget::~Widget() {
    delete ui;
    delete scene_;
}

