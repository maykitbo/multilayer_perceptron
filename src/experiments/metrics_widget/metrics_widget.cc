#include "metrics_widget.h"

using namespace s21;

MetricsWidget::MetricsWidget(QWidget *parent) :
        QWidget(parent), image_(QSize(1, 1), QImage::Format_RGB32),
        classes_precision_(Style::classes), classes_recall_(Style::classes), classes_f1_(Style::classes) {

    image_.fill(Style::background_color);

}

void MetricsWidget::SetMetrics(Metrics &m) {
    accuracy_.push_back(m.accuracy);
    precision_.push_back(Func::MeanError(m.precision));
    recall_.push_back(Func::MeanError(m.recall));
    f1_.push_back(Func::MeanError(m.F1));
    if (count_ == 0) {
        classes_precision_ = m.precision;
        classes_recall_ = m.recall;
        classes_f1_ = m.F1;
    } else {
        unsigned int size = classes_precision_.size();            
        for (unsigned int i = 0; i < size; ++i) {
            classes_precision_[i] = (classes_precision_[i] * count_ + m.precision[i]) / (count_ + 1);
            classes_recall_[i] = (classes_recall_[i] * count_ + m.recall[i]) / (count_ + 1);
            classes_f1_[i] = (classes_f1_[i] * count_ + m.F1[i]) / (count_ + 1);
        }
    }
    ++count_;
    Draw();
}

void MetricsWidget::SetVector(const std::vector<double> &v)
{
    data_.resize(v.size());
    for (unsigned i = 0; i < v.size(); ++i) {
        data_[i] = v[i];
        if (data_[i] > max_) max_ = data_[i];
    }
    max_ *= 1.1;
    for (unsigned i = 0; i < v.size(); ++i) {
        data_[i] /= max_;
    }
    Draw();
}

void MetricsWidget::DrawVector()
{
    std::vector<std::vector<double>*> data = {&data_};
    std::vector<QString> names = {""};
    std::vector<QColor> colors = {Style::accuracy_color};
    DrawCharts(colors, names, data, [](int i) { return QString((char)(i + 'A')); });
}



void MetricsWidget::paintEvent(QPaintEvent *event) {
    QPainter(this).drawImage(0, 0, image_);
    QWidget::paintEvent(event);
}

void MetricsWidget::resizeEvent(QResizeEvent *event) {
    height_ = event->size().height();
    width_ = event->size().width();
    image_ = image_.scaled(event->size(), Qt::IgnoreAspectRatio);
    QWidget::resizeEvent(event);
    Draw();
}

void MetricsWidget::DrawCharts(const std::vector<QColor> &colors,
                const std::vector<QString> &names, const std::vector<std::vector<double>*> data,
                const std::function<QString(int)> &chart_name) {

    unsigned int size = data[0]->size();
    unsigned int metrics_size = data.size();
    int w = width_ - 2 * Style::indent;
    int chart_size = (Style::chart_size_coef * w / size) / metrics_size;
    int x = w / (size + 1) - (metrics_size - 1) * chart_size / 2 + Style::indent;
    int step = w / (size + 1) - (metrics_size - 1) * chart_size;
    int y_delta = chart_size * 0.5;
    int y_low = height_ - Style::indent - y_delta;

    for (unsigned int i = 0; i < size; ++i, x += step) {
        for (unsigned int k = 0; k < metrics_size; ++k) {
            painter_.setPen(QPen(colors[k], chart_size));
            painter_.drawLine(x, y_low, x, y_low - ChartHeight((*data[k])[i]) + 2 * y_delta);
            if (k != metrics_size - 1) x += chart_size;
        }
        painter_.setPen(QPen(Style::axes_text_color, Style::axes_text_size));
        painter_.drawText(x - chart_size * 1.5, height_ - Style::indent * 0.5, chart_name(i));
        // painter_.drawText(x - chart_size * 0.4, height_ - Style::indent * 0.65, chart_name(i));
    }
    int distanse = w / (metrics_size + 1);
    x = distanse;
    for (unsigned int k = 0; k < metrics_size; ++k, x += distanse) {
        painter_.setPen(QPen(colors[k], Style::metrics_text_size));
        painter_.drawText(x, Style::indent * 0.5, names[k]);
    }
}

void MetricsWidget::DrawCrossCharts() {
    std::vector<QColor> colors = {Style::accuracy_color, Style::precision_color,
                                  Style::recall_color, Style::f1_color};

    std::vector<QString> names = {"Accuracy", "Precision", "Recall", "F1"};
    std::vector<std::vector<double>*> data = {&accuracy_, &precision_, &recall_, &f1_};

    DrawCharts(colors, names, data, [](int i) { return QString::number(i + 1); });
}

int MetricsWidget::ChartHeight(double value) {
    if (isnan(value)) return 0;
    return value * (height_ - 2 * Style::indent);
}


void MetricsWidget::DrawAxes() {
    image_.fill(Style::background_color);
    painter_.begin(&image_);

    // draw gorizontal line
    painter_.setPen(QPen(Style::axes_color, Style::axes_size));
    int y = height_ - Style::indent;
    painter_.drawLine(Style::indent,
                        y,
                        width_ - Style::indent,
                        y);
    
    // draw vertical line
    int x = Style::indent;
    painter_.drawLine(x,
                        Style::indent,
                        x,
                        height_ - Style::indent);
    
    painter_.setPen(QPen(Style::grid_color, Style::grid_size));

    // draw gorizontal grid
    for (int k = 0; k < Style::grid_number; ++k) {
        painter_.drawLine(x,
                            Style::indent + k * (height_ - 2 * Style::indent) / Style::grid_number,
                            width_ - Style::indent,
                            Style::indent + k * (height_ - 2 * Style::indent) / Style::grid_number);
    }

    // draw y axis text
    double step = 1.0 / Style::grid_number;
    painter_.setPen(QPen(Style::axes_text_color, Style::axes_text_size));
    for (int k = 0; k < Style::grid_number; ++k) {
        painter_.drawText(Style::indent * 0.2,
                            Style::indent + k * (height_ - 2 * Style::indent) / Style::grid_number,
                            // QString::number((1.0 - k * step) * max_, 'f', 2));
                            QString::number(1.0 - k * step, 'f', 2));
    }
}

void MetricsWidget::DrawClassCharts() {
    std::vector<QColor> colors = {Style::precision_color, Style::recall_color, Style::f1_color};
    std::vector<QString> names = {"Precision", "Recall", "F1"};
    std::vector<std::vector<double>*> data = {&classes_precision_, &classes_recall_, &classes_f1_};
    DrawCharts(colors, names, data, [](int i) { return QString((char)(i + 'A')); }); 
}

void MetricsWidget::Draw() {
    DrawAxes();

    // if (!accuracy_.empty()) {
    //     classes_ ? DrawClassCharts() : DrawCrossCharts();
    // } else {
    //     DrawVector();
    // }
    accuracy_.empty() ? DrawVector() : classes_ ? DrawClassCharts() : DrawCrossCharts();

    painter_.end();
    update();
}

void MetricsWidget::Clear() {
    accuracy_.clear();
    precision_.clear();
    recall_.clear();
    f1_.clear();
    count_ = 0;
    std::fill(classes_precision_.begin(), classes_precision_.end(), 0);
    std::fill(classes_recall_.begin(), classes_recall_.end(), 0);
    std::fill(classes_f1_.begin(), classes_f1_.end(), 0);

    DrawAxes();
    painter_.end();

    update();
}

void MetricsWidget::mouseDoubleClickEvent(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton) {
        classes_ = !classes_;
        Draw();
    }
}

