#pragma once

#include <QWidget>
#include <QImage>
#include <QPainter>
#include <QKeyEvent>
#include <QColor>
#include <QMouseEvent>
#include "../../common/metrics.h"
#include "view_constants.h"


QT_BEGIN_NAMESPACE
namespace Ui { class Widget; }
QT_END_NAMESPACE

namespace s21 {

/**
 * @brief Widget for displaying metrics
 */
class MetricsWidget : public QWidget {
    Q_OBJECT

    using Style = MetricsWidgetStyle;
    public:
        /**
         * @brief Construct a new Metrics Widget object
         * 
         * @param parent 
         */
        MetricsWidget(QWidget *parent = nullptr);
        virtual ~MetricsWidget() = default;

        /**
         * @brief Set the Metrics object
         * Udate the metrics and repaint the widget
         * @param m new metrix
         */
        void SetMetrics(Metrics &m);
        
        /**
         * @brief Clear the metrics and repaint the widget
         * 
         */
        void Clear();

        void SetVector(const std::vector<double> &v);

    protected:
        void paintEvent(QPaintEvent *event) override;
        void resizeEvent(QResizeEvent *event) override;

        /**
         * @brief change the view of the metrics
         * Types of view:
         * 1. Metrices charts. One chart for each metric
         * 2. Classes charts. One chart for each class
         * 
         * @param event 
         */
        void mouseDoubleClickEvent(QMouseEvent *event) override;

    private:

        QImage image_;
        QPainter painter_;

        bool classes_ = false;

        std::vector<double> accuracy_, precision_, recall_, f1_,
                classes_precision_, classes_recall_, classes_f1_;
        std::vector<double> data_;
        unsigned int count_ = 0;
        double max_ = 1.0;

        unsigned int height_, width_;

        void DrawCrossCharts();
        void DrawClassCharts();
        void DrawVector();
        void Draw();
        void DrawAxes();
        int ChartHeight(double value);

        void DrawCharts(const std::vector<QColor> &colors,
                        const std::vector<QString> &names, const std::vector<std::vector<double>*> data,
                        const std::function<QString(int)> &chart_name);

};

} // namespace s21
