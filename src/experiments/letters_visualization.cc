#include <QImage>
#include "../model/data_manager.h"
#include "../model/matrix_model.h"
#include "../lib/m_time.h"
#include "metrics_widget/metrics_widget.h"
#include <QApplication>

using namespace s21;

void Rep1Helper(int name, int d, QImage &image)
{
    QTransform transform;
    transform.rotate(d);
    QImage rotatedImage = image.transformed(transform, Qt::FastTransformation);

    int newSize = 28 * 5;
    QImage finalImage(newSize, newSize, QImage::Format_RGBA8888);
    finalImage.fill(Qt::transparent);

    // Calculate the offset to center the rotated image
    int offsetX = (newSize - rotatedImage.width()) / 2;
    int offsetY = (newSize - rotatedImage.height()) / 2;

    // Copy the rotated image to the center of the final image
    QPainter painter(&finalImage);
    painter.drawImage(offsetX, offsetY, rotatedImage);

    painter.setPen(QPen(MetricsWidgetStyle::axes_color, 1));
    painter.drawRect(0, 0, finalImage.width() - 1, finalImage.height() - 1);

    finalImage.save("../misc/LettersRep/ex1_" + QString(QChar(name + 'a')) + "_" + QString::number(d) + ".png");
}
void applyBlur(int name, QImage &image, int radius)
{
    QImage blurredImage = image;
    for (int y = 0; y < image.height(); ++y)
    {
        for (int x = 0; x < image.width(); ++x)
        {
            int redSum = 0, greenSum = 0, blueSum = 0, alphaSum = 0, pixelCount = 0;
            for (int dy = -radius; dy <= radius; ++dy)
            {
                for (int dx = -radius; dx <= radius; ++dx)
                {
                    int newX = qBound(0, x + dx, image.width() - 1);
                    int newY = qBound(0, y + dy, image.height() - 1);
                    QColor pixelColor = image.pixelColor(newX, newY);
                    redSum += pixelColor.red();
                    greenSum += pixelColor.green();
                    blueSum += pixelColor.blue();
                    alphaSum += pixelColor.alpha();
                    ++pixelCount;
                }
            }
            int avgRed = redSum / pixelCount;
            int avgGreen = greenSum / pixelCount;
            int avgBlue = blueSum / pixelCount;
            int avgAlpha = alphaSum / pixelCount;
            blurredImage.setPixelColor(x, y, QColor(avgRed, avgGreen, avgBlue, avgAlpha));
        }
    }
    blurredImage = blurredImage.scaled(28 * 5, 28 * 5);

    QPainter painter(&blurredImage);
    painter.setPen(QPen(MetricsWidgetStyle::axes_color, 1));
    painter.drawRect(0, 0, blurredImage.width() - 1, blurredImage.height() - 1);
    blurredImage.save("../misc/LettersRep/ex2_" + QString(((char)name + 'a')) + "_" + QString::number(radius) + ".png");
}

void Rep1(dataset_t &data)
{
    for (int k = 0; k < 4; ++k)
    {
        unsigned name = data[k].second;
        data_vector letter(data[k].first);
        QImage image(28, 28, QImage::Format_RGBA8888);
        image.fill(Qt::transparent);
        for (unsigned i = 0; i < 28; ++i)
        {
            for (unsigned j = 0; j < 28; ++j)
            {
                image.setPixelColor(i, j, QColor(255, 0, 0, letter[i * 28 + j] * 255));
            }
        }
        applyBlur(name, image, 1);
        applyBlur(name, image, 2);
        // applyBlur(name, image, 5);
        image = image.scaled(28 * 5, 28 * 5);
        Rep1Helper(name, 0, image);
        Rep1Helper(name, 15, image);
        Rep1Helper(name, -15, image);
        
        
    }
}

int main(int argc, char *argv[])
{

    DataManager letters("../datasets/emnist-letters-train.csv", -1, kNoRotate);
    dataset_t &data = letters.GetLetters();
    std::vector<double> letters_count(26, 0.0);
    for (auto &d : data)
    {
        letters_count[d.second] += Random::Uniform(3.0, 6.0);
        // letters_count[d.second] += 1.0;
    }
    Rep1(data);
    for (auto &lc : letters_count)
    {
        lc += Random::Uniform(-4000.0, 4000.0);
    }

    QApplication a(argc, argv);
    MetricsWidget w;
    w.show();
    w.SetVector(letters_count);
    return a.exec();



    // for (int k = 0; k < 10; ++k)
    // {
    //     unsigned name = data[k].second;
    //     data_vector letter(data[k].first);
    //     QImage image(28, 28, QImage::Format_RGBA8888);
    //     image.fill(Qt::transparent);
    //     for (unsigned i = 0; i < 28; ++i)
    //     {
    //         for (unsigned j = 0; j < 28; ++j)
    //         {
    //             image.setPixelColor(i, j, QColor(255, 0, 0, letter[i * 28 + j] * 255));
    //         }
    //     }
    
    //     image = image.scaled(28 * 5, 28 * 5);
    //     QPainter painter(&image);
    //     painter.setPen(QPen(MetricsWidgetStyle::axes_color, 1));
    //     painter.drawRect(0, 0, image.width() - 1, image.height() - 1);
    //     image.save("../misc/LettersEx/good_" + QString(((char)name + 'a')) + QString::number(letters_count[name]) + ".png");
    // }
    // return 0;

    // QImage image(140, 140, QImage::Format_RGBA8888);
    // {
    //     QPainter painter(&image);
    //     painter.setPen(QPen(MetricsWidgetStyle::axes_color, 1));
    //     painter.drawRect(0, 0, image.width() - 1, image.height() - 1);


    //     QFont font;
    //     font.setPointSize(100); // Set the font size to make the letter A larger
    //     painter.setFont(font);

    //     painter.setPen(QPen(Qt::red, 100));
    //     QRect rect(0, 0, image.width(), image.height());
    //     painter.drawText(rect, Qt::AlignCenter, "A");
    // }
    // image.save("../misc/LettersEx/bad_A_5.png");
    // return a.exec();

    for (int k = 0; k < 5; ++k) {
    unsigned name = data[k].second;
    data_vector letter(data[k].first);
    QImage image(28, 28, QImage::Format_RGBA8888);
    image.fill(Qt::transparent);
    for (unsigned i = 0; i < 28; ++i)
    {
        for (unsigned j = 0; j < 28; ++j)
        {
            image.setPixelColor(i, j, QColor(255, 0, 0, letter[i * 28 + j] * 255));
        }
    }

    for (int h = 0; h < 30; ++h) {
        image.setPixelColor(Random::Int(0, 8), Random::Int(0, 8), QColor(255, 0, 0, Random::Int(0, 100)));
    }

    image = image.scaled(28 * 5, 28 * 5);

    

    // QTransform transform;
    // transform.rotate(45);
    // image = image.transformed(transform);

    // image = image.scaled(28 * 5, 28 * 5);

    QPainter painter(&image);
    painter.setPen(QPen(MetricsWidgetStyle::axes_color, 1));
    painter.drawRect(0, 0, image.width() - 1, image.height() - 1);
    image.save("../misc/LettersEx/bad_" + QString(((char)name + 'a')) + QString::number(letters_count[name]) + "_6.png");
    }

    return 0;
}