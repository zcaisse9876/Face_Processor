#include "mainwindow.h"

#include <QApplication>

void captureVideo();
void drawImage(cv::Mat &image, const std::vector<std::vector<float>> &faces);
bool detect = true;

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}

