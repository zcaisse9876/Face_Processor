#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QDebug>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QImage>
#include <QPixmap>
#include <QCloseEvent>
#include <QMessageBox>
#include <QTimer>
#include <QTime>

#include <iostream>
#include <sstream>
#include "DNNFaceDetector.h"
#include "HAARMaskDetector.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <random>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    void closeEvent(QCloseEvent* event);
    ~MainWindow();

private slots:

    void on_btnPlay_clicked();

    void on_inCam_textChanged(const QString &arg1);

private:
    Ui::MainWindow *ui;
    cv::VideoCapture capture;
    int iCam = 0;
    QGraphicsPixmapItem pixmap;
    bool Playing = false;
    cv::Scalar roiColor;
    QTimer *sessionTimer;
    QTimer *timer;
    const std::string caffeModel = "/home/parallels/Projects/C++/projects/face_detector/assets/res10_300x300_ssd_iter_140000.caffemodel";
    const std::string protoText = "/home/parallels/Projects/C++/projects/face_detector/assets/deploy.prototxt.txt";
    const std::string facemarkModel = "/home/parallels/Projects/C++/projects/face_detector/assets/shape_predictor_68_face_landmarks.dat";
    const std::string mouthModel = "/home/parallels/Projects/C++/projects/face_detector/assets/haarcascade_mcs_mouth.xml";
    const std::string noseModel = "/home/parallels/Projects/C++/projects/face_detector/assets/haarcascade_mcs_nose.xml";
    void startSession();
    void endSession();
    const int sessionTime = 7;
    std::string sessionID = "";
    std::string getUuid();
    DNNFaceDetector detector;
    DNNLandmarkDetector lmdetector;
    cv::CascadeClassifier mouthCascade;
    HAARMaskDetector mouthdetector;
    void captureVideo();
    void drawImage(cv::Mat &image, const std::vector<std::vector<float>> &faces);
    void drawLandmarks(cv::Mat &image, std::map<std::string, std::vector<std::vector<float>>> landmarks);
    bool validDetection(std::vector<std::vector<float>> &faces, int bx1, int bx2, int by1, int by2);
    void drawGuideline(cv::Mat &image, int x1, int x2, int y1, int y2, const cv::Scalar &color);
    void renderFrame(cv::Mat &image);
    void drawBorder(cv::Mat &image, int x1, int y1, int x2, int y2);
    void stopCamera(bool supressInfo = false);
    void startCamera();
    void showTime();
    void showInstruction(QString info, int r = 255, int g = 255, int b = 255);
    bool detectMouth(cv::Mat &image, std::vector<std::vector<float>> &faces);
    bool wearingMask = false;
    bool canAuthenticate = false;
    bool Authenticated = false;
};
#endif // MAINWINDOW_H
