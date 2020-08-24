#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "QIntValidator"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent),
      ui(new Ui::MainWindow),
      roiColor(102, 102, 255),
      detector(this->protoText, this->caffeModel, 0.65),
      lmdetector(this->protoText, this->caffeModel, this->facemarkModel, 0.65),
      mouthCascade(this->mouthModel),
      mouthdetector(this->mouthModel, this->noseModel)
{
    ui->setupUi(this);
    ui->gfxVid->setScene(new QGraphicsScene(this));
    this->ui->inCam->setValidator( new QIntValidator(0, 100, this) );
    this->timer = new QTimer(this->ui->lcdClock);
    this->sessionTimer = new QTimer(this);
    this->sessionTimer->setInterval(this->sessionTime * 1000);
    this->sessionTimer->setSingleShot(true);
    this->connect(sessionTimer, &QTimer::timeout, this, &MainWindow::endSession);
    //this->ui->lcdClock->setDigitCount(8);
    this->ui->lcdClock->connect(timer, &QTimer::timeout, this, &MainWindow::showTime);
    this->timer->start(1000);
    this->showTime();
}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_btnPlay_clicked()
{
    if (this->Playing) {
        this->stopCamera();
        return;
    }
    this->startCamera();
    this->captureVideo();
}

void MainWindow::stopCamera(bool supressInfo) {
    if (!supressInfo)
        this->showInstruction("Closing Camera");
    this->ui->btnPlay->setEnabled(false);
    qApp->processEvents(); qApp->processEvents();
    this->Playing = false;
    ui->gfxVid->scene()->removeItem(&pixmap);
    ui->btnPlay->setText("Record");
    if (!supressInfo)
        this->showInstruction("Click record to begin");
    this->ui->btnPlay->setEnabled(true);
}

void MainWindow::startCamera() {
    this->showInstruction("Opening Camera");
    this->ui->btnPlay->setEnabled(false);
    qApp->processEvents(); qApp->processEvents();
    this->capture.open(this->iCam);
    this->Playing = true;
    ui->gfxVid->scene()->addItem(&pixmap);
    ui->btnPlay->setText("Stop");
    this->ui->btnPlay->setEnabled(true);
}

void MainWindow::captureVideo() {
    int newDetection = 0;
    const int frameSkip = 1;
    std::vector<std::vector<float>> roi;
    cv::Scalar roiValid(0, 165, 255);
    cv::Scalar roiInvalid(102, 102, 255);
    cv::Mat frame;
    cv::Mat displayFrame;
    int framesPassed = 0;
    const int maskAverageFrames = 10;
    int maskDetected = 0;

    if (!this->capture.isOpened()) {
        this->showInstruction("Cant open camera by index", 255, 102, 102);
        this->stopCamera(true);
        return;
    }

    while (this->Playing) {
        capture >> frame;
        if (frame.empty()) {
            this->stopCamera();
            break;
        }
        displayFrame = frame.clone();
//        auto landmarks = lmdetector.getFacialLandmarks(frame);
//        drawLandmarks(frame, landmarks);
        const int boundx1 = frame.cols * .4, boundx2 = frame.cols * .6, boundy1 = frame.rows * .35, boundy2 = frame.rows * .65;
        if (newDetection == 0) {
            roi = detector.getBoundingBoxes(frame);
            if (roi.size())
                this->startSession();
            if (this->canAuthenticate) {
               framesPassed++;
               if (roi.size() > 0 && mouthdetector.hasMask(frame, roi[0][1], roi[0][2], roi[0][3], roi[0][4])) {
                   maskDetected++;
               }
               if (framesPassed >= maskAverageFrames) {
                   this->wearingMask = maskDetected > framesPassed * 0.6;
                   framesPassed = 0;
                   maskDetected = 0;
               }
            }
        }
        this->canAuthenticate = this->validDetection(roi, boundx1, boundx2, boundy1, boundy2);
        this->drawGuideline(displayFrame, boundx1, boundx2, boundy1, boundy2,  this->canAuthenticate ? (this->wearingMask ? cv::Scalar(0, 255, 127) : roiValid) : roiInvalid);
        this->drawImage(displayFrame, roi);
        qApp->processEvents();
        newDetection = (newDetection + 1) % frameSkip;
    }
}

void MainWindow::renderFrame(cv::Mat &image) {
    QImage qimg(image.data, image.cols, image.rows, image.step, QImage::Format_RGB888);
    this->pixmap.setPixmap( QPixmap::fromImage(qimg.rgbSwapped()) );
    ui->gfxVid->fitInView(&pixmap, Qt::KeepAspectRatioByExpanding);
}

bool MainWindow::validDetection(std::vector<std::vector<float>> &faces, int bx1, int bx2, int by1, int by2) {
    if (faces.size() > 1) {
        this->showInstruction("Only one person at a time", 255, 102, 102);
        return false;
    }

    if (!faces.size()) {
        this->showInstruction("No faces detected");
        return false;
    }

    const int &x1 = static_cast<int>(faces[0][1]);
    const int &y1 = static_cast<int>(faces[0][2]);
    const int &x2 = static_cast<int>(faces[0][3]);
    const int &y2 = static_cast<int>(faces[0][4]);

    // If face is smaller than box, return false
    if (x2 - x1 < bx2 - bx1 || y2 - y1 < by2 - by1) {
        this->showInstruction("Move closer to the camera", 255, 102, 102);
        return false;
    }

    // If face is too much larger than box, return false
    if (x2 - x1 > (bx2 - bx1) * 1.3 && y2 - y1 > (by2 - by1) * 1.3) {
        this->showInstruction("Move back from the camera", 255, 102, 102);
        return false;
    }

    // if face is not positioned in center of screen, return false
    if (x2 < bx2 || x1 > bx1 || y2 < by2 || y1 > by1) {
        this->showInstruction("Center yourself using the guidelines", 255, 165, 0);
        return false;
    }

    this->showInstruction("Hold that pose", 127, 255, 0);
    return true;
}

void MainWindow::drawImage(cv::Mat &image, const std::vector<std::vector<float>> &faces)
{
    for (size_t i = 0; i < faces.size(); ++i) {
        const float confidence = faces[i][0];
        const int x1 = static_cast<int>(faces[i][1]);
        const int y1 = static_cast<int>(faces[i][2]);
        const int x2 = static_cast<int>(faces[i][3]);
        const int y2 = static_cast<int>(faces[i][4]);
        const int textY = y1 - 10 > 10 ? y1 - 10 : y1 + 10;
        this->drawBorder(image, x1, y1, x2, y2);
//        cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(158, 207, 255), 2, 4);
        if (this->canAuthenticate)
            cv::putText(image, this->wearingMask ? "Wearing Mask" : "No Mask", cv::Point(x1, textY), cv::FONT_HERSHEY_SIMPLEX, 0.45, this->wearingMask ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 1);
    }
    this->renderFrame(image);
}

void MainWindow::drawLandmarks(cv::Mat &image, std::map<std::string, std::vector<std::vector<float>>> landmarks) {
    std::vector<std::vector<float>> lm = landmarks["LM"];
    std::vector<std::vector<float>> faces = landmarks["ROI"];

    for (size_t i = 0; i < faces.size(); ++i)
    {
        const float confidence = faces[i][0];
        const int x1 = static_cast<int>(faces[i][1]);
        const int y1 = static_cast<int>(faces[i][2]);
        const int x2 = static_cast<int>(faces[i][3]);
        const int y2 = static_cast<int>(faces[i][4]);

        for (size_t j = 0; j < lm[i].size() - 1; j+=2)
        {
            cv::circle(image, cv::Point(lm[i][j], lm[i][j+1]), 3, cv::Scalar(255, 0, 0), cv::FILLED);
        }
        this->drawBorder(image, x1, y1, x2, y2);
    }
    this->renderFrame(image);
}

void MainWindow::drawGuideline(cv::Mat &image, int x1, int x2, int y1, int y2, const cv::Scalar &color) {
    cv::LineIterator top(image, cv::Point(x1, y1), cv::Point(x2, y1), 8);
    cv::LineIterator top2(image, cv::Point(x1, y1 - 1), cv::Point(x2, y1 - 1), 8);
    cv::LineIterator left(image, cv::Point(x1, y1), cv::Point(x1, y2), 8);
    cv::LineIterator left2(image, cv::Point(x1 - 1, y1), cv::Point(x1 - 1, y2), 8);
    cv::LineIterator right(image, cv::Point(x2, y1), cv::Point(x2, y2), 8);
    cv::LineIterator right2(image, cv::Point(x2 + 1, y1), cv::Point(x2 + 1, y2), 8);
    cv::LineIterator bottom(image, cv::Point(x1, y2), cv::Point(x2, y2), 8);
    cv::LineIterator bottom2(image, cv::Point(x1, y2 + 1), cv::Point(x2, y2 + 1), 8);

    // Use a prime number here or you will have uneven dashes
    int dashWidth = 7;
    bool draw = true;
    for (int i = 0; i < top.count; ++i, ++top, ++bottom, ++top2, ++bottom2)
    {
        draw = i % dashWidth == 0 ? !draw : draw;
        if (draw)
        {
            // Yellow
            (*top)[0] = color.val[0]; // Blue
            (*top)[1] = color.val[1]; // Green
            (*top)[2] = color.val[2] ; // Red
            (*top2)[0] = color.val[0]; // Blue
            (*top2)[1] = color.val[1]; // Green
            (*top2)[2] = color.val[2] ; // Red
            (*bottom)[0] = color.val[0]; // Blue
            (*bottom)[1] = color.val[1] ; // Green
            (*bottom)[2] = color.val[2]; // Red
            (*bottom2)[0] = color.val[0]; // Blue
            (*bottom2)[1] = color.val[1] ; // Green
            (*bottom2)[2] = color.val[2]; // Red
        }
    }

    draw = true;
    for (int i = 0; i < left.count; ++i, ++left, ++left2, ++right, ++right2)
    {
        draw = i % dashWidth == 0 ? !draw : draw;
        if (draw)
        if (draw)
        {
            // Yellow
            (*right)[0] = color.val[0]; // Blue
            (*right)[1] = color.val[1]; // Green
            (*right)[2] = color.val[2]; // Red
            (*right2)[0] = color.val[0]; // Blue
            (*right2)[1] = color.val[1]; // Green
            (*right2)[2] = color.val[2]; // Red
            (*left)[0] = color.val[0]; // Blue
            (*left)[1] = color.val[1]; // Green
            (*left)[2] = color.val[2]; // Red
            (*left2)[0] = color.val[0]; // Blue
            (*left2)[1] = color.val[1]; // Green
            (*left2)[2] = color.val[2]; // Red
        }
    }
}

void MainWindow::drawBorder(cv::Mat &image, int x1, int y1, int x2, int y2) {
    const int sDistance = std::min(x2 - x1, y2 - y1);
    const int lineLength = sDistance / 4;

    cv::line(image, cv::Point(x1, y1), cv::Point(x1, y1 + lineLength), this->roiColor, 4);
    cv::line(image, cv::Point(x1, y1), cv::Point(x1 + lineLength, y1), this->roiColor, 4);

    cv::line(image, cv::Point(x1, y2), cv::Point(x1, y2 - lineLength), this->roiColor, 4);
    cv::line(image, cv::Point(x1, y2), cv::Point(x1 + lineLength, y2), this->roiColor, 4);

    cv::line(image, cv::Point(x2, y1), cv::Point(x2 - lineLength, y1), this->roiColor, 4);
    cv::line(image, cv::Point(x2, y1), cv::Point(x2, y1 + lineLength), this->roiColor, 4);

    cv::line(image, cv::Point(x2, y2), cv::Point(x2, y2 - lineLength), this->roiColor, 4);
    cv::line(image, cv::Point(x2, y2), cv::Point(x2 - lineLength, y2), this->roiColor, 4);
}

void MainWindow::closeEvent(QCloseEvent* event)
{
    if (this->capture.isOpened())
        this->stopCamera();
    event->accept();
}

void MainWindow::showTime()
//! [1] //! [2]
{
    QDateTime time = QDateTime::currentDateTime();
    QString dtxt = time.toString("MM-dd-yyyy");
    QString text = time.toString("hh:mm");
    const int hour = time.time().hour();
    const int second = time.time().second();
    QString meridiem = hour >= 12 ? "PM" : "AM";
    if (hour == 24 || hour == 0) {
        text[0] = '1';
        text[1] = '2';
    } else if (hour > 12) {
        int stdTime = hour - 12;
        std::string stdTimeStr = std::to_string(stdTime);
        if (stdTime >= 10) {
            text[0] = stdTimeStr[0];
            text[1] = stdTimeStr[1];
        } else {
            text[0] = '0';
            text[1] = stdTimeStr[0];
        }
    }
    if ((second % 2) == 0) {
        text[2] = ' ';
    }
    this->ui->lblCam->setText(dtxt);
    this->ui->lcdClock->setText(text + meridiem);
}

void MainWindow::showInstruction(QString info, int r, int g, int b) {
    this->ui->lblInstruction->setText(info.toUpper());
    this->roiColor = cv::Scalar(b, g, r);
    this->ui->lblInstruction->setStyleSheet("QLabel {color : rgb(" + QString::number(r) + "," + QString::number(g)+ "," + QString::number(b) + ");}");
}

void MainWindow::on_inCam_textChanged(const QString &arg1)
{
    std::cout << arg1.toStdString() << std::endl;
    this->iCam = arg1.toInt();
}

bool MainWindow::detectMouth(cv::Mat &image, std::vector<std::vector<float>> &faces) {
    if (!faces.size())
        return false;
    try {
        const int x = faces[0][1];
        const int y = faces[0][2] + ((faces[0][4] - faces[0][2]) * 0.45);
        const int width = faces[0][3] - faces[0][1];
        const int height = (faces[0][4] - faces[0][2]) * 0.55;
        if (x < 0 || x >= image.cols || y < 0 || y > image.rows)
            return false;

        if (x + width > image.cols || y + height > image.rows)
            return false;
        cv::Rect roi;
        roi.x = x;
        roi.y = y;
        roi.width = width;
        roi.height = height;
        cv::Mat gray = image(roi);
        cv::cvtColor(gray, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Rect> mouth_rects;
        this->mouthCascade.detectMultiScale(gray, mouth_rects, 1.1, 2, 0, cv::Size());
        return mouth_rects.size() == 0;
    } catch (cv::Exception e) {
        return false;
    }
    return false;
}

void MainWindow::startSession() {
    if (this->sessionTimer->isActive() && this->sessionTimer->remainingTime() < 1000) {
        this->sessionTimer->stop();
        this->sessionTimer->start();
        std::cout << "Session Extended: " << this->sessionID << std::endl;
    } else if (this->sessionTimer->isActive()) {
        return;
    } else {
        this->sessionID = this->getUuid();
        std::cout << "Session Started: " << this->sessionID << std::endl;
        this->sessionTimer->start();
    }
}

void MainWindow::endSession() {
    std::cout << "Session Ended: " << this->sessionID << std::endl;
    this->Authenticated = false;
    this->sessionID = "";

}

std::string MainWindow::getUuid() {
    static std::random_device dev;
    static std::mt19937 rng(dev());

    std::uniform_int_distribution<int> dist(0, 15);

    const char *v = "0123456789abcdef";
    const bool dash[] = { 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0 };

    std::string res;
    for (int i = 0; i < 16; i++) {
        if (dash[i]) res += "-";
        res += v[dist(rng)];
        res += v[dist(rng)];
    }
    return res;
}
