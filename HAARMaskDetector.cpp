#include "HAARMaskDetector.h"
#include <iostream>

HAARMaskDetector::HAARMaskDetector () {}

HAARMaskDetector::HAARMaskDetector(const std::string &model, const std::string &model2): mouthCascade(model), noseCascade(model2) {}

bool HAARMaskDetector::prepareFace(cv::Mat &image, cv::Mat &dest, int x1, int y1, int x2, int y2) {
    if (!cropToJaw(image, dest, x1, y1, x2, y2))
        return false;

    cv::cvtColor(dest, dest, cv::COLOR_BGR2GRAY);
    return true;
}

bool HAARMaskDetector::cropToJaw(cv::Mat &image, cv::Mat &dest, int x1, int y1, int x2, int y2) {
    const int x = x1;
    const int y = y1 + ((y2 - y1) * 0.45);
    const int width = x2 - x1;
    const int height = (y2 - y1) * 0.55;
    if (x < 0 || x >= image.cols || y < 0 || y > image.rows)
        return false;

    if (x + width > image.cols || y + height > image.rows)
        return false;

    cv::Rect roi;
    roi.x = x;
    roi.y = y;
    roi.width = width;
    roi.height = height;
    dest = image(roi);
    return true;
}

bool HAARMaskDetector::hasMask(cv::Mat image, int x1, int y1, int x2, int y2) {
    try {
        cv::Mat processed;
        if (!this->prepareFace(image, processed, x1, y1, x2, y2))
            return false;
        std::vector<cv::Rect> mouth_rects;
        std::vector<cv::Rect> nose_rects;
        this->noseCascade.detectMultiScale(processed, nose_rects, 1.1, 3, 0, cv::Size());
        if (nose_rects.size() > 0)
            return false;
        this->mouthCascade.detectMultiScale(processed, mouth_rects, 2, 3, 0, cv::Size());
        return mouth_rects.size() == 0 && nose_rects.size() == 0;
    } catch (cv::Exception e) {
        return false;
    }
    return false;
}
