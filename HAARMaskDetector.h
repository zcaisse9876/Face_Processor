#ifndef HAARMASKDETECTOR_H
#define HAARMASKDETECTOR_H

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

class HAARMaskDetector {
private:
    cv::CascadeClassifier mouthCascade;
    cv::CascadeClassifier noseCascade;
    HAARMaskDetector();
    bool prepareFace(cv::Mat &img, cv::Mat &dest, int x1, int y1, int x2, int y2);
    bool cropToJaw(cv::Mat &image, cv::Mat &dest, int x1, int y1, int x2, int y2);

public:
    HAARMaskDetector(const std::string &model, const std::string &model2);
    bool hasMask(cv::Mat image, int x1, int y1, int x2, int y2);
};

#endif // HAARMASKDETECTOR_H
