#ifndef DNNFACEDETECTOR_H
#define DNNFACEDETECTOR_H
#include <opencv2/dnn.hpp>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/image_transforms.h>
#include <dlib/opencv/cv_image.h>
#include <opencv2/imgproc.hpp>
#include <map>
#include <vector>

class DNNFaceDetector
{
protected:
  cv::dnn::Net net;
  float confidenceThresh;
  DNNFaceDetector();

private:
  cv::Mat prepareImage(cv::Mat &image);
  cv::Mat forwardPass(cv::Mat blob);

public:
  DNNFaceDetector(std::string protoText, std::string caffeModel, float confidenceThresh = 0.5);
  std::vector<std::vector<float>> getBoundingBoxes(cv::Mat &image);
};

class DNNLandmarkDetector : protected DNNFaceDetector
{
private:
  dlib::shape_predictor sp;
  DNNLandmarkDetector();

public:
  DNNLandmarkDetector(std::string protoText, std::string caffeModel, std::string landmarkModel, float confidenceThresh = 0.5);
  std::map<std::string, std::vector<std::vector<float>>> getFacialLandmarks(cv::Mat &image);
};
#endif // DNNFACEDETECTOR_H
