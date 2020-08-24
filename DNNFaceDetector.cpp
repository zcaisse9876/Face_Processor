#include "DNNFaceDetector.h"

// Privatize default constructor, can't be used
DNNFaceDetector::DNNFaceDetector() {}

cv::Mat DNNFaceDetector::prepareImage(cv::Mat &image)
{
  cv::Mat imageSm;
  // Resize image to 300x300, model trained on this size image
  cv::resize(image, imageSm, cv::Size(300, 300), 0, 0, cv::INTER_CUBIC);
  // https://pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
  return cv::dnn::blobFromImage(imageSm, 1.0, cv::Size(300, 300), cv::Scalar(104.0, 177.0, 123.0));
}

cv::Mat DNNFaceDetector::forwardPass(cv::Mat blob)
{
  // Set input data for nueral net
  this->net.setInput(blob, "data");
  // Run forward pass using input data and receive output from model
  return net.forward("detection_out");
}

DNNFaceDetector::DNNFaceDetector(std::string protoText, std::string caffeModel, float confidenceThresh)
{
  // Confidence on face detections must be greater than threshold to be reported
  this->confidenceThresh = confidenceThresh;
  // Load nueral net into memory
  this->net = cv::dnn::readNetFromCaffe(protoText, caffeModel);
}

std::vector<std::vector<float>> DNNFaceDetector::getBoundingBoxes(cv::Mat &image)
{
  std::vector<std::vector<float>> detectionInfo;
  cv::Mat blob = this->prepareImage(image);
  cv::Mat detection = this->forwardPass(blob);
  // This is some cut and paste code from OpenCV, will investigate later
  cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
  // There can be multiple faces in an image, iterate across nn output to find them all
  for (int i = 0; i < detectionMat.rows; i++)
  {
    float confidence = detectionMat.at<float>(i, 2);
    if (confidence >= this->confidenceThresh)
    {
      float x1 = detectionMat.at<float>(i, 3) * image.cols;
      float y1 = detectionMat.at<float>(i, 4) * image.rows;
      float x2 = detectionMat.at<float>(i, 5) * image.cols;
      float y2 = detectionMat.at<float>(i, 6) * image.rows;
      detectionInfo.push_back({confidence, x1, y1, x2, y2});
    }
  }
  return detectionInfo;
}

DNNLandmarkDetector::DNNLandmarkDetector() {}

DNNLandmarkDetector::DNNLandmarkDetector(std::string protoText, std::string caffeModel, std::string landmarkModel, float confidenceThresh)
    : DNNFaceDetector(protoText, caffeModel, confidenceThresh)
{
  // Initialize shape predictor;
  dlib::deserialize(landmarkModel) >> sp;
}

std::map<std::string, std::vector<std::vector<float>>> DNNLandmarkDetector::getFacialLandmarks(cv::Mat &image)
{
  auto ROI = this->getBoundingBoxes(image);
  std::vector<std::vector<float>> landmarkInfo;
  for (size_t i = 0; i < ROI.size(); ++i)
  {
    dlib::rectangle face(ROI[i][1], ROI[i][2], ROI[i][3], ROI[i][4]);
    dlib::full_object_detection shape = sp(dlib::cv_image<dlib::bgr_pixel>(image), face);
    std::vector<float> points;
    for (size_t i = 0; i < shape.num_parts(); ++i)
    {
      points.push_back(shape.part(i).x());
      points.push_back(shape.part(i).y());
    }
    landmarkInfo.push_back(points);
  }
  return {{"ROI", ROI}, {"LM", landmarkInfo}};
}
