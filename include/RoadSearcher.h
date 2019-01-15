#pragma once

#include <opencv2/core/mat.hpp>

#include "hogdetector.h"

template<class T>
using PairOf = std::pair<T, T>;

class RoadSearcher
{
public:
    RoadSearcher(std::string carDetectorFile, std::optional<std::string> pedestriantsDetFile = std::nullopt);

    void SearchVideo(std::string filename);
    void SearchImages(std::string filepath);

private:
    void ProceedSearching(cv::Mat& image);

    HOGDetector carDetector;
    HOGDetector pedestriantsDetector;
};
