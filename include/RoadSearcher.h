#pragma once

#include <unordered_map>

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

    std::unordered_map<std::string, HOGDetector> mDetectors;
};
