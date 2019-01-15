#include "RoadSearcher.h"
#include "hogdetector.h"

#include <opencv2/highgui.hpp>

RoadSearcher::RoadSearcher(std::string carDetectorFile, std::optional<std::string> pedestriantsDetFile)
{
    if(!carDetector.Load(carDetectorFile))
        throw(std::string{"Cannot load "} + carDetectorFile);
    if(pedestriantsDetFile)
    {
        if(!pedestriantsDetector.Load(*pedestriantsDetFile))
            throw(std::string{"Cannot load "} + *pedestriantsDetFile);
    }
    else
    {
        pedestriantsDetector.SetDefaultPeopleDetector();
    }

}

void RoadSearcher::SearchVideo(std::string filename)
{
    std::cerr << "Not implemented yet!\n";
}

void RoadSearcher::SearchImages(std::string filepath)
{
    std::vector<std::string> files;
    cv::glob(filepath, files, true);

    cv::namedWindow("Detection", cv::WINDOW_NORMAL);
    for(const auto& file: files)
    {
        auto img = cv::imread(file);
        if(img.data)
        {
            carDetector.Detect(img, false);
            pedestriantsDetector.Detect(img, false);
        }
        cv::imshow("Detection", img);
        cv::waitKey(0);
    }
}


void RoadSearcher::ProceedSearching(cv::Mat &image)
{
    carDetector.Detect(image, false);
    pedestriantsDetector.Detect(image, false);

    cv::imshow("Detection", image);
}
