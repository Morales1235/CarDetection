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
    cv::VideoCapture vid;
    vid.open(filename);
    cv::VideoWriter writer{"output.avi",
                           cv::VideoWriter::fourcc('M','J','P','G'),
                           vid.get(cv::CAP_PROP_FPS),
                           cv::Size{(int)vid.get(cv::CAP_PROP_FRAME_WIDTH),
                                       (int)vid.get(cv::CAP_PROP_FRAME_HEIGHT)}};
    while(vid.isOpened())
    {
        cv::Mat frame;
        vid >> frame;
        if(frame.data)
        {
            ProceedSearching(frame);
            writer.write(frame);
        }
        else break;
    }
    vid.release();
    writer.release();
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
            ProceedSearching(img);
        }
        cv::imshow("Detection", img);
        cv::waitKey(0);
    }
}


void RoadSearcher::ProceedSearching(cv::Mat &image)
{
    carDetector.Detect(image, false);
    pedestriantsDetector.Detect(image, false);
}
