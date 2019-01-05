#include "hogdetector.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/objdetect.hpp"

#include <boost/range/combine.hpp>
#include <boost/foreach.hpp>
#include <boost/range/adaptor/indexed.hpp>

HOGDetector::HOGDetector(SVMParams params)
{
    mSVM = cv::ml::SVM::create();
    mSVM->setCoef0(params.coef0);
    mSVM->setDegree(params.degree);
    mSVM->setTermCriteria(params.termCriteria);
    mSVM->setGamma(params.gamma);
    mSVM->setKernel(params.kernel);
    mSVM->setNu(params.ny);
    mSVM->setP(params.P);
    mSVM->setC(params.C);
    mSVM->setType(params.type);
}

auto HOGDetector::loadImages(PairOf<std::string> &&dirNames)
{
    std::vector<cv::Mat> positives, negatives;
    auto positivesDir = dirNames.first;
    auto negativesDir = dirNames.second;
    std::clog << "Loading images...";
    loadImages(positivesDir, positives);
    loadImages(negativesDir, negatives);
    std::clog << "OK\n";
    return std::make_pair(positives, negatives);
}

auto HOGDetector::GetDetector()
{
    assert(mSVM);
    if(mSVM)
    {
        cv::Mat sv = mSVM->getSupportVectors();
        cv::Mat alpha, svidx;
        double rho = mSVM->getDecisionFunction(0, alpha, svidx);

        CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv.rows == 1);
        CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
                   (alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
        CV_Assert(sv.type() == CV_32F);

        std::vector<float> detector(sv.cols + 1);
        memcpy(&detector[0], sv.ptr(), sv.cols*sizeof(detector[0]));
        detector[sv.cols] = (float)-rho;
        return detector;
    }
    return std::vector<float>{};
}

void HOGDetector::Train(PairOf<std::string>&& trainDataDirNames, bool flipSamples)
{
    std::vector<cv::Mat> gradients;
    std::vector<int> labels;
    cv::Mat trainData;
    auto positivesDir = trainDataDirNames.first;
    auto negativesDir = trainDataDirNames.second;

    auto [positiveImages, negativeImages] = loadImages(std::make_pair(positivesDir, negativesDir));

    labels.assign(positiveImages.size() * (flipSamples ? 2 : 1), +1);
    labels.insert(labels.end(), negativeImages.size() * (flipSamples ? 2 : 1), -1);

    std::clog << "Calculating histograms of gradients...";
    computeHOGs(positiveImages, gradients, flipSamples);
    computeHOGs(negativeImages, gradients, flipSamples);
    assert(labels.size() == gradients.size());
    std::clog << "OK. (samples count : " << gradients.size() << ")\n";

    CumulateData(gradients, trainData);

    std::clog << "Training SVM...";
    mSVM->train(trainData, cv::ml::ROW_SAMPLE, labels);
    mHOGd.winSize = positiveImages[0].size();
    mHOGd.setSVMDetector(GetDetector());
    mIsTrained = true;
}

void HOGDetector::CumulateData(const std::vector<cv::Mat>& samples, cv::Mat& trainData)
{
    if(samples.empty())
    {
        std::cerr << "No data to cumulate\n";
        return;
    }
    const int rows = (int)samples.size();
    const int cols = (int)std::max(samples[0].cols, samples[0].rows);
    cv::Mat tmp(1, cols, CV_32FC1);
    trainData = cv::Mat(rows, cols, CV_32FC1);

    for(const auto& sample: samples | boost::adaptors::indexed(0))
    {
        CV_Assert(sample.value().cols == 1 || sample.value().rows == 1);

        if(sample.value().cols == 1)
        {
            transpose(sample.value(), tmp);
            tmp.copyTo(trainData.row(sample.index()));
        }
        else if(sample.value().rows == 1)
        {
            sample.value().copyTo(trainData.row(sample.index()));
        }
    }
}

void HOGDetector::loadImages(const std::string & dirName, std::vector<cv::Mat> & images)
{
    std::vector<std::string> files;
    cv::glob(dirName, files, true);
    if(files.empty())
    {
        std::cerr << "No images to load\n";
        return;
    }

    cv::Mat img = cv::imread(files.front());
    auto imgSize = img.size();
    for(const auto& file: files)
    {
        img = cv::imread(file);
        if (img.empty() || img.size() != imgSize)
            continue;
        images.push_back(img);
    }
}

void HOGDetector::computeHOGs(const std::vector<cv::Mat> & images, std::vector<cv::Mat> & gradients, bool useFlip)
{
    if(images.empty())
    {
        std::cerr << "No images for compute hogs\n";
        return;
    }
    auto wsize = images.front().size();
    cv::HOGDescriptor hog;
    hog.winSize = wsize;
    cv::Mat gray;
    std::vector<float> descriptors;

    for(const auto& image: images)
    {
        if (image.cols>= wsize.width && image.rows >= wsize.height)
        {
            cvtColor(image, gray, cv::COLOR_BGR2GRAY);
            hog.compute(gray, descriptors, cv::Size(8, 8), cv::Size(0, 0));
            gradients.push_back(cv::Mat(descriptors).clone());
            if (useFlip)
            {
                cv::flip(gray, gray, 1);
                hog.compute(gray, descriptors, cv::Size(8, 8), cv::Size(0, 0));
                gradients.push_back(cv::Mat(descriptors).clone());
            }
        }
    }
}

void HOGDetector::testVideo(std::string videoName, bool show, bool save)
{
    if(!mIsTrained)
        return;
    cv::VideoCapture vid;
    vid.open(videoName);
    auto fps = vid.get(cv::CAP_PROP_FPS);
    cv::VideoWriter writer{"output.avi",
                cv::VideoWriter::fourcc('M','J','P','G'),
                fps,
                cv::Size{(int)vid.get(cv::CAP_PROP_FRAME_WIDTH), (int)vid.get(cv::CAP_PROP_FRAME_HEIGHT)}};
    int cnt = 0;
    while(vid.isOpened())
    {
        cv::Mat frame;
        vid >> frame;
        std::cout << "Wroking on frame: " << cnt++ << std::endl;
        if(frame.data)
            Detect(frame, show, 1);
        else
            break;
        if(save)
            writer.write(frame);
    }
    vid.release();
    writer.release();
}

void HOGDetector::testImages(std::string dirName, bool show, bool save)
{
    if(!mIsTrained)
        return;
    std::vector<std::string> files;
    cv::glob(dirName, files, true);
    for(const auto& file: files)
    {
        cv::Mat img = cv::imread(file);
        if(img.data)
            Detect(img, show);
        else
            break;
        if(save)
            cv::imwrite(dirName + "detected_", img);
    }
}

void HOGDetector::Detect(cv::Mat &image, bool display, int delay)
{
    if(!image.data)
        return;
    std::vector<cv::Rect> detections;
    std::vector<double> weights;
    cv::Rect det;
    double weight;
    int shift = image.rows/2;
    auto roi{image.rowRange(shift, image.rows)};
    if(display)
        cv::namedWindow("Testing", cv::WINDOW_NORMAL);

    mHOGd.detectMultiScale(roi, detections, weights);
    BOOST_FOREACH(boost::tie(det, weight), boost::combine(detections, weights))
        if(weight > 0.8)
        {
            det.y += shift;
            cv::rectangle(image, det, {255, 0, 0});
        }
    if(display)
    {
        cv::imshow("Testing", image);
        cv::waitKey(delay);
    }
}

void HOGDetector::Test(std::string testDir, bool show, bool save)
{
    if(!mIsTrained)
    {
        std::cerr << "Cannot test on non trained classifier\n";
        return;
    }
    std::cout << "Testing trained detector...\n";
    if(testDir.compare(testDir.length()-4, 4, ".mp4") == 0)
        testVideo(testDir, show, save);
    else
        testImages(testDir, show, save);
}

void HOGDetector::Save(std::string destFile)
{
    mHOGd.save(destFile);
}

void HOGDetector::TestFromFile(std::string detectorFilename, std::string testDir, bool show, bool save)
{
    mIsTrained = mHOGd.load(detectorFilename);
    Test(testDir, show, save);

}
