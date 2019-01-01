#ifndef HOGDETECTOR_H
#define HOGDETECTOR_H

#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/objdetect/objdetect.hpp>

template<class T>
using PairOf = std::pair<T, T>;

struct SVMParams
{
    using SVM = cv::ml::SVM;

    double coef0{0.0};
    double degree{0.0};
    cv::TermCriteria termCriteria{cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 1000, 1e-3};
    double gamma{1.0};
    SVM::KernelTypes kernel{SVM::RBF};
    double ny{0.0};
    double P{0.0};
    double C{0.0};
    SVM::Types type{SVM::C_SVC};
};

class HOGDetector
{
public:
    HOGDetector(SVMParams params = SVMParams());
    void Train(PairOf<std::string>&& trainDataDirNames, bool flipSamples);
    void TestFromFile(std::string detectorFilename, std::string testDir, bool show = true, bool save = false);
    void Test(std::string testDir, bool show = true, bool save = false);
    void Save(std::string destFile);

private:
    auto GetDetector();
    void CumulateData(const std::vector< cv::Mat >& samples, cv::Mat& trainData);
    void loadImages(const  std::string& dirName, std::vector<cv::Mat>& images);
    auto loadImages(PairOf<std::string>&& dirNames);
    void computeHOGs(const std::vector<cv::Mat>& images, std::vector< cv::Mat>& gradients, bool useFlip);
    bool checkImgDimensions(PairOf<std::vector<cv::Mat>>&& images);
    void testVideo(std::string videoName, bool show = true, bool save = false);
    void testImages(std::string dirName, bool show = true, bool save = false);
    void Detect(cv::Mat& image, bool display = true, int delay = 0);

    cv::Ptr<cv::ml::SVM> mSVM;
    cv::HOGDescriptor mHOGd;
    bool mIsTrained{false};
};

#endif // HOGDETECTOR_H
