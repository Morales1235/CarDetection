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

constexpr auto COLOR_START_LINE = __LINE__;
enum class Color
{
    white = 0,
    blue,
    red
};
constexpr auto COLOR_COUNT = __LINE__ - COLOR_START_LINE - 4;

class HOGDetector
{
public:
    HOGDetector(SVMParams params = SVMParams());
    void Train(PairOf<std::string>&& trainDataDirNames, bool flipSamples);
    void TestSavedDetector(std::string detectorFilename, std::string testDir, bool show = true, bool save = false);
    void Test(std::string testDir, bool show = true, bool save = false);
    void Save(std::string destFile);
    bool Load(std::string filepath);
    void SetDefaultPeopleDetector();
    void Detect(cv::Mat& image, bool display = true);

private:
    auto GetDetector();
    void CumulateData(const std::vector< cv::Mat >& samples, cv::Mat& trainData);
    void loadImages(const  std::string& dirName, std::vector<cv::Mat>& images);
    auto loadImages(PairOf<std::string>&& dirNames);
    void computeHOGs(const std::vector<cv::Mat>& images, std::vector< cv::Mat>& gradients, bool useFlip);
    bool checkImgDimensions(PairOf<std::vector<cv::Mat>>&& images);
    void testVideo(std::string videoName, bool show = true, bool save = false);
    void testImages(std::string dirName, bool show = true, bool save = false);

    cv::Ptr<cv::ml::SVM> mSVM;
    cv::HOGDescriptor mHOGd;
    bool mIsTrained{false};
    static int currentColor;
    int mMyColor;
    static const std::array<cv::Scalar, COLOR_COUNT> colors;
};

#endif // HOGDETECTOR_H
