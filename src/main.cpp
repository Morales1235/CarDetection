#include "opencv2/highgui.hpp"

#include <iostream>
#include <time.h>

#include "hogdetector.h"


int main( int argc, char** argv )
{
    const char* keys =
    {
        "{help h|     | show help message}"
        "{pd    |     | path of directory contains positive images}"
        "{nd    |     | path of directory contains negative images}"
        "{td    |     | path of image, video file or directory wiwth test images}"
        "{f     |false| indicates if the program will generate and use mirrored samples or not}"
        "{t     |false| test a trained detector}"
        "{sd    |false| save trained detector}"
        "{sr    |false| save result (image or video) of detection}"
        "{sw    |true| show results of edtection}"
        "{fn    |my_detector.yml| file name of trained SVM}"
    };

    cv::CommandLineParser parser( argc, argv, keys );

    if ( parser.has( "help" ) )
    {
        parser.printMessage();
        exit( 0 );
    }

    std::string pos_dir = parser.get< std::string >( "pd" );
    std::string neg_dir = parser.get< std::string >( "nd" );
    std::string test_dir = parser.get< std::string >( "td" );
    std::string obj_det_filename = parser.get< std::string >( "fn" );
    bool test_detector = parser.get< bool >( "t" );
    bool flip_samples = parser.get< bool >( "f" );
    bool save_detector = parser.get< bool >( "sd" );
    bool show_result = parser.get< bool >("sw");
    bool save_result = parser.get< bool >("sr");

    SVMParams params;
    params.degree = 3.0;
    params.ny = 0.5;
    params.P = 0.1;
    params.C = 0.01;
    params.termCriteria = cv::TermCriteria{cv::TermCriteria::MAX_ITER, 1000, 1e-4};
    params.kernel = cv::ml::SVM::LINEAR;
    params.type = cv::ml::SVM::EPS_SVR;
    HOGDetector my_detector{params};

    if ( test_detector )
    {
        my_detector.TestFromFile( obj_det_filename, test_dir, show_result, save_result );
        exit( 0 );
    }

    if( pos_dir.empty() || neg_dir.empty() )
    {
        parser.printMessage();
        std::cout << "Wrong number of parameters.\n\n"
             << "Example command line:\n" << argv[0] << " -dw=64 -dh=128 -pd=/INRIAPerson/96X160H96/Train/pos -nd=/INRIAPerson/neg -td=/INRIAPerson/Test/pos -fn=HOGpedestrian64x128.xml -d\n"
             << "\nExample command line for testing trained detector:\n" << argv[0] << " -t -fn=HOGpedestrian64x128.xml -td=/INRIAPerson/Test/pos";
        exit( 1 );
    }

    my_detector.Train(std::make_pair(pos_dir, neg_dir), flip_samples);
    my_detector.Test( test_dir, show_result, save_result );
    if( save_detector )
        my_detector.Save( obj_det_filename );
    cv::destroyAllWindows();

    return 0;
}
