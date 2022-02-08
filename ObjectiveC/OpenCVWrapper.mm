//
//  OpenCVWrapper.m
//  Pods-StreamMotionDetection_Example
//
//  Created by Дмитрий Семенюк on 8.02.22.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import <OpenCVWrapper.h>

//#pragma clang diagnostic push
//#pragma clang diagnostic ignored "-Wdocumentation"

#include "opencv.hpp"
#include "imgproc.hpp"
#include "videoio.hpp"
#include "highgui.hpp"
#include "video.hpp"
#include "segm.hpp"

//#pragma clang diagnostic pop

#include <stdio.h>
#include <iostream>
#include <sstream>



using namespace cv;
using namespace std;

cv::Mat prevImage;
//cv::Mat imgLines;
int iLastX = -1;
int iLastY = -1;

cv::Ptr<BackgroundSubtractorKNN> bgrSubstract = createBackgroundSubtractorKNN(1000, 300, false);
int frameCount = 0;
int numberCountFPS = 30;
time_t start, end;
int fps = 0;
bool isFPS = false;

/// Converts an UIImage to Mat.
/// Orientation of UIImage will be lost.

//int main(int argc, char** argv) {
//    cout << "Start work" << endl;
//}

static void imageToMat(UIImage *image, cv::Mat &mat) {
    assert(image.size.width > 0 && image.size.height > 0);
    assert(image.CGImage != nil || image.CIImage != nil);

    // Create a pixel buffer.
    NSInteger width = image.size.width;
    NSInteger height = image.size.height;
    cv::Mat mat8uc4 = cv::Mat((int)height, (int)width, CV_8UC4);

    // Draw all pixels to the buffer.
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    if (image.CGImage) {
        // Render with using Core Graphics.
        CGContextRef contextRef = CGBitmapContextCreate(mat8uc4.data, mat8uc4.cols, mat8uc4.rows, 8, mat8uc4.step, colorSpace, kCGImageAlphaPremultipliedLast | kCGBitmapByteOrderDefault);
        CGContextDrawImage(contextRef, CGRectMake(0, 0, width, height), image.CGImage);
        CGContextRelease(contextRef);
    } else {
        // Render with using Core Image.
        static CIContext* context = nil; // I do not like this declaration contains 'static'. But it is for performance.
        if (!context) {
            context = [CIContext contextWithOptions:@{ kCIContextUseSoftwareRenderer: @NO }];
        }
        CGRect bounds = CGRectMake(0, 0, width, height);
        [context render:image.CIImage toBitmap:mat8uc4.data rowBytes:mat8uc4.step bounds:bounds format:kCIFormatRGBA8 colorSpace:colorSpace];
    }
    CGColorSpaceRelease(colorSpace);

    // Adjust byte order of pixel.
    cv::Mat mat8uc3 = cv::Mat((int)width, (int)height, CV_8UC3);
    cv::cvtColor(mat8uc4, mat8uc3, cv::COLOR_RGBA2BGR);

    mat = mat8uc3;
}

/// Converts a Mat to UIImage.
static UIImage *matToImage(cv::Mat &mat) {

    // Create a pixel buffer.
    assert(mat.elemSize() == 1 || mat.elemSize() == 3);
    cv::Mat matrgb;
    if (mat.elemSize() == 1) {
        cv::cvtColor(mat, matrgb, cv::COLOR_GRAY2RGB);
    } else if (mat.elemSize() == 3) {
        cv::cvtColor(mat, matrgb, cv::COLOR_BGR2RGB);
    }

    // Change a image format.
    NSData *data = [NSData dataWithBytes:matrgb.data length:(matrgb.elemSize() * matrgb.total())];
    CGColorSpaceRef colorSpace;
    if (matrgb.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    CGImageRef imageRef = CGImageCreate(matrgb.cols, matrgb.rows, 8, 8 * matrgb.elemSize(), matrgb.step.p[0], colorSpace, kCGImageAlphaNone|kCGBitmapByteOrderDefault, provider, NULL, false, kCGRenderingIntentDefault);
    UIImage *image = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);

    return image;
}

static void histogram(UIImage *image, cv::Mat &mat) {
    Mat src;
    imageToMat(image, src);
    vector<Mat> bgr_planes;
    split( src, bgr_planes );
    int histSize = 256;
    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange[] = { range };
    bool uniform = true, accumulate = false;
    Mat b_hist, g_hist, r_hist;
    calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, histRange, uniform, accumulate );
    calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, histRange, uniform, accumulate );
    calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, histRange, uniform, accumulate );
    int hist_w = 1024, hist_h = 768;
    int bin_w = cvRound( (double) hist_w/histSize );
    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    for( int i = 1; i < histSize; i++ )
    {
        line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ),
             cv::Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
             Scalar( 255, 0, 0), 2, 8, 0  );
        line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ),
             cv::Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
             Scalar( 0, 255, 0), 2, 8, 0  );
        line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ),
             cv::Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
             Scalar( 0, 0, 255), 2, 8, 0  );
    }
    mat = histImage;
}

@implementation OpenCVWrapper

+ (UIImage *)backgroundSubstract:(nonnull UIImage *)image {
    frameCount = frameCount + 1;
    Mat originFrame;
    imageToMat(image, originFrame);

    Mat maskLines = cv::Mat(originFrame.size(), CV_8UC3);
    Mat grayLines = cv::Mat(originFrame.size(), CV_8UC3);

    Mat grayFrame;
    Mat gausFrame;

    cv::cvtColor(originFrame, grayFrame, cv::COLOR_BGR2GRAY);

    Mat substractMask;
    bgrSubstract->apply(grayFrame, substractMask);

    vector<vector<cv::Point>> maskContours;
    vector<vector<cv::Point>> grayContours;
    vector<vector<cv::Point>> originContours;

    cv::findContours(substractMask, maskContours, cv::RETR_CCOMP, cv::CHAIN_APPROX_TC89_KCOS);
    cout << "maskContours: " << maskContours.size() << endl;
    for(const vector<cv::Point> &contour: maskContours) {
        double area = cv::contourArea(contour);
        cv::Rect rect = cv::boundingRect(contour);
        Vec3b hsv = originFrame.at<Vec3b>(rect.x, rect.y);
        Scalar contourColor;
        //contourColor = cv::mean(contour);
        //cout << "Color:" << contourColor.val << endl;
//        cout << " Area: " << cv::contourArea(contour);
//        cout << " Coordinate: X->" << rect.x;
//        cout << " Y->" << rect.y;
//        cout << " W->" << rect.width;
//        cout << " H->" << rect.height << endl;
//        int B = hsv.val[0];
//        int G = hsv.val[1];
//        int R = hsv.val[2];
//        cout << "Color: B->" << B;
//        cout << " G->" << G;
//        cout << " R->" << R << "\n";
        if (area > 600) {
            cv::rectangle(maskLines, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), cv::Scalar(0,255,0), 2);
        }
    }

    cv::Mat mergeMatWithHistogram;
    originFrame = originFrame + maskLines;
    mergeMatWithHistogram.push_back(originFrame);
    cv::Mat histMat;
    histogram(image, histMat);
    mergeMatWithHistogram.push_back(histMat);

    UIImage *convertedImage = matToImage(mergeMatWithHistogram);
    return convertedImage;

}

@end
