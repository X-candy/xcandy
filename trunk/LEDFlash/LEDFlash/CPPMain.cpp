#pragma once
#include <iostream>
#include <Windows.h>
#include <time.h>
#include <opencv2\opencv.hpp>
#include "DrawAOI.h"
using namespace cv;
using namespace std;

#if _DEBUG
#pragma comment(lib,"opencv_core245d.lib")
#pragma comment(lib,"opencv_imgproc245d.lib")
#pragma comment(lib,"opencv_highgui245d.lib")
#pragma comment(lib,"opencv_ml245d.lib")
#pragma comment(lib,"opencv_photo245d.lib")
#pragma comment(lib,"opencv_video245d.lib")
#pragma comment(lib,"opencv_features2d245d.lib")
#pragma comment(lib,"opencv_calib3d245d.lib")
#pragma comment(lib,"opencv_objdetect245d.lib")
#pragma comment(lib,"opencv_contrib245d.lib")
#pragma comment(lib,"opencv_legacy245d.lib")
#pragma comment(lib,"opencv_flann245d.lib")
#else
#pragma comment(lib,"opencv_core245.lib")
#pragma comment(lib,"opencv_imgproc245.lib")
#pragma comment(lib,"opencv_highgui245.lib")
#pragma comment(lib,"opencv_ml245.lib")
#pragma comment(lib,"opencv_video245.lib")
#pragma comment(lib,"opencv_photo245.lib")
#pragma comment(lib,"opencv_features2d245.lib")
#pragma comment(lib,"opencv_calib3d245.lib")
#pragma comment(lib,"opencv_objdetect245.lib")
#pragma comment(lib,"opencv_contrib245.lib")
#pragma comment(lib,"opencv_legacy245.lib")
#pragma comment(lib,"opencv_flann245.lib")
#endif

int main(int argc,char** argv)
{
	char lpsPath[1024]="E:\\videobg\\flash.avi";
	VideoCapture capture;
	Mat frame;
	capture.open(lpsPath);

	if(!capture.isOpened())
		return -1;

	capture>>frame;

	CDrawAOI AOI;
	AOI(frame);


	while (1)
	{
		capture>>frame;

		if(frame.empty())
			break;
		rectangle(frame,AOI.m_AOI,Scalar(255,0,0));
		


		imshow("frame",frame);
		waitKey(1);
	}
	return 1;
}