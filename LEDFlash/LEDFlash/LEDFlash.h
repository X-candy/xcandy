#pragma once
#include "opencv/cxcore.h"
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv/ml.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace cv;


#ifndef LEDF_DEBUG
#define LEDF_DEBUG 1
#endif


class CLEDFlash
{
public:
	CLEDFlash(void);
	~CLEDFlash(void);
	void operator()(Mat _img);
	int DFTTransform(Mat _img, Mat &_mag);
	int convolveDFT(Mat _A, Mat _B, Mat &_C);
	Mat m_frame;

};
