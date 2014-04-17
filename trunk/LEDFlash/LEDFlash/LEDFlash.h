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

enum {
	FILTER_LOW_GLPF=1,//gaussian��ͨ�˲���
	FILTER_LOW_ILPF=2,//�����ͨ�˲���
	FILTER_LOW_BLPF=3,//��˹��ͨ�˲���
	FILTER_HIGH_GLPF=4,//gaussian��ͨ�˲���
	FILTER_HIGH_ILPF=5,//�����ͨ�˲���
	FILTER_HIGH_BLPF=6//��˹��ͨ�˲���
};

class CLEDFlash
{
public:
	CLEDFlash(void);
	~CLEDFlash(void);
	void operator()(Mat _img);
	int DFTTransform(Mat _img, Mat &_mag);
	int convolveDFT(Mat _A, Mat _B, Mat &_C);
	Mat m_frame;
	int low_and_high_pass_filters(Mat _img,Mat &_rlt,int _ntype);

	void gernerationFilter(int _M,int _N, int _ntype);
};
