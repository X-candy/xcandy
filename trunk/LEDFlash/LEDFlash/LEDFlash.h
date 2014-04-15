#pragma once
#include "opencv/cxcore.h"
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv/ml.h"
using namespace cv;
class CLEDFlash
{
public:
	CLEDFlash(void);
	~CLEDFlash(void);
	void operator()(Mat _img);

}
