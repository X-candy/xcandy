#pragma once
#include "opencv/cxcore.h"
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv/ml.h"

using namespace cv;


class CDrawAOI
{
public:
	CDrawAOI(void);
	~CDrawAOI(void);

	int operator()(Mat _img);
	vector<Point> m_arrPoint;
	Rect m_AOI;
	//	void onMouseClick( int event, int x, int y, int, void* );
private:
	Mat m_image;
	int m_nWidth;
	int m_nHeight;
	

	bool m_bFlag;
};

