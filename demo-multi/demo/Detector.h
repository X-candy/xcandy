#pragma once

#include "common.h"
#define CV_CVX_WHITE    CV_RGB(0xff,0xff,0xff)
#define CV_CVX_BLACK    CV_RGB(0x00,0x00,0x00)

struct DETECTOBJECT
{
	Rect detect_rect;
	Point detect_center;
	DETECTOBJECT()
	{
		detect_rect = Rect();
		detect_center = Point();
	}
};

struct BackgroundModel
{
	DWORD dwBGID;
	long ByteSize;			 //Í¼Ïñ´óÐ¡

	Mat background_frame;
	Mat threshold_Mat;
};

class CDetector
{
public:
	CDetector(void);
	~CDetector(void);
	int detectObject(Mat _frame);
	vector<Rect> m_detect_rect;
	double m_delta;
	double m_SimilarThrehold;


private:
	Scalar  getMSSIM( const Mat &i1, const Mat &i2);
	void ConnectedComponents(Mat _mask_process, vector<Rect> &_detect_rect);

	BackgroundSubtractorMOG2 m_Mog;
	long m_nFrameNum;
	Mat m_matLastFrame;

	Mat m_matBackground;
	Mat m_matForeground;

	Mat m_matBGStore;
	vector<vector<Point> > m_vContours;
	vector<Vec4i> m_vHierarchy;

	Mat m_GaussianFG;
	Mat m_FrameDiffFG;
};

