#include "DrawAOI.h"


CDrawAOI::CDrawAOI(void)
{
	m_image  = NULL;
	m_nWidth =0;
	m_nHeight =0;
	m_bFlag =false;

	m_arrPoint.clear();
}


CDrawAOI::~CDrawAOI(void)
{
	if(!m_image.empty())
		m_image.release();
	m_image = NULL;

	m_arrPoint.clear();
}
vector<Point> arrPoint;
bool bflag  = true;


void onMouseClick( int event, int x, int y, int, void* )
{
	Point p1;
	if(event==CV_EVENT_LBUTTONDOWN && bflag)  
	{  
		//当前鼠标位置（x，y）  
		p1.x=x;  
		p1.y=y;  
		arrPoint.push_back(p1);
		int nSize = arrPoint.size();
		printf("point %d:%d,%d\n",nSize,x,y);
	} 

	if(event==CV_EVENT_RBUTTONDOWN)
	{
		bflag =FALSE;
	}

}

int CDrawAOI::operator()(Mat _img)
{
	if (_img.empty())
	{
		return -1;
	}

	_img.copyTo(m_image);

	m_nHeight = m_image.rows;
	m_nWidth = m_image.cols;
	bflag =true;
	arrPoint.clear();
	namedWindow("drawRoi");

	while (bflag)
	{
		imshow("drawRoi",m_image);
		setMouseCallback("drawRoi",onMouseClick);
		waitKey(1);
	}
	m_arrPoint = arrPoint;
	int nSize=m_arrPoint.size();
	if(nSize<1)
		return -1;
	cv::polylines(m_image,m_arrPoint,1,Scalar(255,255,0));
	
	
	int nMaxX=0;
	int nMinX=m_arrPoint[0].x;
	int nMaxY=0;
	int nMinY=m_arrPoint[0].y;
	for(int i=0;i<m_arrPoint.size();i++)
	{
		nMaxX = nMaxX > m_arrPoint[i].x ? nMaxX : m_arrPoint[i].x ;
		nMaxY = nMaxY > m_arrPoint[i].y ? nMaxY : m_arrPoint[i].y ;
		nMinX = nMinX < m_arrPoint[i].x ? nMinX : m_arrPoint[i].x ;
		nMinY = nMinY < m_arrPoint[i].y ? nMinY : m_arrPoint[i].y ;
	}

	m_AOI.x = nMinX;
	m_AOI.y = nMinY;
	m_AOI.width = nMaxX -nMinX;
	m_AOI.height= nMaxY -nMinY;
	rectangle(m_image,m_AOI,Scalar(255,0,0));
	imshow("drawRoi",m_image);
	printf("Press any key to continue\n");
	waitKey();
	destroyWindow("drawRoi");
}


