#include "Detector.h"

CDetector::CDetector(void)
{
	m_delta = -0.5;
	m_SimilarThrehold = 0.97;
	m_nFrameNum = 0;
	m_detect_rect.clear();
}


CDetector::~CDetector(void)
{
}

Scalar CDetector::getMSSIM( const Mat &i1, const Mat &i2)
{
	const double C1 = 6.5025, C2 = 58.5225;
	/***************************** INITS **********************************/
	int d = CV_32F;

	Mat I1, I2;
	i1.convertTo(I1, d);
	i2.convertTo(I2, d);

	Mat I2_2   = I2.mul(I2);        // I2^2
	Mat I1_2   = I1.mul(I1);        // I1^2
	Mat I1_I2  = I1.mul(I2);        // I1 * I2

	/*************************** END INITS **********************************/

	Mat mu1, mu2;                   // PRELIMINARY COMPUTING
	GaussianBlur(I1, mu1, Size(11, 11), 1.5);
	GaussianBlur(I2, mu2, Size(11, 11), 1.5);

	Mat mu1_2   =   mu1.mul(mu1);
	Mat mu2_2   =   mu2.mul(mu2);
	Mat mu1_mu2 =   mu1.mul(mu2);

	Mat sigma1_2, sigma2_2, sigma12;

	GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
	sigma1_2 -= mu1_2;

	GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
	sigma2_2 -= mu2_2;

	GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
	sigma12 -= mu1_mu2;

	///////////////////////////////// FORMULA ////////////////////////////////
	Mat t1, t2, t3;

	t1 = 2 * mu1_mu2 + C1;
	t2 = 2 * sigma12 + C2;
	t3 = t1.mul(t2);                 // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

	t1 = mu1_2 + mu2_2 + C1;
	t2 = sigma1_2 + sigma2_2 + C2;
	t1 = t1.mul(t2);                 // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

	Mat ssim_map;
	divide(t3, t1, ssim_map);        // ssim_map =  t3./t1;

	Scalar mssim = mean(ssim_map);   // mssim = average of ssim map
	return mssim;
}

void CDetector::ConnectedComponents(Mat _mask_process, vector<Rect> &_detect_rect)
{

	Mat mask;
	_mask_process.copyTo(mask);
	//开闭操作
	//窗口大小，最优值待讨论
	int an = 3;
	//窗口的形状
	int element_shape = MORPH_RECT;
	Mat element = getStructuringElement(element_shape, Size(an * 2 + 1, an * 2 + 1), Point(an, an) );
	morphologyEx(mask, mask, CV_MOP_OPEN, element);
	morphologyEx(mask, mask, CV_MOP_CLOSE, element);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;


	findContours( mask, contours, hierarchy,	CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
	if(contours.size() == 0)
		return ;

	vector<vector<Point> > contours_poly( contours.size() );
	vector<Rect> boundRect;
	for( int idx = 0; idx >= 0; idx = hierarchy[idx][0] )
	{
		//对多边形曲线做适当近似
		approxPolyDP( Mat(contours[idx]), contours_poly[idx], 3, true );
		//计算并返回包围轮廓点集的最小矩形
		boundRect.push_back(boundingRect( Mat(contours_poly[idx]) ));
		drawContours( mask, contours, idx, Scalar(255, 255, 255), CV_FILLED, 8, hierarchy );
	}
	mask.setTo(0);
	for(int i = 0; i < boundRect.size(); i++)
	{
		rectangle(mask, boundRect[i], Scalar(255, 0, 0), -1);
	}

	contours.clear();
	hierarchy.clear();
	findContours( mask, contours, hierarchy,	CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
	if(contours.size() == 0)
		return ;

	_detect_rect.clear();

	for(int idx = 0; idx >= 0; idx = hierarchy[idx][0] )
	{
		//对多边形曲线做适当近似
		approxPolyDP(Mat(contours[idx]) , contours_poly[idx], 3, true );
		//计算并返回包围轮廓点集的最小矩形
		_detect_rect.push_back(boundingRect( Mat(contours_poly[idx]) ));
	}
}

int getObject(Mat _foreground)
{


	return 1;
}
//return 1:背景无改变
//return 2：背景发生改变
int CDetector::detectObject(Mat _frame)
{
	//主程序路径
	if(_frame.channels() < 3)
		return -1;

	//高斯背景模型学习与前景获取
	m_Mog(_frame, m_GaussianFG, m_delta);

	//帧差分获取前景图像
	Mat gray;
	cvtColor(_frame, gray, CV_RGB2GRAY);
	Mat diffMat;
	Mat mat1(5, 5, CV_8U, cv::Scalar(1));
	if(!m_matLastFrame.empty())
	{
		absdiff(gray, m_matLastFrame, diffMat);
		Mat canny_output;

		Canny( diffMat, canny_output, 100, 200 , 3 );

		cv::dilate(canny_output, canny_output, mat1);
		findContours( canny_output, m_vContours, m_vHierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
		m_FrameDiffFG = Mat::zeros( canny_output.size(), CV_8UC1 );
		for( int i = 0; i < m_vContours.size(); i++ )
		{
			drawContours(m_FrameDiffFG, m_vContours, i, Scalar(255), CV_FILLED);
		}
		gray.copyTo(m_matLastFrame);
	}
	else
	{
		gray.copyTo(m_matLastFrame);
	}

	if(!m_FrameDiffFG.empty())
	{
		Mat mask;
		if(mask.empty())
		{
			mask.create(m_FrameDiffFG.rows, m_FrameDiffFG.cols, CV_8UC3);
		}

		mask.setTo(Scalar(0, 0, 0));
		bitwise_and(m_GaussianFG, m_FrameDiffFG, m_matForeground);
		dilate(m_matForeground, m_matForeground, mat1);
		threshold(m_matForeground, m_matForeground, 50, 255, CV_IMWRITE_PXM_BINARY);
		m_detect_rect.clear();
		ConnectedComponents(m_matForeground, m_detect_rect);
	}

#pragma region background process
	//获取模型的背景图片
	m_Mog.getBackgroundImage(m_matBackground);
	getObject(m_matForeground);

	if(m_matBGStore.empty())
	{
		m_matBGStore = Mat(m_matBackground.rows, m_matBackground.cols, m_matBackground.type());
		m_matBackground.copyTo(m_matBGStore);
	}

	//检测存储背景有无发生改变
	//检测间隔
	if(!m_matBGStore.empty() && m_nFrameNum % 1000 == 0)
	{
		//计算背景间相似度
		Scalar mssim = getMSSIM(m_matBGStore, m_matBackground);
		double mssimMean = 0;
		int count = 0;
		for(int i = 0; i < 4; i++)
		{
			if (mssim.val[i] != 0)
			{
				mssimMean = mssim.val[i] + mssimMean;
				count++;
			}
		}

		mssimMean = mssimMean / count;
		if(mssimMean < m_SimilarThrehold)
		{
			m_matBackground.copyTo(m_matBGStore);
			return 2;
		}
	}
	return 1;
#pragma endregion background process
}



