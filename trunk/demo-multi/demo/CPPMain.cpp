#include "common.h"
#include "Detector.h"
#include "tracker.h"
int main()
{
	char lpFileName[100]="f:\\test_352x288_clip.MP4";
	unsigned char* pData=NULL;
	VideoCapture capture;
	capture.open(lpFileName);
	if(!capture.isOpened())
		return -1;

	CDetector detector;
	CTracker tracker;
	Mat frame;
	vector<Rect> last_rect;
	vector<Point> point;
	long frameNum=0;
	int c=-1;
	while (1)
	{
		cout<<"frameNum="<<frameNum<<endl;
		capture>>frame;
		if(frame.empty())
			break;

		detector.detectObject(frame);
		imshow("frame",frame);
		for(int i=0;i<detector.m_detect_rect.size();i++)
		{
			rectangle(frame,detector.m_detect_rect[i],Scalar(255,0,0));
		}

		if(last_rect.size()>0)
		{
			for(int i=0;i<last_rect.size();i++)
			{
				rectangle(frame,last_rect[i],Scalar(255,255,0));
			}
		}

		
		imshow("_frame",frame);
		tracker.tracker(frameNum,detector.m_detect_rect,frame);
		//检测前景的目标信息
	
		frameNum++;
		if(c==-1)
			c=waitKey(1);
		else if(c==' ')
		{
			c=waitKey();
		}else
		{
			c=waitKey(1);
		}

		last_rect = detector.m_detect_rect;
	}
	return 1;
}