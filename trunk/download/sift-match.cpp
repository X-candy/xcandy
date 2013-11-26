#include "opencv2/opencv.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/legacy/legacy.hpp"
#include <iostream>
#include <ctype.h>


#if _DEBUG
#pragma comment(lib,"opencv_core245d.lib")
#pragma comment(lib,"opencv_imgproc245d.lib")
#pragma comment(lib,"opencv_highgui245d.lib")
#pragma comment(lib,"opencv_ml245d.lib")
#pragma comment(lib,"opencv_photo245d.lib")
#pragma comment(lib,"opencv_video245d.lib")
#pragma comment(lib,"opencv_features2d245d.lib")
#pragma comment(lib,"opencv_nonfree245d.lib")
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
#pragma comment(lib,"opencv_nonfree245.lib")
#pragma comment(lib,"opencv_calib3d245.lib")
#pragma comment(lib,"opencv_objdetect245.lib")
#pragma comment(lib,"opencv_contrib245.lib")
#pragma comment(lib,"opencv_legacy245.lib")
#pragma comment(lib,"opencv_flann245.lib")
#endif


using namespace cv;
using namespace std;

bool selectObject = false;
int trackObject = 0;
int InitialTrack = 0;
Point origin;
Mat frame;
Rect selectRect;
vector<Rect> selection;

static void onMouse( int event, int x, int y, int, void * )
{
    if( selectObject )
    {
        selectRect.x = MIN(x, origin.x);
        selectRect.y = MIN(y, origin.y);
        selectRect.width = std::abs(x - origin.x);
        selectRect.height = std::abs(y - origin.y);

        selectRect &= Rect(0, 0, frame.cols, frame.rows);

    }

    switch( event )
    {
    case CV_EVENT_LBUTTONDOWN:
        origin = Point(x, y);
        selectRect = Rect(x, y, 0, 0);
        selectObject = true;
        break;
    case CV_EVENT_LBUTTONUP:
        selectObject = false;

        if(InitialTrack && selectRect.width > 0 && selectRect.height > 0)
            selection.push_back(selectRect);
        break;
    }
}
#define CV_CVX_WHITE    CV_RGB(0xff,0xff,0xff)
#define CV_CVX_BLACK    CV_RGB(0x00,0x00,0x00)

void ConnectedComponents(Mat &_mask_process, int _poly1_hull0, float _perimScale, int &_number,
                         vector<Rect> &_bounding_box, vector<Point> &_contour_centers)
{
    /*����4�������Ϊ�˼���ԭ�����ӿڣ����ڲ�ʹ�õ���c��񣬵�����ӿ���c++����*/
    IplImage *mask = &_mask_process.operator IplImage();
    int *num = &_number;
    static CvMemStorage    *mem_storage    = NULL;
    static CvSeq            *contours    = NULL;
    //CLEAN UP RAW MASK
    //���������ã�ƽ��������ȥ��ϸ��,�Ͽ�ȱ��
    cvMorphologyEx( mask, mask, NULL, NULL, CV_MOP_OPEN, 1 );//������mask���п�������CVCLOSE_ITRΪ�������Ĵ��������Ϊmaskͼ��
    //���������ã�ƽ������������ȱ��
    cvMorphologyEx( mask, mask, NULL, NULL, CV_MOP_CLOSE, 1 );//������mask���бղ�����CVCLOSE_ITRΪ�ղ����Ĵ��������Ϊmaskͼ��
    //FIND CONTOURS AROUND ONLY BIGGER REGIONS
    if( mem_storage == NULL ) mem_storage = cvCreateMemStorage(0);
    else cvClearMemStorage(mem_storage);
    //CV_RETR_EXTERNAL=0����types_c.h�ж���ģ�CV_CHAIN_APPROX_SIMPLE=2Ҳ���ڸ��ļ��ж����
    CvContourScanner scanner = cvStartFindContours(mask, mem_storage, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    CvSeq *c;
    int numCont = 0;
    //��while�ڲ�ֻ��ԱȽϴ���������߽����滻����
    while( (c = cvFindNextContour( scanner )) != NULL )
    {
        double len = cvContourPerimeter( c );
        double q = (mask->height + mask->width) / _perimScale;  //calculate perimeter len threshold
        if( len < q ) //Get rid of blob if it's perimeter is too small
        {
            cvSubstituteContour( scanner, NULL );    //��NULL����ԭ�����Ǹ�����
        }
        else //Smooth it's edges if it's large enough
        {
            CvSeq *c_new;
            if(_poly1_hull0) //Polygonal approximation of the segmentation
                c_new = cvApproxPoly(c, sizeof(CvContour), mem_storage, CV_POLY_APPROX_DP, 2, 0);
            else //Convex Hull of the segmentation
                c_new = cvConvexHull2(c, mem_storage, CV_CLOCKWISE, 1);
            cvSubstituteContour( scanner, c_new ); //�ʼ��������͹�����߶���ʽ��������滻
            numCont++;
        }
    }
    contours = cvEndFindContours( &scanner );    //�����������Ҳ���
    // PAINT THE FOUND REGIONS BACK INTO THE IMAGE
    cvZero( mask );
    IplImage *maskTemp;
    //CALC CENTER OF MASS AND OR BOUNDING RECTANGLES
    if(*num != 0)
    {
        int N = *num, numFilled = 0, i = 0;
        CvMoments moments;
        double M00, M01, M10;
        maskTemp = cvCloneImage(mask);
        for(i = 0, c = contours; c != NULL; c = c->h_next, i++ )   //h_nextΪ���������е���һ������
        {
            if(i < N) //Only process up to *num of them
            {
                //CV_CVX_WHITE�ڱ��������ǰ�ɫ����˼
                cvDrawContours(maskTemp, c, CV_CVX_WHITE, CV_CVX_WHITE, -1, CV_FILLED, 8);
                //Find the center of each contour
                Point centers;
                if(centers != Point(-1, -1))
                {
                    cvMoments(maskTemp, &moments, 1);  //����maskͼ�����ߴ�3�׵ľ�
                    M00 = cvGetSpatialMoment(&moments, 0, 0); //��ȡx��0�κ�y��0�ξ�
                    M10 = cvGetSpatialMoment(&moments, 1, 0); //��ȡx��1�κ�y��0�ξ�
                    M01 = cvGetSpatialMoment(&moments, 0, 1); //��ȡx��0�κ�y��1�ξ�

                    centers.x = (int)(M10 / M00);  //���þصĽ��������������ĵ�����
                    centers.y = (int)(M01 / M00);

                    _contour_centers.push_back(centers);
                }
                //Bounding rectangles around blobs
                Rect bbs;
                bbs = cvBoundingRect(c); //�������c����Ӿ���
                _bounding_box.push_back(bbs);
                //memcpy(&bbs[i],contour_centers,sizeof(Rect));

                cvZero(maskTemp);
                numFilled++;
            }
            //Draw filled contours into mask
            cvDrawContours(mask, c, CV_CVX_WHITE, CV_CVX_WHITE, -1, CV_FILLED, 8); //draw to central mask
        } //end looping over contours
        *num = numFilled;
        cvReleaseImage( &maskTemp);
    }
    //ELSE JUST DRAW PROCESSED CONTOURS INTO THE MASK
    else
    {
        for( c = contours; c != NULL; c = c->h_next )
        {
            cvDrawContours(mask, c, CV_CVX_WHITE, CV_CVX_BLACK, -1, CV_FILLED, 8);
        }
    }
}


void detectObject(Mat &_curframe, BackgroundSubtractorMOG2 &_mog, vector<Rect> &_rect, vector<Point> &_point, int _num)
{
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    Mat foreground;
    _mog(_curframe, foreground);
    threshold(foreground, foreground, 128, 255, CV_IMWRITE_PXM_BINARY);
    imshow("_curframe", _curframe);
    imshow("foreground1", foreground);
    int num = _num;
    ConnectedComponents(foreground, 0, 8.0, num, _rect, _point);
    imshow("foreground2", foreground);
    waitKey(20);
}


struct OBJECTDESCRIPTOR
{
    Rect cRect;
    Mat objImage;
    vector<KeyPoint> keypoints;
    Mat decriptor;
};
int main( int argc, const char **argv )
{
    char lpFileName[100] = "f:\\1.avi";

    BackgroundSubtractorMOG2 mog;
    VideoCapture pCapture;
    //pCapture.open(lpFileName);
    pCapture.open(0);
    vector<OBJECTDESCRIPTOR> arrayObj;


    int nNum = 10;
    int nFrameNum = 0;

    if( !pCapture.isOpened() )
    {
        cout << "***Could not initialize capturing...***\n";

        return -1;
    }
    namedWindow("demo");
    setMouseCallback( "demo", onMouse, 0 );

    for(;;)
    {
        pCapture >> frame;
        nFrameNum++;
        if(nFrameNum < 270)
            continue;
        printf("FrameNum=%d\n", nFrameNum);
        if(frame.empty())
            break;

        if( selection.size() > 0  && InitialTrack == 0)
        {
            Mat gray;
            cvtColor(frame, gray, CV_RGB2GRAY);
            SIFT siftFeature(0.06f, 10.0);
            vector<KeyPoint> keypoints;
            Mat decriptor;
            siftFeature.detect(gray, keypoints);
            siftFeature.compute(gray, keypoints, decriptor);

            for(int i = 0; i < arrayObj.size(); i++)
            {
                BruteForceMatcher<L2<float>> matcher;
                vector<DMatch>matches;
                matcher.match(arrayObj[i].decriptor, decriptor, matches);

                char c[2] = {0};
                c[0] = i;
                namedWindow(c, 1);
                Mat img_matches;
                drawMatches(arrayObj[i].objImage, arrayObj[i].keypoints, frame, keypoints, matches, img_matches);
                imshow(c, img_matches);

                char strFilePath[100] = {0};
                sprintf(strFilePath, "d:\\test1\\Match_%d.jpg", i);
                imwrite(strFilePath, img_matches);
            }
        }


        if(InitialTrack )
        {
            if( selectObject && selectRect.width > 0 && selectRect.height > 0 )
            {
                Mat roi(frame, selectRect);
                bitwise_not(roi, roi);
            }

            if(selection.size())
            {
                for(int i = 0; i < selection.size(); i++)
                {
                    char strFilePath[100] = {0};
                    sprintf(strFilePath, "d:\\test1\\SelectObj_%d.jpg", i);
                    Mat roi = Mat(frame, selection[i]);


                    OBJECTDESCRIPTOR objDescriptor;
                    roi.copyTo(objDescriptor.objImage);

                    Mat gray;
                    cvtColor(roi, gray, CV_RGB2GRAY);
                    SIFT siftFeature(0.06f, 10.0);
                    siftFeature.detect(gray, objDescriptor.keypoints);
                    siftFeature.compute(gray, objDescriptor.keypoints, objDescriptor.decriptor);
                    if(i + 1 > arrayObj.size())
                    {
                        arrayObj.push_back(objDescriptor);
                        imwrite(strFilePath, arrayObj[i].objImage);
                    }
                }
                trackObject = selection.size();
            }
        }

        imshow( "demo", frame );

        char c = (char)waitKey(10);
        switch(c)
        {
        case 'i':
            InitialTrack = 1;
            selection.clear();
            arrayObj.clear();
            break;
        case 't':
            imwrite("d:\\test1\\bg1.jpg", frame);
            InitialTrack = 0;
            break;
        case 'c':
            trackObject = 0;
            selection.clear();
            arrayObj.clear();
            break;
        case ' ':
            waitKey();
        }
    }


    return 0;
}

