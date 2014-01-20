#include "common.h"
struct HSRect
{
	float x;
	float y;
	float width;
	float height;
	float center_point[2];
	HSRect()
	{
		x=0;
		y=0;
		width=0;
		height=0;
		memset(center_point,0,sizeof(float)*2);
	}
};


struct DATASET
{
	char name[255];
	int id;
	int t_start;
	int t_end;
	int length;
	vector<int> omega;
	vector<HSRect> rect;
	DATASET()
	{
		memset(name,0,255);
		t_start=0;
		t_end = 0;
		length=0;
		omega.clear();
		rect.clear();
	}
};

struct DETECTRECT
{
	vector<int> idx;
	vector<HSRect> detect_rect;
	DETECTRECT()
	{
		idx.clear();
		detect_rect.clear();
	}
};


struct I_TRACK_LINK
{
	int t_start;
	int t_end;
	int length;
	Mat omega;
	Mat data;
};

int read_dataset(char* _file_path,vector<DATASET> &_dataset);
int ProcessDataSet(vector<DETECTRECT> &_detect_rect,vector<DATASET> &_dataset);
int findAssociations(vector<DETECTRECT> &_detect_rect,int _ratio_threhold,vector<Mat> &_b,vector<Mat> &_distance);
int linkDetectionTracklets(vector<DETECTRECT> &_detect_rect,vector<Mat> _b,vector<Mat> _distance);
int initial_track();