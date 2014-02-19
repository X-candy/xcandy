#pragma once
#include "common.h"
struct DETECTRECT
{
	vector<int> idx;
	vector<Rect> detect_rect;
	vector<Point> detect_rect_center;
	long frame_num;
	DETECTRECT()
	{
		idx.clear();
		detect_rect.clear();
	}
};

struct I_TRACK_LINK
{
	DWORD id;
	int t_start;
	int t_end;
	int length;
	Mat omega;
	Mat xy_data;
	vector<Rect> detect_rect;
	double rank;
	I_TRACK_LINK()
	{
		id =0xfffffff;
		t_start=0;
		t_end=0;
		length=0;
		omega=Mat();
		xy_data=Mat();
		rank=INF;
		detect_rect.clear();
	}
};

struct DEFAULT_PARAMS
{
	int similarity_method;
	float min_similarity;
	float mota_threhold;
	bool debug;
	int hor;
	float eta_max;
	BOOL qcheck;
	double gap;
	double hor_max ;
	double gap_max ;
	double slope_max;

	DEFAULT_PARAMS()
	{
		similarity_method=1;
		min_similarity=0.01;
		mota_threhold=0.5;
		hor = 80;
		eta_max = 3;
		debug=1;
		gap=0;
		qcheck=false;
		hor_max = INF;
		gap_max = INF;
		slope_max= INF;
	}
};


struct RESULTS
{
	double  rank;
	double max_rank;
	double min_rank;
	BOOL qcheck;
	double gap;
	double lambda;

	double nr;
	double nc;
	int MaxIter;
	double Tol;
	RESULTS()
	{
		rank=INF;
		qcheck=0;
		max_rank=INF;
		min_rank=INF;
		gap = INF;
		nr =INF;
		nc=INF;
		MaxIter=1000;
		Tol=0.0000001;
		lambda=0.1;
	}
};




class CTracker
{
public:
	CTracker(void);
	~CTracker(void);
	int tracker(int _frame_num,vector<Rect> _detect_rect,Mat _frame);
	vector<I_TRACK_LINK> m_itl;
private:
	int InputDetectRect(vector<DETECTRECT> &_detect_rect_squence,int _frame_num,vector<Rect> _detect_rect);
	int FindAssociations(vector<DETECTRECT> &_detect_rect,int _ratio_threhold,vector<Mat> &_b,vector<Mat> &_distance);
	int LinkDetectionTracklets(vector<DETECTRECT> &_detect_rect_squence,vector<Mat> _b,vector<Mat> _distance,vector<I_TRACK_LINK> &_itl);
	int CalRectDistance(DETECTRECT _detect_rect_t,DETECTRECT _detect_rect_tp1,int _ratio_threhold,Mat& _mat_distance,Mat& _rlt);
	int getxychain(vector<DETECTRECT> &_detect_rect_squence,vector<Mat> &_b,int _frame,int _ind,Mat &_xy);
	void nonunique(Mat _a,Mat &_b);
	void DiffMat(Mat _a,Mat &_b);
	int associate_itl(vector<I_TRACK_LINK> _itl,int _t_start,int _t_end);
	int get_itl_horizon(vector<I_TRACK_LINK> _itl,int _t_start,int _t_end,vector<I_TRACK_LINK> &_itlh);
	int compute_itl_similarity_matrix(vector<I_TRACK_LINK> &_itl,DEFAULT_PARAMS _param);
	int m_dSimilarity_method;
	int m_ratio_threhold;
	int m_nHor;
	float m_dMin_similarity;
	float m_dMota_threhold;
	float m_dEta_max;
	bool m_bDebug;
	bool m_bQcheck;
	double m_dGap;
	double m_dHor_max ;
	double m_dGap_max ;
	double m_dSlope_max;
	long m_frame_num;
	int m_nHistoryLen;
	vector<DETECTRECT> m_detect_rect_squence;
	vector<Mat> m_B;
	vector<Mat> m_distanceSQ;

	DEFAULT_PARAMS m_param;
	Mat m_frame;
	Mat m_Ratio;
	Mat m_f;
	double m_lastminD2;


	//REALTIME
	int LinkDetectionTracklets(vector<DETECTRECT> &_detect_rect_squence,Mat _b,int _ind,Mat _xy);
};

