#pragma once

#include "TLibCommon//TComPic.h"

//extern "C" _declspec(dllimport) 
void run_model(TComPic* pcPic, TComPicYuv* dstYuv);

typedef struct
{
	int startX;
	int endX;
	int startY;
	int endY;
}Rect;
