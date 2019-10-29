/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2015, ITU/ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
 *    be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

/** \file     TComPic.cpp
    \brief    picture class
*/

#include "TComPic.h"
#include "SEI.h"

//TComPicYuv* TComPic::p_pcPicYuvPred = NULL;

void print_pic(const char* file_name,TComPicYuv* rec_pic,char* md)
{   
	Pel* rec_pic_y = rec_pic->getAddr(COMPONENT_Y,0);
	Pel* rec_pic_u = rec_pic->getAddr(COMPONENT_Cb,0);
	Pel* rec_pic_v = rec_pic->getAddr(COMPONENT_Cr,0);
	int height, width;
    height = rec_pic->getHeight(COMPONENT_Y);
	width=rec_pic->getWidth(COMPONENT_Y);

	const int Y_size = height*width;
	uint8_t* ucy=new uint8_t[Y_size];
	uint8_t* ucu=new uint8_t[Y_size/4];
	uint8_t* ucv=new uint8_t[Y_size/4];
	int strideY = rec_pic->getStride(COMPONENT_Y);
	int strideU = rec_pic->getStride(COMPONENT_Cb);
	int strideV = rec_pic->getStride(COMPONENT_Cr);
	for (int i=0;i<height;i++)
	{
		for (int j = 0; j < width; j++)
		{
			ucy[i*width + j] = abs(rec_pic_y[i*strideY + j]);
		}
	}
	for (int i = 0; i<height/2; i++)
	{
		for (int j = 0; j<width/2; j++)
		{
			ucu[i*width/2 + j] = abs(rec_pic_u[i*strideU + j]);
			ucv[i*width/2 + j] = abs(rec_pic_v[i*strideU + j]);
		}
	}

	FILE* ruf = fopen(file_name, md);
	fwrite(ucy, sizeof(uint8_t), Y_size, ruf);
	fwrite(ucu, sizeof(uint8_t), Y_size/4, ruf);
	fwrite(ucv, sizeof(uint8_t), Y_size/4, ruf);
	fclose(ruf);
	delete[] ucy;
	delete[] ucu;
	delete[] ucv;
}

void read_pic(const char* file_name, TComPicYuv* rec_pic)
{
	int height, width;
	height = rec_pic->getHeight(COMPONENT_Y);
	width = rec_pic->getWidth(COMPONENT_Y);
	const int Y_size = height * width;

	uint8_t* ucy = new uint8_t[Y_size];
	uint8_t* ucu = new uint8_t[Y_size / 4];
	uint8_t* ucv = new uint8_t[Y_size / 4];
	FILE* ruf = fopen(file_name, "rb");
	fread(ucy, sizeof(uint8_t), Y_size, ruf);
	fread(ucu, sizeof(uint8_t), Y_size / 4, ruf);
	fread(ucv, sizeof(uint8_t), Y_size / 4, ruf);
	fclose(ruf);

	Pel* rec_pic_y = rec_pic->getAddr(COMPONENT_Y, 0);
	Pel* rec_pic_u = rec_pic->getAddr(COMPONENT_Cb, 0);
	Pel* rec_pic_v = rec_pic->getAddr(COMPONENT_Cr, 0);
	int strideY = rec_pic->getStride(COMPONENT_Y);
	int strideU = rec_pic->getStride(COMPONENT_Cb);
	int strideV = rec_pic->getStride(COMPONENT_Cr);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			 rec_pic_y[i*strideY + j] = ucy[i*width + j];
		}
	}
	for (int i = 0; i < height / 2; i++)
	{
		for (int j = 0; j < width / 2; j++)
		{
			rec_pic_u[i*strideU + j] = ucu[i*width / 2 + j];
			rec_pic_v[i*strideU + j] = ucv[i*width / 2 + j];
		}
	}

	delete[] ucy;
	delete[] ucu;
	delete[] ucv;
}

void PrintPredPic(const char* file_name, TComPicYuv* rec_pic, TComPicYuv* resi_pic, char* md)
{
	Pel* rec_pic_y = rec_pic->getAddr(COMPONENT_Y, 0);
	Pel* rec_pic_u = rec_pic->getAddr(COMPONENT_Cb, 0);
	Pel* rec_pic_v = rec_pic->getAddr(COMPONENT_Cr, 0);

	Pel* resi_pic_y = resi_pic->getAddr(COMPONENT_Y, 0);
	Pel* resi_pic_u = resi_pic->getAddr(COMPONENT_Cb, 0);
	Pel* resi_pic_v = resi_pic->getAddr(COMPONENT_Cr, 0);

	int height, width;
	height = rec_pic->getHeight(COMPONENT_Y);
	width = rec_pic->getWidth(COMPONENT_Y);

	FILE* fid = fopen("height_width", "wb+");
	short hw[2];
	hw[0] = (short)height;
	hw[1] = (short)width;
	fwrite(hw, sizeof(short), sizeof(hw), fid);
	fclose(fid);

	const int Y_size = height*width;
	UChar* ucy = new UChar[Y_size];
	UChar* ucu = new UChar[Y_size / 4];
	UChar* ucv = new UChar[Y_size / 4];
	int strideY = rec_pic->getStride(COMPONENT_Y);
	int strideU = rec_pic->getStride(COMPONENT_Cb);
	int strideV = rec_pic->getStride(COMPONENT_Cr);
	for (int i = 0; i<height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			ucy[i*width + j] = (UChar)rec_pic_y[i*strideY + j];
		}
	}
	for (int i = 0; i<height / 2; i++)
	{
		for (int j = 0; j<width / 2; j++)
		{
			ucu[i*width / 2 + j] = (UChar)rec_pic_u[i*strideU + j];
			ucv[i*width / 2 + j] = (UChar)rec_pic_v[i*strideU + j];
		}
	}

	strideY = resi_pic->getStride(COMPONENT_Y);
	strideU = resi_pic->getStride(COMPONENT_Cb);
	strideV = resi_pic->getStride(COMPONENT_Cr);
	for (int i = 0; i<height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			resi_pic_y[i*strideY + j] = (ucy[i*width + j] -= resi_pic_y[i*strideY + j]);
		}
	}
	for (int i = 0; i<height / 2; i++)
	{
		for (int j = 0; j<width / 2; j++)
		{
			resi_pic_u[i*strideU + j] = (ucu[i*width / 2 + j] -= resi_pic_u[i*strideU + j]);
			resi_pic_v[i*strideU + j] = (ucv[i*width / 2 + j] -= resi_pic_v[i*strideU + j]);
		}
	}
	if (md != NULL){
		FILE* ruf = fopen(file_name, md);
		fwrite(ucy, sizeof(UChar), Y_size, ruf);
		fwrite(ucu, sizeof(UChar), Y_size / 4, ruf);
		fwrite(ucv, sizeof(UChar), Y_size / 4, ruf);
		fclose(ruf); 
	}
	delete[] ucy;
	delete[] ucu;
	delete[] ucv;
}

//! \ingroup TLibCommon
//! \{

// ====================================================================================================================
// Constructor / destructor / create / destroy
// ====================================================================================================================

TComPic::TComPic()
: m_uiTLayer                              (0)
, m_bUsedByCurr                           (false)
, m_bIsLongTerm                           (false)
, m_pcPicYuvPred                          (NULL)
, m_pcPicYuvResi                          (NULL)
, m_bReconstructed                        (false)
, m_bNeededForOutput                      (false)
, m_uiCurrSliceIdx                        (0)
, m_bCheckLTMSB                           (false)
{
  for(UInt i=0; i<NUM_PIC_YUV; i++)
  {
    m_apcPicYuv[i]      = NULL;
  }
}

TComPic::~TComPic()
{
}

Void TComPic::create( const TComSPS &sps, const TComPPS &pps, const Bool bIsVirtual)
{
  const ChromaFormat chromaFormatIDC = sps.getChromaFormatIdc();
  const Int          iWidth          = sps.getPicWidthInLumaSamples();
  const Int          iHeight         = sps.getPicHeightInLumaSamples();
  const UInt         uiMaxCuWidth    = sps.getMaxCUWidth();
  const UInt         uiMaxCuHeight   = sps.getMaxCUHeight();
  const UInt         uiMaxDepth      = sps.getMaxTotalCUDepth();

  max_cu_width = uiMaxCuWidth;
  max_cu_height = uiMaxCuHeight;
  max_depth = uiMaxDepth;
  p_bUseMargin = true;

  m_picSym.create( sps, pps, uiMaxDepth );
  m_pcPicYuvResi = new TComPicYuv;  
  m_pcPicYuvResi->create(iWidth, iHeight, chromaFormatIDC, uiMaxCuWidth, uiMaxCuHeight, uiMaxDepth, true);
  m_pcPicYuvPred = new TComPicYuv;
  m_pcPicYuvPred->create(iWidth, iHeight, chromaFormatIDC, uiMaxCuWidth, uiMaxCuHeight, uiMaxDepth, true);

  if (!bIsVirtual)
  {
    m_apcPicYuv[PIC_YUV_ORG    ]   = new TComPicYuv;  m_apcPicYuv[PIC_YUV_ORG     ]->create( iWidth, iHeight, chromaFormatIDC, uiMaxCuWidth, uiMaxCuHeight, uiMaxDepth, true );
    m_apcPicYuv[PIC_YUV_TRUE_ORG]  = new TComPicYuv;  m_apcPicYuv[PIC_YUV_TRUE_ORG]->create( iWidth, iHeight, chromaFormatIDC, uiMaxCuWidth, uiMaxCuHeight, uiMaxDepth, true );
  }
  m_apcPicYuv[PIC_YUV_REC]  = new TComPicYuv;  m_apcPicYuv[PIC_YUV_REC]->create( iWidth, iHeight, chromaFormatIDC, uiMaxCuWidth, uiMaxCuHeight, uiMaxDepth, true );

  // there are no SEI messages associated with this picture initially
  if (m_SEIs.size() > 0)
  {
    deleteSEIs (m_SEIs);
  }
  m_bUsedByCurr = false;

  if (p_pcPicYuvPred == NULL)
  {
	  p_pcPicYuvPred = new TComPicYuv();
	  TComPicYuv* PicYuvRec = getPicYuvRec();
	  p_pcPicYuvPred->create(PicYuvRec->getWidth(COMPONENT_Y), PicYuvRec->getHeight(COMPONENT_Y), PicYuvRec->getChromaFormat(),
		  max_cu_width, max_cu_height, max_depth, p_bUseMargin);
  }

}

Void TComPic::destroy()
{
  m_picSym.destroy();

  for(UInt i=0; i<NUM_PIC_YUV; i++)
  {
    if (m_apcPicYuv[i])
    {
      m_apcPicYuv[i]->destroy();
      delete m_apcPicYuv[i];
      m_apcPicYuv[i]  = NULL;
    }
  }

  deleteSEIs(m_SEIs);
}

Void TComPic::compressMotion()
{
  TComPicSym* pPicSym = getPicSym();
  for ( UInt uiCUAddr = 0; uiCUAddr < pPicSym->getNumberOfCtusInFrame(); uiCUAddr++ )
  {
    TComDataCU* pCtu = pPicSym->getCtu(uiCUAddr);
    pCtu->compressMV();
  }
}

Bool  TComPic::getSAOMergeAvailability(Int currAddr, Int mergeAddr)
{
  Bool mergeCtbInSliceSeg = (mergeAddr >= getPicSym()->getCtuTsToRsAddrMap(getCtu(currAddr)->getSlice()->getSliceCurStartCtuTsAddr()));
  Bool mergeCtbInTile     = (getPicSym()->getTileIdxMap(mergeAddr) == getPicSym()->getTileIdxMap(currAddr));
  return (mergeCtbInSliceSeg && mergeCtbInTile);
}

UInt TComPic::getSubstreamForCtuAddr(const UInt ctuAddr, const Bool bAddressInRaster, TComSlice *pcSlice)
{
  UInt subStrm;
  const bool bWPPEnabled=pcSlice->getPPS()->getEntropyCodingSyncEnabledFlag();
  const TComPicSym &picSym            = *(getPicSym());

  if ((bWPPEnabled && picSym.getFrameHeightInCtus()>1) || (picSym.getNumTiles()>1)) // wavefronts, and possibly tiles being used.
  {
    if (bWPPEnabled)
    {
      const UInt ctuRsAddr                = bAddressInRaster?ctuAddr : picSym.getCtuTsToRsAddrMap(ctuAddr);
      const UInt frameWidthInCtus         = picSym.getFrameWidthInCtus();
      const UInt tileIndex                = picSym.getTileIdxMap(ctuRsAddr);
      const UInt numTileColumns           = (picSym.getNumTileColumnsMinus1()+1);
      const TComTile *pTile               = picSym.getTComTile(tileIndex);
      const UInt firstCtuRsAddrOfTile     = pTile->getFirstCtuRsAddr();
      const UInt tileYInCtus              = firstCtuRsAddrOfTile / frameWidthInCtus;
      // independent tiles => substreams are "per tile"
      const UInt ctuLine                  = ctuRsAddr / frameWidthInCtus;
      const UInt startingSubstreamForTile =(tileYInCtus*numTileColumns) + (pTile->getTileHeightInCtus()*(tileIndex%numTileColumns));
      subStrm = startingSubstreamForTile + (ctuLine - tileYInCtus);
    }
    else
    {
      const UInt ctuRsAddr                = bAddressInRaster?ctuAddr : picSym.getCtuTsToRsAddrMap(ctuAddr);
      const UInt tileIndex                = picSym.getTileIdxMap(ctuRsAddr);
      subStrm=tileIndex;
    }
  }
  else
  {
    // dependent tiles => substreams are "per frame".
    subStrm = 0;
  }
  return subStrm;
}


//! \}
