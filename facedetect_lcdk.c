/*
 * facedetect_lcdk.c
 *
 * This file contains face detection demo code for LCDK.
 *
 * Copyright (C) 2009 Texas Instruments Incorporated - http://www.ti.com/
 *
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *    Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 *    Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the
 *    distribution.
 *
 *    Neither the name of Texas Instruments Incorporated nor the names of
 *    its contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

/**
 *  \file   facedetect_lcdk.c
 *
 *  \brief  This file contains the face detection demo code for LCDK.
 *
 *          This application captures the live camera input which is then
 * 			passed to the DSP. The DSP runs the face detection algorithm
 * 			provided by OpenCV and draws a rectangular box around the
 * 			detected face. The output is displayed over LCDC.
 *
 */

/* ========================================================================== */
/*                          INCLUDE FILES                                     */
/* ========================================================================== */
#include <xdc/std.h>
#include <xdc/runtime/Error.h>
#include <xdc/runtime/System.h>
#include <ti/sysbios/BIOS.h>
#include <ti/sysbios/knl/Clock.h>
#include <ti/sysbios/knl/Task.h>
#include <ti/sysbios/knl/Semaphore.h>
#include <ti/sysbios/family/c64p/Hwi.h>

#include <stdio.h>
#include "psc.h"
#include "vpif.h"
#include "raster.h"
#include "interrupt.h"
#include "lcdkC6748.h"
#include "soc_C6748.h"
#include "hw_psc_C6748.h"
#include "adv7343.h"
#include "tvp5147.h"
#include "cdce913.h"
#include "codecif.h"
#include "i2cgpio.h"
#include "cv.h"
#include "cxcore.h"
#include "math.h"
#include "cxtypes.h"
#include "string.h"
#include "facedetect.h"

/******************************************************************************
**                      INTERNAL MACROS
*******************************************************************************/
#define MSTPRI1									(0x01C14114)
#define MSTPRI2									(0x01C14118)

#define CAPTURE_IMAGE_WIDTH					 	(720)
#define CAPTURE_IMAGE_HEIGHT				 	(488)
#define DISPLAY_IMAGE_WIDTH						(640)
#define DISPLAY_IMAGE_HEIGHT					(480)
#define CLASSIFIER_CASCADE_SIZE      			(52519)
#define REFERENCE_LOCATION_1         			(0xC30011B0)
#define REFERENCE_LOCATION_2		 			(0xC303464B)
#define CV_DEFAULT_IMAGE_ROW_ALIGN   			(8)

#define I2C_SLAVE_CODEC_ADV7343                 (0x2Au)
#define I2C_SLAVE_CODEC_TVP5147_1_SVIDEO    	(0x5Cu)
#define I2C_SLAVE_CODEC_TVP5147_2_COMPOSITE    	(0x5Du)
#define I2C_SLAVE_CDCE913      					(0x65u)
#define I2C_SLAVE_UI_EXPANDER	      			(0x20u)

#define INT_CHANNEL_I2C                       	(0x2u)

#define OFFSET1									((CAPTURE_IMAGE_HEIGHT - DISPLAY_IMAGE_HEIGHT)/2)
#define OFFSET2									((CAPTURE_IMAGE_WIDTH - DISPLAY_IMAGE_WIDTH)/2)
#define OFFSET 									(CAPTURE_IMAGE_WIDTH * OFFSET1 + OFFSET2)

/******************************************************************************
**                      GLOBAL FUNCTION PROTOTYPES
*******************************************************************************/
extern void cbcr422sp_to_rgb565_c(const unsigned char * restrict, unsigned int,
		unsigned int, const short*, const unsigned char *restrict,
		unsigned int, unsigned short *restrict, unsigned int, unsigned);

/******************************************************************************
**                      INTERNAL FUNCTION PROTOTYPES
*******************************************************************************/
static void SetUpVPIFRx(void);
static void SetUpLCD(void);
Void VPIFIsr(UArg arg);
Void LCDIsr(UArg arg);

Void colorconvertTask(UArg arg0, UArg arg1);
//Void facedetectTask(UArg arg0, UArg arg1);
Void handdetectTask(UArg arg0, UArg arg1);
Void clockHandler(UArg arg);

/******************************************************************************
**                      INTERNAL VARIABLE DEFINITIONS
*******************************************************************************/
unsigned int            pingpong=1, buffcount=0, buffcount2, display_buff_1=0;
int 					facedetect_complete=1, buffer_ready=0;
volatile unsigned int   changed=1, updated=3, processed=1;
unsigned char 			*buff_luma[2], *buff_chroma[2];
unsigned char			buff_luma1[CAPTURE_IMAGE_WIDTH * CAPTURE_IMAGE_HEIGHT];
unsigned char			buff_luma2[CAPTURE_IMAGE_WIDTH * CAPTURE_IMAGE_HEIGHT];
unsigned char			buff_chroma1[CAPTURE_IMAGE_WIDTH * CAPTURE_IMAGE_HEIGHT];
unsigned char			buff_chroma2[CAPTURE_IMAGE_WIDTH * CAPTURE_IMAGE_HEIGHT];
unsigned char			proc_buff_luma[DISPLAY_IMAGE_WIDTH * DISPLAY_IMAGE_HEIGHT];
unsigned char           *videoTopY, *videoTopC, *proc_buff_ptr;
unsigned short          *videoTopRgb1, *videoTopRgb2;
unsigned short          Rgb_buffer1[DISPLAY_IMAGE_WIDTH*DISPLAY_IMAGE_HEIGHT + 16];
unsigned short			Rgb_buffer2[DISPLAY_IMAGE_WIDTH*DISPLAY_IMAGE_HEIGHT + 16];
const short             ccCoeff[5] = {0x2000, 0x2BDD, -0x0AC5, -0x1658, 0x3770};
const int 				HAND_AREA_THRESHOLD = 30000;

CvHaarClassifierCascade 	*cascade;
IplImage 					*image=NULL, *image1=NULL;
CvMemStorage 				*storage;
CvSeq 						*dsp_sequence = NULL, *dsp_sequence2 = NULL;
CvScalar 	  				blue = {255, 0, 0, 0};
CvPoint 					c1, c2;
CvRect 						maxAreaRect;
CvPoint						max_tl, max_br;
int							maxArea;

Semaphore_Handle 			sem1;
Semaphore_Handle 			sem2;

/******************************************************************************
**                       FUNCTION DEFINITIONS
*******************************************************************************/
Void main()
{
	Hwi_Params 		params;
	Hwi_Handle 		Hwi1, Hwi2;
	Task_Params 	taskParams;
	Task_Handle 	task1, task2;
	Clock_Params 	clockParams;
	Clock_Handle 	Clk;
	Error_Block 	eb;

    /* Create a Semaphore object to be use as a resource lock */
    sem1 = Semaphore_create(0, NULL, NULL);
    sem2 = Semaphore_create(0, NULL, NULL);

    // TODO: replace this in main with our own task
	/* Create the Color Convert Task */
    Error_init(&eb);
	Task_Params_init(&taskParams);
	taskParams.priority = 1;
	taskParams.stackSize = 10000;
	task1 = Task_create(colorconvertTask, &taskParams, &eb);
	if (task1 == NULL)
	{
		System_printf("Task_create() failed for Color Conversion!\n");
		BIOS_exit(0);
	}

	// TODO: replace this in main with our own task
	/* Create the FaceDetect Task*/
	task2 = Task_create(facedetectTask, &taskParams, &eb);
	if (task2 == NULL)
	{
		System_printf("Task_create() failed for Face Detection!\n");
		BIOS_exit(0);
	}


	/* Create the VPIF Interrupt*/
	Hwi_Params_init(&params);
	params.eventId = 95;
	params.enableInt = FALSE;
	Hwi1 = Hwi_create(8, &VPIFIsr, &params, &eb);
	if (Hwi1 == NULL)
	{
		System_printf("Hwi_create() failed for VPIF!\n");
		BIOS_exit(0);
	}

	// NOTE: this interrupt contains the drawing rectangle stuff
	/* Create the LCDC Interrupt*/
	/*Hwi_Params_init(&params);
	params.eventId = 73;
	params.enableInt = FALSE;
	Hwi2 = Hwi_create(9, &LCDIsr, &params, &eb);
	if (Hwi2 == NULL)
	{
		System_printf("Hwi_create() failed for LCDC!\n");
		BIOS_exit(0);
	}*/

	/* Create a clock that would go off every 1 Clock tick*/
	Clock_Params_init(&clockParams);
	clockParams.period = 1;
	clockParams.startFlag = TRUE;
	Clk = Clock_create((Clock_FuncPtr)clockHandler, 4, &clockParams, &eb);
	if (Clk == NULL) {
	System_abort("Clock0 create failed");
	}

	/* Setting the Master priority for LCD DMA controller to highest level */
	*((volatile uint32_t *) MSTPRI2) = *((volatile uint32_t *) MSTPRI2) & 0x0FFFFFFF;

	BIOS_start(); /* enable interrupts and start SYS/BIOS */
}

/*
** ClockHandler
*/
Void clockHandler(UArg arg)
{
	/* Call Task_yield every 1 ms */
	Task_yield();
}

/*
** ColorconvertTask - This would color convert the captured frames and make it available for the LCDC display.
** Also, if the facedetect algorithm has finished working on the frame provided, it would provide a new frame.
*/
Void colorconvertTask(UArg a0, UArg a1)
{
	int   i, *buffer;

	/* Allocate and initialize 'image' which would then be passed to the face detection algorithm */
	image = (IplImage *) cvAlloc(sizeof(*image));
	cvInitImageHeader(image, cvSize(DISPLAY_IMAGE_WIDTH, DISPLAY_IMAGE_HEIGHT), IPL_DEPTH_8U, 1,
			IPL_ORIGIN_TL, CV_DEFAULT_IMAGE_ROW_ALIGN);

	/* Allocate and initialize 'image1' which would then be used to draw a box on the output buffer */
	image1 = (IplImage *) cvAlloc(sizeof(*image1));
	cvInitImageHeader(image1, cvSize(DISPLAY_IMAGE_WIDTH, DISPLAY_IMAGE_HEIGHT), IPL_DEPTH_16U,
			1, IPL_ORIGIN_TL, CV_DEFAULT_IMAGE_ROW_ALIGN);

	/* Initialize the dsp_sequnce which would contain the face coordinates */
	dsp_sequence2 = (CvSeq *) cvAlloc(sizeof(CvSeq) * 2);
	memset(dsp_sequence2, 0, sizeof(CvSeq) * 2);

	/* Allocate pointers to buffers */
	buff_luma[0] = buff_luma1;
	buff_luma[1] = buff_luma2;
	buff_chroma[0] = buff_chroma1;
	buff_chroma[1] = buff_chroma2;
	proc_buff_ptr = proc_buff_luma;

	/* Initializing palette for first buffer */
	*Rgb_buffer1 = 0x4000;
	for (i = 1; i < 16; i++)
		*(Rgb_buffer1 +i) = 0x0000;
	videoTopRgb1 = Rgb_buffer1 + i;

	/* Initializing palette for second buffer */
	*Rgb_buffer2 = 0x4000;
	for (i = 1; i < 16; i++)
		*(Rgb_buffer2 + i) = 0x0000;
	videoTopRgb2 = Rgb_buffer2 + i;

	/* Power on VPIF */
	PSCModuleControl(SOC_PSC_1_REGS, HW_PSC_VPIF, PSC_POWERDOMAIN_ALWAYS_ON,
			PSC_MDCTL_NEXT_ENABLE);

	/* Initialize I2C and program UI GPIO expander, TVP5147, and ADV7343 via I2C */
	I2CPinMuxSetup(0);

	/*Initialize the TVP5147 to accept composite video */
	I2CCodecIfInit(SOC_I2C_0_REGS, INT_CHANNEL_I2C,
			I2C_SLAVE_CODEC_TVP5147_2_COMPOSITE);
	TVP5147CompositeInit(SOC_I2C_0_REGS);

	/* Setup VPIF pinmux */
	VPIFPinMuxSetup();

	/* Setup LCD */
	SetUpLCD();

	/* Initialize VPIF */
	SetUpVPIFRx();
	VPIFDMARequestSizeConfig(SOC_VPIF_0_REGS, VPIF_REQSIZE_ONE_TWENTY_EIGHT);
	VPIFEmulationControlSet(SOC_VPIF_0_REGS, VPIF_HALT);

	/* Initialize buffer addresses for 1st frame*/
	VPIFCaptureFBConfig(SOC_VPIF_0_REGS, VPIF_CHANNEL_0, VPIF_TOP_FIELD,
			VPIF_LUMA, (unsigned int) buff_luma[0], CAPTURE_IMAGE_WIDTH*2);
	VPIFCaptureFBConfig(SOC_VPIF_0_REGS, VPIF_CHANNEL_0, VPIF_TOP_FIELD,
			VPIF_CHROMA, (unsigned int) buff_chroma[0], CAPTURE_IMAGE_WIDTH*2);
	VPIFCaptureFBConfig(SOC_VPIF_0_REGS, VPIF_CHANNEL_0, VPIF_BOTTOM_FIELD,
			VPIF_LUMA, (unsigned int) (buff_luma[0] + CAPTURE_IMAGE_WIDTH), CAPTURE_IMAGE_WIDTH*2);
	VPIFCaptureFBConfig(SOC_VPIF_0_REGS, VPIF_CHANNEL_0, VPIF_BOTTOM_FIELD,
			VPIF_CHROMA, (unsigned int) (buff_chroma[0] + CAPTURE_IMAGE_WIDTH), CAPTURE_IMAGE_WIDTH*2);

	/* Configuring the base ceiling */
	RasterDMAFBConfig(SOC_LCDC_0_REGS, (unsigned int) Rgb_buffer2,
			(unsigned int) (Rgb_buffer2 + DISPLAY_IMAGE_WIDTH * DISPLAY_IMAGE_HEIGHT + 15), 0);
	RasterDMAFBConfig(SOC_LCDC_0_REGS, (unsigned int) Rgb_buffer2,
			(unsigned int) (Rgb_buffer2 + DISPLAY_IMAGE_WIDTH * DISPLAY_IMAGE_HEIGHT + 15), 1);

	/* Enable interrupts */
	Hwi_enable();
	Hwi_enableInterrupt(8);
	Hwi_enableInterrupt(9);

	/* Enable capture */
	VPIFCaptureChanenEnable(SOC_VPIF_0_REGS, VPIF_CHANNEL_0);

	/* Enable VPIF interrupt */
	VPIFInterruptEnable(SOC_VPIF_0_REGS, VPIF_FRAMEINT_CH0);
	VPIFInterruptEnableSet(SOC_VPIF_0_REGS, VPIF_FRAMEINT_CH0);

	/* Enable End of frame interrupt */
	//RasterEndOfFrameIntEnable(SOC_LCDC_0_REGS);

	/* Enable raster */
	//RasterEnable(SOC_LCDC_0_REGS);

	buffcount++;
	buffcount2 = buffcount - 1;

	/* Run forever */
	while (1)
	{
		/* Wait here till a new frame is not captured */
		Semaphore_pend(sem1, BIOS_WAIT_FOREVER);

		/* Process the next buffer only when both the raster buffers
		 * are pointing to the current buffer to avoid jitter effect
		 */
		Semaphore_pend(sem2, BIOS_WAIT_FOREVER);
		processed = 0;
		changed = 0;
		updated = 0;

		// TODO: change this - replace facedetect_complete. image init is okay.
		/* Copy the current frame into another buffer for running the facedetect algorithm */
		if(facedetect_complete)
		{
			facedetect_complete=0;
			for(i=0;i<DISPLAY_IMAGE_HEIGHT;i++)
				memcpy((proc_buff_ptr + i*DISPLAY_IMAGE_WIDTH),(videoTopY + OFFSET +i*CAPTURE_IMAGE_WIDTH),
						DISPLAY_IMAGE_WIDTH);
			image->imageData = (char *)proc_buff_ptr;
			buffer_ready=1;
		}

	    /* Convert the buffer from CBCR422 semi-planar to RGB565,
	     *  Set the flag for the buffer to be displayed on the LCD (which would be the processed buffer)
	     *  and notify the LCD of availability of a processed buffer.
	     *  The output buffers are ping-ponged each time.
	     */
		if (pingpong)
		{
			cbcr422sp_to_rgb565_c(
					(const unsigned char *) (videoTopC + OFFSET),
					DISPLAY_IMAGE_HEIGHT, CAPTURE_IMAGE_WIDTH, ccCoeff,
					(const unsigned char *) (videoTopY + OFFSET),
					CAPTURE_IMAGE_WIDTH, videoTopRgb1, DISPLAY_IMAGE_WIDTH,
					DISPLAY_IMAGE_WIDTH);
			display_buff_1 = 1;
		}
		else
		{
			cbcr422sp_to_rgb565_c(
					(const unsigned char *) (videoTopC + OFFSET),
					DISPLAY_IMAGE_HEIGHT, CAPTURE_IMAGE_WIDTH, ccCoeff,
					(const unsigned char *) (videoTopY + OFFSET),
					CAPTURE_IMAGE_WIDTH, videoTopRgb2, DISPLAY_IMAGE_WIDTH,
					DISPLAY_IMAGE_WIDTH);
			display_buff_1 = 0;
		}
		pingpong = !pingpong;
		processed = 1;
		changed = 1;
	}
}

// TODO: change this. replace this function
/*
** FacedetectTask - Runs the Face Detection Algorithm on the frame provided to it
*/

CvRect handBorderDetect(CvMemStorage* storage)
{
	const double thresh = 20;
	const double maxval = 255;

	cvCvtColor(image, image, CV_BGR2GRAY);
	cvSmooth(image, image, CV_GAUSSIAN, 7, 7, 1.5, 0);

	IplImage* thresholdImg;
	thresholdImg = (IplImage*)cvAlloc(sizeof(*thresholdImg));
	cvInitImageHeader(thresholdImg, cvSize(DISPLAY_IMAGE_WIDTH, DISPLAY_IMAGE_HEIGHT), IPL_DEPTH_8U, 1,
				IPL_ORIGIN_TL, CV_DEFAULT_IMAGE_ROW_ALIGN);
	cvThreshold(image, thresholdImg, thresh, maxval, CV_THRESH_BINARY);

	// create bound around hand
	IplImage* canny_output;
	canny_output = (IplImage*)cvAlloc(sizeof(*canny_output));
	cvInitImageHeader(canny_output, cvSize(DISPLAY_IMAGE_WIDTH, DISPLAY_IMAGE_HEIGHT), IPL_DEPTH_8U, 1,
						IPL_ORIGIN_TL, CV_DEFAULT_IMAGE_ROW_ALIGN);
	cvCanny(thresholdImg, canny_output, thresh, thresh * 2, 3);

	CvSeq* contours; // points to the first contour detected
	cvFindContours(canny_output, storage, &contours, sizeof(CvContour), CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0));

	cvReleaseImage(&canny_output);
	cvReleaseImage(&thresholdImg);


}

Void facedetectTask(UArg a0, UArg a1)
{
	int i;
	while(1)
	{
		while(!buffer_ready)
			Task_sleep(10);
		buffer_ready=0;

		storage = cvCreateMemStorage(0);

		// Apply OpenCV haardetect algorithm
		dsp_sequence = handBorderDetect(storage);

		// Make a copy of the detected sequence to draw it on each frame
		for (i = 0; i < 2; i++)
			*(dsp_sequence2 + i) = *(dsp_sequence + i);

		cvReleaseMemStorage(&storage);

		facedetect_complete=1;
	}
}

double calcDistance(int x1, int y1, int x2, int y2)
{
    double result = ( (x1 - x2) * (x1 - x2) ) + ( (y1 - y2) * (y1 - y2) );
    result = sqrt(result);
    return result;
}

/*
 * Initialize capture
 */
static void SetUpVPIFRx(void)
{
	/* Disable interrupts */
	VPIFInterruptDisable(SOC_VPIF_0_REGS, VPIF_FRAMEINT_CH1);
	VPIFInterruptDisable(SOC_VPIF_0_REGS, VPIF_FRAMEINT_CH0);

	/* Disable capture ports */
	VPIFCaptureChanenDisable(SOC_VPIF_0_REGS, VPIF_CHANNEL_1);
	VPIFCaptureChanenDisable(SOC_VPIF_0_REGS, VPIF_CHANNEL_0);

	/* Interrupt after capturing the bottom field of every frame */
	VPIFCaptureIntframeConfig(SOC_VPIF_0_REGS, VPIF_CHANNEL_0, VPIF_FRAME_INTERRUPT_BOTTOM);

	/* Y/C interleaved capture over 8-bit bus */
	VPIFCaptureYcmuxModeSelect(SOC_VPIF_0_REGS, VPIF_CHANNEL_0, VPIF_YC_MUXED);

	/* Capturing 480I (SD NTSC) */
	VPIFCaptureModeConfig(SOC_VPIF_0_REGS, VPIF_480I, VPIF_CHANNEL_0, 0, (struct vbufParam *) 0);
}

/*
 * Configures raster to display image
 */
static void SetUpLCD(void)
{
    PSCModuleControl(SOC_PSC_1_REGS, HW_PSC_LCDC, PSC_POWERDOMAIN_ALWAYS_ON,
		     PSC_MDCTL_NEXT_ENABLE);

    LCDPinMuxSetup();

    /* disable raster */
    RasterDisable(SOC_LCDC_0_REGS);

    /* configure the pclk */
    RasterClkConfig(SOC_LCDC_0_REGS, 25000000, 150000000);

    /* configuring DMA of LCD controller */
    RasterDMAConfig(SOC_LCDC_0_REGS, RASTER_DOUBLE_FRAME_BUFFER,
                    RASTER_BURST_SIZE_16, RASTER_FIFO_THRESHOLD_8,
                    RASTER_BIG_ENDIAN_DISABLE);

    /* configuring modes(ex:tft or stn,color or monochrome etc) for raster controller */
    RasterModeConfig(SOC_LCDC_0_REGS, RASTER_DISPLAY_MODE_TFT,
                     RASTER_PALETTE_DATA, RASTER_MONOCHROME, RASTER_RIGHT_ALIGNED);

    /* frame buffer data is ordered from least to Most significant bye */
    RasterLSBDataOrderSelect(SOC_LCDC_0_REGS);

    /* disable nibble mode */
    RasterNibbleModeDisable(SOC_LCDC_0_REGS);

     /* configuring the polarity of timing parameters of raster controller */
    RasterTiming2Configure(SOC_LCDC_0_REGS, RASTER_FRAME_CLOCK_LOW |
                                            RASTER_LINE_CLOCK_LOW  |
                                            RASTER_PIXEL_CLOCK_LOW |
                                            RASTER_SYNC_EDGE_RISING|
                                            RASTER_SYNC_CTRL_ACTIVE|
                                            RASTER_AC_BIAS_HIGH     , 0, 255);

    /* configuring horizontal timing parameter */
   RasterHparamConfig(SOC_LCDC_0_REGS, DISPLAY_IMAGE_WIDTH, 64, 48, 48);

    /* configuring vertical timing parameters */
   RasterVparamConfig(SOC_LCDC_0_REGS, DISPLAY_IMAGE_HEIGHT, 2, 11, 31);

   /* configuring fifo delay to */
   RasterFIFODMADelayConfig(SOC_LCDC_0_REGS, 2);
}

/*
** VPIF Interrupt service routine.
*/
Void VPIFIsr(UArg arg)
{
	CvScalar red = CV_RGB(0, 0, 255);
	CvPoint tl, br;
	tl.x = 50;
	tl.y = 50;
	br.x = 250;
	br.y = 250;
	cvRectangle(image1, tl, br, red, 1, 8, 0);
	if (dsp_sequence == NULL);
	else
	{
		// get rectangles of all contours and find the one with max area
		maxArea = -1;


		CvSeq* points;

		while (contours)
		{
			points = cvApproxPoly(contours, sizeof(CvContour), storage, CV_POLY_APPROX_DP, 3, 1);
			CvRect boundRect = cvBoundingRect(points, 0);

			int currArea = boundRect.height * boundRect.width;
			if (currArea > maxArea)
			{
				maxAreaRect = boundRect;
				maxArea = currArea;
				max_tl.x = boundRect.x;
				max_tl.y = boundRect.y;
				max_br.x = boundRect.x + boundRect.width;
				max_br.y = boundRect.y + boundRect.height;
			}

			contours = contours->h_next;
		}



		int cog_x = 0;
		int cog_y = 0;
		// find cog_x, cog_y
		if (maxArea > HAND_AREA_THRESHOLD) // has the hand region
		{
			int r, c;
			// get all the black points from hand region
			int num_pixels = 0;
			int lowest_row = max_tl.y + 1;
			int highest_row = max_br.y - 1;
			int lowest_col = max_tl.x + 1;
			int highest_col = max_br.x - 1;

			double sum_x = 0;
			double sum_y = 0;
			for (r = lowest_row; r < highest_row; r++)
			{
				for (c = lowest_col; c < highest_col; c++)
				{
					if (CV_IMAGE_ELEM(thresholdImg, unsigned char, r, c)  == 0)
					{
						sum_x += c;
						sum_y += r;
						num_pixels++;
					}
				}
			}
			cog_x = (sum_x / num_pixels + 0.5);
			cog_y = (sum_y / num_pixels + 0.5);

			int max_dist_from_cog = -1;
			char closestEdge = '0';
			int maxDist_x = -1;
			int maxDist_y = -1;
			// go through all the edges of rectangle and calc distance to cog
			// left edge:
			for (r = lowest_row; r <= highest_row; r++)
			{
				if (CV_IMAGE_ELEM(thresholdImg, unsigned char, r, lowest_col) == 0)
				{
					int distance = calcDistance(lowest_col, r, cog_x, cog_y) + 0.5;
					if (distance > max_dist_from_cog)
					{
						max_dist_from_cog = distance;
						closestEdge = 'l';
						maxDist_x = lowest_col;
						maxDist_y = r;
					}
				}
			}
			// right edge:
			for (r = lowest_row; r <= highest_row; r++)
			{
				if (CV_IMAGE_ELEM(thresholdImg, unsigned char, r, highest_col) == 0)
				{
					int distance = calcDistance(highest_col, r, cog_x, cog_y) + 0.5;
					if (distance > max_dist_from_cog)
					{
						max_dist_from_cog = distance;
						closestEdge = 'r';
						maxDist_x = highest_col;
						maxDist_y = r;
					}
				}
			}
			// top edge:
			for (c = lowest_col; c <= highest_col; c++)
			{
				if (CV_IMAGE_ELEM(thresholdImg, unsigned char, lowest_row, c) == 0)
				{
					int distance = calcDistance(c, lowest_row, cog_x, cog_y) + 0.5;
					if (distance > max_dist_from_cog)
					{
						max_dist_from_cog = distance;
						closestEdge = 't';
						maxDist_x = c;
						maxDist_y = lowest_row;
					}
				}
			}
			// bot edge:
			for (c = lowest_col; c <= highest_col; c++)
			{
				if (CV_IMAGE_ELEM(thresholdImg, unsigned char, highest_row, c) == 0)
				{
					int distance = calcDistance(c, highest_row, cog_x, cog_y) + 0.5;
					if (distance > max_dist_from_cog)
					{
						max_dist_from_cog = distance;
						closestEdge = 'b';
						maxDist_x = c;
						maxDist_y = highest_row;
					}
				}
			}

			CvScalar blue = CV_RGB(255, 0, 0);
			CvScalar red = CV_RGB(0, 0, 255);
			switch (closestEdge)
			{
			case 'l':

				cvRectangle(image1, max_tl, max_br, blue, 1, 8, 0);
				//printf("right\n"); // webcam inverts left and right
				break;
			case 'r':
				cvRectangle(image1, max_tl, max_br, red, 1, 8, 0);
				//printf("left\n"); // webcam inverts left and right
				break;
			default:
				//printf("no turn\n");
			}
		}
		else
		{
			//printf("Hand region not found\n");
		}
	}

#ifdef _TMS320C6X
	IntEventClear(SYS_INT_VPIF_INT);
#endif

	/* If previously captured frame is not processed, clear this interrupt and return */
	if (!processed)
	{
		VPIFInterruptStatusClear(SOC_VPIF_0_REGS, VPIF_FRAMEINT_CH0);
		return;
	}

	/* buffcount represents buffer to be given to capture driver and
	 * buffcount2 represents the newly captured buffer to be processed
	 */
	processed = 0;
	buffcount++;
	buffcount2 = buffcount - 1;
	// Currently only two buffers are being used for capture
	if (buffcount == 2)
		buffcount = 0;

	/* Initialize buffer addresses for a new frame*/
	VPIFCaptureFBConfig(SOC_VPIF_0_REGS, VPIF_CHANNEL_0, VPIF_TOP_FIELD,
			VPIF_LUMA, (unsigned int) buff_luma[buffcount], CAPTURE_IMAGE_WIDTH*2);
	VPIFCaptureFBConfig(SOC_VPIF_0_REGS, VPIF_CHANNEL_0, VPIF_TOP_FIELD,
			VPIF_CHROMA, (unsigned int) buff_chroma[buffcount], CAPTURE_IMAGE_WIDTH*2);
	VPIFCaptureFBConfig(SOC_VPIF_0_REGS, VPIF_CHANNEL_0, VPIF_BOTTOM_FIELD,
			VPIF_LUMA, (unsigned int) (buff_luma[buffcount] + CAPTURE_IMAGE_WIDTH), CAPTURE_IMAGE_WIDTH*2);
	VPIFCaptureFBConfig(SOC_VPIF_0_REGS, VPIF_CHANNEL_0, VPIF_BOTTOM_FIELD,
			VPIF_CHROMA, (unsigned int) (buff_chroma[buffcount] + CAPTURE_IMAGE_WIDTH), CAPTURE_IMAGE_WIDTH*2);

	/* Initialize buffer addresses with the captured frame ready to be processed */
	videoTopC = buff_chroma[buffcount2];
	videoTopY = buff_luma[buffcount2];

	/* clear interrupt */
	VPIFInterruptStatusClear(SOC_VPIF_0_REGS, VPIF_FRAMEINT_CH0);

	Semaphore_post(sem1);
}

/***************************** End Of File ************************************/
