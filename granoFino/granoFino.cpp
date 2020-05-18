///////////////////////////////////////////////////////////////////////
// Dpto ATC. Universidad de Sevilla. http://www.atc.us.es 
//    Ejemplo de aplicación con OpenCV y medida del tiempo de ejecución
//    usando La biblioteca QueryPerfomanceTiming (hay dos versiones)
// (c)2020 Miguel Angel Rodriguez Jodar. Día 57 de confinamiento.
//
//---------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>  // para el soporte OpenMP
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

//--------------------------------------------------------
// Funciones de procesado de tiempo
double crono;
vector<double > iterCrono;
Mat src, dst;
double alpha = 2.0;
int beta = 0;

int main(void)
{	                
	
	String ventana_src = "Ventana imagen original";
	String ventana_dst = "Ventana imagen procesada";

	namedWindow(ventana_src, WINDOW_AUTOSIZE); // Create a window for display.
	namedWindow(ventana_dst, WINDOW_AUTOSIZE); // Create a window for display.

	src = imread("test.jpg", CV_LOAD_IMAGE_COLOR); // Read a JPG file
	dst = src.clone();
	//threshold(res, res, threshold_value, max_BINARY_value, threshold_type);

	if (src.data == NULL)  // si no se pudo cargar ninguna imagen, mostrar un texto de error en ambas imágenes
	{
		src = Mat(480, 640, CV_8UC3, cv::Scalar(255, 255, 255));  // imagen de 640x480, con fondo blanco
		cv::putText (src, "Error loading input image. Check if cat_eye.jpg is", 
			cv::Point(0, 20), 
			cv::FONT_HERSHEY_SIMPLEX,
            0.75,
            CV_RGB(0, 0, 0),
            2);
		cv::putText(src, "in the same directory as this executable file",
			cv::Point(0, 50),
			cv::FONT_HERSHEY_SIMPLEX,
			0.75,
			CV_RGB(0, 0, 0),
			2);
		dst = src;
	}

	//procesado imagen
	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) {
			for (int c = 0; c < 3; c++) {
				dst.at<Vec3b>(y, x)[c] =
					saturate_cast<uchar>(alpha*(src.at<Vec3b>(y, x)[c]) + beta);
			}
		}
	}

	imshow(ventana_src, src);   // Mostrar imagen original
	imshow(ventana_dst, dst);   // Mostrar imagen procesada (en este ejemplo en realidad no se le ha hecho nada)

	printf("\nPress key to close output window...");
	waitKey(0); // Wait for any keystroke in the window

	destroyWindow(ventana_src); //destroy the created window
	destroyWindow(ventana_dst); //destroy the created window

	return 0;
}


