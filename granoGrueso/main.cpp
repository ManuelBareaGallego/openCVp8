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

// Para cronometrar tiempos, se pueden usar las clases definidas en estos
// archivos de cabecera, o bien la función omp_get_wtime()

#include "QueryPerformanceTiming_rdtsc.h"
#include "QueryPerformanceTiming_windows_hpc.h"

using namespace cv;
using namespace std;

int main(void)
{
	QPTimerRDTSC crono;  // usa la clase QPTimerHPC en su lugar si quieres usar la medida del tiempo vía Windows	                
	Mat src, dst;
	String ventana_src = "Ventana imagen original";
	String ventana_dst = "Ventana imagen procesada";

	namedWindow(ventana_src, WINDOW_AUTOSIZE); // Create a window for display.
	namedWindow(ventana_dst, WINDOW_AUTOSIZE); // Create a window for display.

	crono.Calibrate(); //calibrates timer overhead and set cronometer to zero	

	src = imread("cat_eye.jpg", CV_LOAD_IMAGE_COLOR); // Read a JPG file
	dst = imread("cat_eye.jpg", CV_LOAD_IMAGE_COLOR); // Read a JPG file (don't process imagen, actually, for this example)

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

	imshow(ventana_src, src);   // Mostrar imagen original
	imshow(ventana_dst, dst);   // Mostrar imagen procesada (en este ejemplo en realidad no se le ha hecho nada)

	printf("\nPress key to close output window...");
	waitKey(0); // Wait for any keystroke in the window

	destroyWindow(ventana_src); //destroy the created window
	destroyWindow(ventana_dst); //destroy the created window

	return 0;
}
