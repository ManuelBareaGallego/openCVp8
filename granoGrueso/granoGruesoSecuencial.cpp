//SPD_P11_plantillaOpenMP.cpp

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <vector>
#include <numeric>

using namespace cv;
using namespace std;

#ifdef __cplusplus 
#define ourImread(filename, isColor) cvLoadImage(filename, isColor)
#else
#define ourImread(filename, isColor) imread(filename, isColor)
#endif


int threshold_value = 0;
int const max_BINARY_value = 255;
int threshold_type = 3;

Mat procesado1(Mat img);
Mat procesado2(Mat img);
Mat procesado3(Mat img);
Mat procesado4(Mat img);

//--------------------------------------------------------
// Funciones de procesado de tiempo
double crono;
vector<double > iterCrono;

int main(int argc, char** argv)
{


	Mat src, dst1, dst2, dst3, dst4;
	String ventana_src = "Ventana imagen original";
	String ventana_dst1 = "Ventana imagen procesada 1";
	String ventana_dst2 = "Ventana imagen procesada 2";
	String ventana_dst3 = "Ventana imagen procesada 3";
	String ventana_dst4 = "Ventana imagen procesada 4";

	//time = omp_get_wtime();

	src = imread("test.jpg", CV_LOAD_IMAGE_COLOR); // Read a JPG file

	if (src.data == NULL)  // si no se pudo cargar ninguna imagen, mostrar un texto de error en ambas imágenes
	{
		src = Mat(480, 640, CV_8UC3, cv::Scalar(255, 255, 255));  // imagen de 640x480, con fondo blanco
		cv::putText(src, "Error loading input image. Check if test.jpg is",
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
		dst1 = src;
	}

	int nIt = 4;

	for (int i = 0; i < nIt; i++) {

		// Iniciamos el cronómetro para cada iteración
		crono = omp_get_wtime();

		// Procesado de imagenes
		dst1 = procesado1(src);
		dst2 = procesado2(src);
		dst3 = procesado3(src);
		dst4 = procesado4(src);

		//guarda el tiempo de procesado en el vector iterCrono
		double  wtime = omp_get_wtime() - crono;
		iterCrono.push_back(wtime);

	}

	namedWindow(ventana_src, WINDOW_NORMAL); // Create a window for display.
	namedWindow(ventana_dst1, WINDOW_NORMAL); // Create a window for display.
	namedWindow(ventana_dst2, WINDOW_NORMAL);
	namedWindow(ventana_dst3, WINDOW_NORMAL);
	namedWindow(ventana_dst4, WINDOW_NORMAL);

	imshow(ventana_src, src);   // Mostrar imagen original
	imshow(ventana_dst1, dst1);   // Mostrar imagen procesada 
	imshow(ventana_dst2, dst2);   // Mostrar imagen procesada 
	imshow(ventana_dst3, dst3);   // Mostrar imagen procesada 
	imshow(ventana_dst4, dst4);   // Mostrar imagen procesada

	auto n = iterCrono.size();
	float average = 0.0f;
	if (n != 0) {
		average = accumulate(iterCrono.begin(), iterCrono.end(), 0.0f) / n;
	}

	cout << "\nTiempo medio de ejecucion algoritmo secuencial:\n\n" + to_string(average) << std::endl;

	double best = *min_element(iterCrono.begin(), iterCrono.end());
	cout << "\nTiempo mínumo de ejecucion algoritmo secuencial:\n\n" + to_string(best) << std::endl;


	printf("\nPress key to close output window...");

	waitKey(0);
	destroyWindow(ventana_src); //destroy the created window
	destroyWindow(ventana_dst1); //destroy the created window
	destroyWindow(ventana_dst2); //destroy the created window
	destroyWindow(ventana_dst3); //destroy the created window
	destroyWindow(ventana_dst4); //destroy the created window


	return 0;
}


Mat procesado1(Mat img) {
	Mat res;
	int threshold_value = 63;
	int threshold_type = 3;
	int const max_BINARY_value = 255;
	GaussianBlur(img, res, Size(9, 9), 0);
	cvtColor(res, res, CV_RGB2GRAY);
	threshold(res, res, threshold_value, max_BINARY_value, threshold_type);
	return res;
}

Mat procesado2(Mat img) {
	Mat res;
	GaussianBlur(img, res, Size(9, 9), 0);
	cvtColor(res, res, CV_RGB2GRAY);
	equalizeHist(res, res);
	return res;
}

Mat procesado3(Mat img) {
	Mat res;
	Mat greyMat, colorMat;
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	Mat src, src_gray;
	double scale = 1.0;
	double delta = 0.0;
	int ddepth = CV_8U; // CV_16S;

	GaussianBlur(img, res, Size(9, 9), 0);
	cvtColor(res, res, CV_RGB2GRAY);

	/// Gradient X
	Sobel(res, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	/// Gradient Y
	Sobel(res, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	/// Suma Gradient X Y
	convertScaleAbs(grad_x, abs_grad_x);
	convertScaleAbs(grad_y, abs_grad_y);
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, res);
	return res;
}

Mat procesado4(Mat img) {
	Mat res;
	int kernel_size = 3;
	double scale = 1.0;
	double delta = 0.0;
	int ddepth = CV_8U; // CV_16S;

	GaussianBlur(img, res, Size(9, 9), 0);
	cvtColor(res, res, CV_RGB2GRAY);
	Laplacian(res, res, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
	//convertScaleAbs(res, res);
	return res;

}


