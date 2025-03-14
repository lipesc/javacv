package org.lipesc.computervision;


import java.io.IOException;
import java.net.URISyntaxException;

import static org.bytedeco.opencv.global.opencv_face.drawFacemarks;
import static org.bytedeco.opencv.global.opencv_highgui.cvWaitKey;
import static org.bytedeco.opencv.global.opencv_highgui.imshow;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.equalizeHist;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point2fVector;
import org.bytedeco.opencv.opencv_core.Point2fVectorVector;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_face.FacemarkKazemi;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

/**
 *
 * @author lipesc
 */
public class JavaCVFacemarkKazemi {

    public static void main(String[] args)  throws IOException, URISyntaxException, InterruptedException {

        //carregar face detector
        CascadeClassifier faceD = new CascadeClassifier("src/main/resources/haarcascade_frontalface_alt.xml");

        // criar instancia facemark
        FacemarkKazemi facemark = FacemarkKazemi.create();

        // carregar landmark modelo

        facemark.loadModel("src/main/resources/face_landmark_model.dat");

        // load image 
        Mat img = imread("src/main/resources/face1.jpg");


        // converter para grayscale para a detectação
        Mat grayscale = new Mat();
        cvtColor(img, grayscale, COLOR_BGR2GRAY);
        equalizeHist(grayscale, grayscale);

        // achar rostos
        RectVector faceFind = new RectVector();
        faceD.detectMultiScale(grayscale , faceFind);

        System.out.printf("quantidade de faces: %d%n", faceFind.size());

        // landmarks uma face é um vector mais pode ter mais de uma face
        Point2fVectorVector landmars = new Point2fVectorVector();

        boolean itsWorking = facemark.fit(img, faceFind, landmars);

        if(itsWorking) {
            for(long i =0; i < landmars.size(); i++) {
                Point2fVector v = landmars.get(i);
                drawFacemarks(img, v, Scalar.RED);
            }
        }

        imshow("javaCV testes:", img);
        cvWaitKey(0);
        imwrite("teste.jpg", img);

    }
}
 