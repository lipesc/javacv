package org.lipesc.computervision;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BayerBG2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_face.FacemarkKazemi;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

/**
 *
 * @author lipesc
 */
public class JavaCVFacemarkKazemi {

    public static void main(String[] args) {

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
        cvtColor(img, grayscale, COLOR_BayerBG2GRAY);
    }
}
 