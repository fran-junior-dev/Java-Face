package org.example;

import nu.pattern.OpenCV;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

public class Main {
    public static void main(String[] args) {
        // 1. Load the OpenCV library
        OpenCV.loadLocally();
        System.out.println("OpenCV Loaded. Booting up webcam...");

        // 2. Initialize the Webcam (0 is usually the default laptop camera)
        VideoCapture camera = new VideoCapture(0);
        if (!camera.isOpened()) {
            System.out.println("Error: Cannot open webcam! Check your permissions.");
            return;
        }

        // 3. Load the Face Detector (The file you just downloaded)
        CascadeClassifier faceDetector = new CascadeClassifier("haarcascade_frontalface_default.xml");
        if (faceDetector.empty()) {
            System.out.println("Error: Could not load the cascade XML file. Is it in the right folder?");
            return;
        }

        Mat frame = new Mat();
        MatOfRect faceDetections = new MatOfRect();

        System.out.println("Webcam active! Press 'ESC' on the video window to close it.");

        // 4. The Real-Time Video Loop
        while (true) {
            camera.read(frame); // Grab a single frame from the camera
            if (frame.empty()) break;

            // Detect faces in that frame
            faceDetector.detectMultiScale(frame, faceDetections);

            // 5. Draw a box around every face found
            for (Rect rect : faceDetections.toArray()) {
                // Draw the rectangle
                Imgproc.rectangle(
                        frame,
                        new Point(rect.x, rect.y),
                        new Point(rect.x + rect.width, rect.y + rect.height),
                        new Scalar(0, 255, 0), // Green box (Blue, Green, Red)
                        3 // Thickness
                );

                // Placeholder text for where our Recognition data will go!
                Imgproc.putText(
                        frame,
                        "Scanning Identity...",
                        new Point(rect.x, rect.y - 10),
                        Imgproc.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        new Scalar(0, 255, 0),
                        2
                );
            }

            // 6. Show the video feed in a window
            HighGui.imshow("Java Real-Time Face Tracker", frame);

            // Wait 30ms for a key press. If the key is 'Esc' (ASCII 27), break the loop
            if (HighGui.waitKey(30) == 27) {
                break;
            }
        }

        // 7. Clean up and release the camera when done
        camera.release();
        HighGui.destroyAllWindows();
        System.exit(0);
    }
}