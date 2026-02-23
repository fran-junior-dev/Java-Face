package org.example;

// IMPORTANT: We use bytedeco's loader to fix the UnsatisfiedLinkError
import org.bytedeco.javacpp.Loader;
import org.bytedeco.opencv.opencv_java;
import org.opencv.core.*;
import org.opencv.face.FaceRecognizer;
import org.opencv.face.LBPHFaceRecognizer;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Scanner;

public class Main {
    static HashMap<Integer, String> idToName = new HashMap<>();
    static int currentId = 0;

    static volatile boolean isCapturing = false;
    static volatile boolean shouldExit = false;
    static volatile String currentName = "";
    static int captureCount = 0;

    public static void main(String[] args) {
        // --- FIX: The New Loading Method ---
        Loader.load(opencv_java.class);
        System.out.println("OpenCV + Face Modules Loaded. Booting...");

        File datasetFolder = new File("dataset");
        if (!datasetFolder.exists()) datasetFolder.mkdir();

        FaceRecognizer recognizer = trainModel(datasetFolder);

        VideoCapture camera = new VideoCapture(0);
        CascadeClassifier faceDetector = new CascadeClassifier("haarcascade_frontalface_default.xml");

        // --- TERMINAL COMMAND THREAD ---
        Thread commandThread = new Thread(() -> {
            Scanner scanner = new Scanner(System.in);
            while (!shouldExit) {
                System.out.println("\n[MENU] Type 'c' to Capture, 'q' to Quit:");
                String input = scanner.nextLine().toLowerCase();

                if (input.equals("c")) {
                    System.out.print("Enter name for person: ");
                    currentName = scanner.nextLine().trim();
                    if (!currentName.isEmpty()) {
                        captureCount = 0;
                        isCapturing = true;
                        System.out.println("Switch to window! Capturing 50 shots of the largest face seen...");
                    }
                } else if (input.equals("q")) {
                    shouldExit = true;
                }
            }
        });
        commandThread.setDaemon(true);
        commandThread.start();

        Mat frame = new Mat();
        Mat grayFrame = new Mat();
        MatOfRect faceDetections = new MatOfRect();

        while (!shouldExit) {
            camera.read(frame);
            if (frame.empty()) break;

            Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
            faceDetector.detectMultiScale(grayFrame, faceDetections);
            Rect[] facesArray = faceDetections.toArray();

            // Find the LARGEST face (assume this is the user)
            Rect largestFace = null;
            for (Rect rect : facesArray) {
                if (largestFace == null || (rect.width * rect.height > largestFace.width * largestFace.height)) {
                    largestFace = rect;
                }
            }

            // Loop through detected faces to draw boxes
            for (Rect rect : facesArray) {
                Scalar color = new Scalar(0, 255, 0); // Default Green

                // Recognition Logic
                Mat resizedFace = new Mat();
                Imgproc.resize(new Mat(grayFrame, rect), resizedFace, new Size(200, 200));

                if (isCapturing && rect.equals(largestFace)) {
                    // Only save the largest face
                    String filename = "dataset/" + currentName + "_" + captureCount + ".jpg";
                    Imgcodecs.imwrite(filename, resizedFace);
                    captureCount++;

                    color = new Scalar(0, 0, 255); // Red while capturing
                    Imgproc.putText(frame, "RECORDING: " + captureCount, new Point(rect.x, rect.y - 10),
                            1, 1.5, color, 2);

                    if (captureCount >= 50) isCapturing = false;
                } else if (recognizer != null) {
                    int[] label = new int[1];
                    double[] confidence = new double[1];
                    recognizer.predict(resizedFace, label, confidence);

                    double distance = confidence[0];
                    int matchPercent = (int) Math.max(0, 100 * (1 - (distance / 115.0)));

                    if (distance < 70) {
                        String name = idToName.getOrDefault(label[0], "Unknown");
                        Imgproc.putText(frame, name + " " + matchPercent + "%", new Point(rect.x, rect.y - 10),
                                1, 1.2, color, 2);
                    } else {
                        Imgproc.putText(frame, "Unknown", new Point(rect.x, rect.y - 10), 1, 1.2, new Scalar(0, 165, 255), 2);
                    }
                }

                Imgproc.rectangle(frame, rect, color, 2);
            }

            HighGui.imshow("Face Recognition - Check Terminal for Commands", frame);
            if (HighGui.waitKey(1) == 27) shouldExit = true;
        }

        camera.release();
        HighGui.destroyAllWindows();
        System.exit(0);
    }

    private static FaceRecognizer trainModel(File datasetFolder) {
        File[] files = datasetFolder.listFiles((dir, name) -> name.endsWith(".jpg"));
        if (files == null || files.length == 0) return null;

        LBPHFaceRecognizer recognizer = LBPHFaceRecognizer.create();
        List<Mat> images = new ArrayList<>();
        List<Integer> labelsList = new ArrayList<>();
        HashMap<String, Integer> nameToIdMap = new HashMap<>();

        for (File file : files) {
            String name = file.getName().split("_")[0];
            if (!nameToIdMap.containsKey(name)) {
                nameToIdMap.put(name, currentId);
                idToName.put(currentId, name);
                currentId++;
            }
            int label = nameToIdMap.get(name);
            images.add(Imgcodecs.imread(file.getAbsolutePath(), Imgcodecs.IMREAD_GRAYSCALE));
            labelsList.add(label);
        }

        MatOfInt labelsMat = new MatOfInt();
        labelsMat.fromList(labelsList);
        System.out.println("Training AI on " + images.size() + " images...");
        recognizer.train(images, labelsMat);
        return recognizer;
    }
}