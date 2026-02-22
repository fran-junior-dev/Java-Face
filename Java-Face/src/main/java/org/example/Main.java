package org.example;
import nu.pattern.OpenCV;
import org.opencv.core.Core;

public class Main {
    public static void main(String[] args) {
        // Load the OpenCV library
        OpenCV.loadLocally();

        System.out.println("Welcome to OpenCV " + Core.VERSION);
        System.out.println("Let's detect some faces!");
    }
}