#  Road Line Detection

<p align="center">
<img src = "examples/laneLines_thirdPass.jpg?raw=true" width="65%"/>
</p>

## Simple Description
Simple content based on the udacity self driving car nanodegree.  It will try to estimate the road lines over an image or a video, hardcoded vertices and using RGB color space but we know that are better color spaces to do the analyis.  Also another approach is to do perspective transformation of the road to get a better edge approximation.

This readme will be expanded the next week for better explanations.

## Testing
Just open jupyter notebook and test it.

For testing one image only as a script file you must be at the root folder and then: 

- python SDC\LaneLinesDetector.py

This will call the __main__ section that is found inside the class

For testing one video only as a script file you must be at the root folder and then:

- python LaneLines.py

This will help you to use the script as a stand alone class