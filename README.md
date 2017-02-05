**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./pics/undistortion.png "Undistorted"
[image2]: ./pics/undistort_image.jpg "Road Transformed"
[image3]: ./pics/white_line_detection.png "Binary Example of White Line"
[image4]: ./pics/yellow_line_detection.png "Binary Example of Yellow Line"
[image5]: ./pics/sobel_thresh.png "Sobel X"
[image6]: ./pics/color_edge.png "Color Edge"
[image7]: ./pics/gaussianblur.png "Gaussian Blur"
[image8]: ./pics/warped.png "Warped image"
[image9]: ./pics/left_right_lane.png "Left Right Lane"
[image10]: ./pics/polyfit.png "2nd order polynomial fit"
[image11]: ./pics/plot_back.png "Plot Back"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Image distortion occurs when a camera looks at 3D objects in the real world and transforms them into 2D image. This transformation isn't perfect(different size or shape). Therefore we have to undistort the image. By undistorting image, we can get correct and more useful information from an image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![Image undistortion][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one(Left:Original image Right:Undistorted image):
![alt text][image2]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)


First I get white line and yellow line by using RGB to HSV color transformation.
The threshold of the HSV is as follows.

| Color Space | White Line    | Yellow Line   |
|:---------:  |:-------------:|:-------------:|
| H low       | 0             | 0             |
| H high      | 255           | 80            |
| S low       | 0             | 65            |
| S high      | 32            | 255           |
| V low       | 180           | 80            |
| V high      | 255           | 255           |

The binary image of the white and yellow line are below.
![White Line][image3]
![Yellow Line][image4]

I also used Edge detection by Sobel x operator


```
sobel = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
```

The binary image of sobel x thresholding is below.
(I apply the sobel x to the original image,not transformed ones,so that we can see the entire image of sobel x effect.)
![Sobel X][image5]


Then I put it together.
![Color Edge][image6]

Finally I apply Gaussian Blur so that the detected lines area are enlarged.
![Gaussian Blur][image7]


####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in the 3rd code cell of the IPython notebook.  The `warp()` function takes as inputs an image (`img`) and returns the transformed image,transformation matrics and inverse transformation matrics. I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(0,img_size[1]),
      (img_size[0],img_size[1]),
      (0.6*img_size[0],2./3*img_size[1]),
      (0.4*img_size[0],2./3*img_size[1])]])
dst = np.float32(
    [[0,img_size[1]],
    [img_size[0],img_size[1]],
    [img_size[0],0],
    [0,0]])
```
This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 585, 460      | 320, 0        |
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Warped image][image8]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Until here, I've got binary image of lanes.Therefore, from here, I'll discuss how I identified lane line pixels and fit their positions with a polynomial.

First I divied the image into 10 frames from top to bottom.
Then, I take the moving average to the binary image of each frame. "1" means the pixel is white and "0" means the pixel is black. After calculating moving average, I set threthold to 0.005 empirically to decide where lane pixels are in the pics. I set 0 in the binary picture where the moving average is below threshold. Then I got the left and right lanes as follows.
![Left and Right Lanes][image9]

After getting left and right lanes, I set threshold again to each lane to find the better position. As for the left lane, I set threthold as 0.08 and for the right lane, 0.008. Thus I got the lane pixels. To prevent detecting anomally pixels(anomally location), I chose 5 to 90 percentile of the pixels from the left and 5 to 95 percentiles of the pixels from the right. Finally, I got the x and y lane pixels from the image, So from these points, I fit second order of polynomial fit.
![2nd order of polynomial fit][image10]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I've found lane lines so far. The lane lines are in the transformed image so I have to retransform it into the original image. I implemented this step in the EDA.ipynb




![Plot Back][image11]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.
