# Algo-Trading-Math-Models
## Math Techniques viz. ARIMA, Frequency Decomposition, Fourier Filtering, Linear Regression &amp;  Bi-directional LSTMs on Feature Engineered Stock Market Data.


**Quant Trading**
Quant strategies follow a data-driven approach to pick stocks. This approach  which seeks to reduce the role of human bias conceptually fall in between active and passive trading. The stock data is a classic example of " time series" where the prices are sampled at regular intervals.

![alt text](summaryImg/hough.jpg)

**Method 1: Double Peaks**
<br>

This **will not work if the text is in CAPITAL letters or in some other language**, as the "double peak logic" would likely falter.<br>
Another numerical way to address the problem is to make use of the font shape, such as **'Water Fill Technique'** or to mathematically represent the character shape, as given below. **We can describe any shape mathematically using shape context and log-bin histograms. **

**Method 2: Shape Contexts using Log-Bin Histograms**
1. Find text bounding boxes from images using **EAST**. {below}
2. Crop image inside bounding box and apply **Canny edge detection.**
3. Take a dummy image with alphanumeric as base input. **Find bounding boxes around each character** in base input and image from step (b). Do steps {d}-{h} to find best correspondence between character pairs.
4. **Randomly sample N points** from edge elements of each character shape.
5. **Construct a new shape descriptor - shape context. The shape context at a point captures the distribution over relative positions of other shape points and thus summarizes global shape.**

![alt text](summaryImg/shape_context.jpg) <br><br>
![alt text](summaryImg/shape_context_A.jpg)

6. Compare the **log-polar histograms using Pearson's chi-squared test or cosine distance.**
7. Find the numeral with minimum distance for each bounding box in base image. **Sum up the cost values of each bounding box to find Sigma( Φ).**
<br><br>![alt text](summaryImg/text_compare.jpg)
<br>
7. Invert cropped image from step (b) and do steps {d}-{h} to **compute Sigma( Φ'). **Compare the Sigma values to know text inversion.

<br>![alt text](summaryImg/invertedText_output.jpg)

**EAST (An Efficient and Accurate Scene Text Detector)**

The textual content inside an image can be localized using EAST algorithm.

<br>![alt text](summaryImg/east.gif)

Here again, we can use a math-hack to localize text in an image, instead of using AI-based EAST algorithm. **You can find consecutive local minima of y-projections of pixels to find consecutive trough that corresponds to line separation in an image.** Once a line is found, you can run method 2, starting from step (b).

The above method would work irrespective of font case or language.

**Skew Correction**

Most of the scanned documents are skewed. Thus, it is required to de-skew the image before feeding an OCR or even to display.

**Method 1: Iterative Projection**

1. Rotate the image from -10 degrees to +10 degrees.
2. Compute projection of all pixels on y-axis.
3. Calculate the pixel incidence density.
4. Step the rotation angle by 0.5 angles and repeat steps 2, 3
5. Find the angle Θ with maximum pixel incidence density.<br><br>
The drawbacks of the above algorithm are:<br>
<br> a. Iterative computation increases time complexity.
<br> b. Potential error of 0.5 degrees due to step size.<br>
<br>Mostly, scanned document would be of form format or tabular data containing lines or point spread of lines (lines can be disjoint in scanned image, due to lack of scan or print quality). Hence, the question boils down to "whether we can compute the line and Θ, given a point spread as input?"

**Method 2: Hough Transform Peak**

1. Read the skewed image and do Canny Edge detection
2. Hough Space = **Call Hough_Transform (Edge Detected Image)**
3. Find the **maxima in Hough space transform** (accumulator matrix)
4. **Find Θ of the significant lines using tangent of slope**
5. Calculate **median of slopes, Θ'**
6. Rotate the image by Θ'

<br>![alt text](summaryImg/hough.jpg)
![alt text](summaryImg/hough_detected.jpg)
![alt text](summaryImg/paul_receipt_skew.jpg)


<p style="float:center;">
*Skew Correction Functional Workflow*
</p>

<br>![alt text](summaryImg/hough_peak.jpg)

**Rotation Classification**

Rotation is a common problem in scanned images. The document can be rotated 90° or more, while being scanned.

You can use the above skew correction code to find Θ and rotate. The only drawback is, rotation of 90+Θ could be detected as 90-Θ, and -90-Θ as -90+Θ. Hence, the image can get flipped, once you rotate!

**To solve the above problem, just pass the de-skewed image to the text inversion code and flip it upright, if deemed necessary.**

**Homography**

Let's say you want to find an object (template) inside a bigger image with multiple objects. We can use Object detection models like SSD or YOLO with annotated Query Images to train different classes of objects to be found. But how do we use simple math to find and locate an object in a bigger image? <br>

**We can use homography to find point correspondences and transform the coordinates from one perspective to another. Homography is a transformation ( 3×3 matrix ) that maps the points in one image to the corresponding points in the other image.**

These are the steps you can follow.

1. Firstly, open the **template image and the image to be matched.**
2. Find all features from both input images.
3. Create an **ORB keypoint detector** which is less compute intensive than SIFT and SURF.
4. Find the key points and their descriptors with the orb detector.
5. **Create matches of descriptors, then sort them based on distances.**
6. Use cv2.drawMatchesKnn to draw all the k best matches.
7. **Extract the matched keypoints** from both images.
8. **Find homography matrix and do perspective transform**

![alt text](summaryImg/homography_box.jpg)<br>
![alt text](summaryImg/homography_custom.jpg)

**Object Search**

Let's say, you need to find an object from a set of images. You can use an AI model, as it is a classic case of image classification. But, can we use traditional math to do this? Here's how…

1. Read image of the object to search (Query Image)
2. Do Canny edge detection and find bounding  box around contour.
3. **Randomly sample 'n' random points** to describe the shape inside image.
4. Iterate and get all images inside the input folder.
5. Do steps 2 & 3 on each image.
6. **Compute the correlation value of random shape points of 'Query Image' with shape points of each image in the folder. **
7. **Find the image with minimum correlation value.** This image contains the nearest match of the object you are searching for.

![alt text](summaryImg/correlation.jpg)

**Above equation conceptually formulates correlation as the  similarity in deviation around mean. Thus, numerator signifies distribution similarity and denominator quantifies L2-norm for normalization.**

*Input Images and Compare Value*

![alt text](summaryImg/object_images.jpg)
![alt text](summaryImg/obj_search_output.png)

Please note that a different car (purple) with similar shape has the second nearest match value, right after the red car. The correlation distance to other shapes are distinctively more. Thus you can see shape matching is functional.

**Please note the correlation values will not be  0, even for same images, as random sampling of points is done to describe shapes. There are other ways to describe shapes without random sampling but time complexity of shape matching would become an order higher. One such method, known as Turning Function**, is depicted below.

![alt text](summaryImg/turningFn.jpg)
![alt text](summaryImg/turning_compare.jpg)

**References**
<br>
<br> [1] *Inversion Detection in Text Document Images. Hamid Pilevar, A. G. Ramakrishnan, Medical Intelligence and Language Engineering Lab, Department of Electrical Engineering, Indian Institute of Science, Bangalore (JCIS 2006)*<br>
<br> [2] *Shape Context: A new descriptor for shape matching and object recognition. Serge Belongie, Jitendra Malik and Jan Puzicha. Department of Electrical Engineering and Computer Sciences, University of California at Berkeley (NIPS 2000)*<br>
<br> [3] *Shape Matching and Object Recognition Using Shape Contexts. Serge Belongie, Jitendra Malik and Jan Puzicha. Computer Science Division, University of California at Berkeley (PAMI 2002)*
