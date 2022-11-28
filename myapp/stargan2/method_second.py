#!/usr/bin/env python
# coding: utf-8

# In[9]:


import cv2
import sys
import dlib
import time
import math
import numpy as np
import matplotlib.pyplot as plt

# import the necessary packages
from imutils import face_utils
import dlib
import cv2

# In[103]:


import cv2
import dlib
import numpy as np
import math
import random


# Returns 8 points on the boundary of a rectangle
def getEightBoundaryPoints(h, w):
    boundaryPts = []
    boundaryPts.append((0, 0))
    boundaryPts.append((w / 2, 0))
    boundaryPts.append((w - 1, 0))
    boundaryPts.append((w - 1, h / 2))
    boundaryPts.append((w - 1, h - 1))
    boundaryPts.append((w / 2, h - 1))
    boundaryPts.append((0, h - 1))
    boundaryPts.append((0, h / 2))
    return np.array(boundaryPts, dtype=np.float)


# Constrains points to be inside boundary
def constrainPoint(p, w, h):
    p = (min(max(p[0], 0), w - 1), min(max(p[1], 0), h - 1))
    return p


# convert Dlib shape detector object to list of tuples
def dlibLandmarksToPoints(shape):
    points = []
    for p in shape.parts():
        pt = (p.x, p.y)
        points.append(pt)
    return points


# Compute similarity transform given two sets of two points.
# OpenCV requires 3 pairs of corresponding points.
# We are faking the third one.
def similarityTransform(inPoints, outPoints):
    s60 = math.sin(60 * math.pi / 180)
    c60 = math.cos(60 * math.pi / 180)

    inPts = np.copy(inPoints).tolist()
    outPts = np.copy(outPoints).tolist()

    # The third point is calculated so that the three points make an equilateral triangle
    xin = c60 * (inPts[0][0] - inPts[1][0]) - s60 * (inPts[0][1] - inPts[1][1]) + inPts[1][0]
    yin = s60 * (inPts[0][0] - inPts[1][0]) + c60 * (inPts[0][1] - inPts[1][1]) + inPts[1][1]

    inPts.append([np.int(xin), np.int(yin)])

    xout = c60 * (outPts[0][0] - outPts[1][0]) - s60 * (outPts[0][1] - outPts[1][1]) + outPts[1][0]
    yout = s60 * (outPts[0][0] - outPts[1][0]) + c60 * (outPts[0][1] - outPts[1][1]) + outPts[1][1]

    outPts.append([np.int(xout), np.int(yout)])

    # Now we can use estimateRigidTransform for calculating the similarity transform.
    tform = cv2.estimateRigidTransform(np.array([inPts]), np.array([outPts]), False)
    return tform


# Normalizes a facial image to a standard size given by outSize.
# Normalization is done based on Dlib's landmark points passed as pointsIn
# After normalization, left corner of the left eye is at (0.3 * w, h/3 )
# and right corner of the right eye is at ( 0.7 * w, h / 3) where w and h
# are the width and height of outSize.
def normalizeImagesAndLandmarks(outSize, imIn, pointsIn):
    h, w = outSize

    # Corners of the eye in input image
    eyecornerSrc = [pointsIn[36], pointsIn[45]]

    # Corners of the eye in normalized image
    eyecornerDst = [(np.int(0.3 * w), np.int(h / 3)),
                    (np.int(0.7 * w), np.int(h / 3))]

    # Calculate similarity transform
    tform = similarityTransform(eyecornerSrc, eyecornerDst)
    imOut = np.zeros(imIn.shape, dtype=imIn.dtype)

    # Apply similarity transform to input image
    imOut = cv2.warpAffine(imIn, tform, (w, h))

    # reshape pointsIn from numLandmarks x 2 to numLandmarks x 1 x 2
    points2 = np.reshape(pointsIn, (pointsIn.shape[0], 1, pointsIn.shape[1]))

    # Apply similarity transform to landmarks
    pointsOut = cv2.transform(points2, tform)

    # reshape pointsOut to numLandmarks x 2
    pointsOut = np.reshape(pointsOut, (pointsIn.shape[0], pointsIn.shape[1]))

    return imOut, pointsOut


# find the point closest to an array of points
# pointsArray is a Nx2 and point is 1x2 ndarray
def findIndex(pointsArray, point):
    dist = np.linalg.norm(pointsArray - point, axis=1)
    minIndex = np.argmin(dist)
    return minIndex


# Check if a point is inside a rectangle
def rectContains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


# Calculate Delaunay triangles for set of points
# Returns the vector of indices of 3 points for each triangle
def calculateDelaunayTriangles(rect, points):
    # Create an instance of Subdiv2D
    subdiv = cv2.Subdiv2D(rect)

    # Insert points into subdiv
    for p in points:
        subdiv.insert((p[0], p[1]))

    # Get Delaunay triangulation
    triangleList = subdiv.getTriangleList()

    # Find the indices of triangles in the points array
    delaunayTri = []

    for t in triangleList:
        # The triangle returned by getTriangleList is
        # a list of 6 coordinates of the 3 points in
        # x1, y1, x2, y2, x3, y3 format.
        # Store triangle as a list of three points
        pt = []
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            # Variable to store a triangle as indices from list of points
            ind = []
            # Find the index of each vertex in the points list
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if (abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)
                # Store triangulation as a list of indices
            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))

    return delaunayTri


# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size):
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None,
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2):
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)

    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
                (1.0, 1.0, 1.0) - mask)
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2Rect


# detect facial landmarks in image
def getLandmarks(faceDetector, landmarkDetector, im, FACE_DOWNSAMPLE_RATIO=1):
    points = []
    imSmall = cv2.resize(im, None,
                         fx=1.0 / FACE_DOWNSAMPLE_RATIO,
                         fy=1.0 / FACE_DOWNSAMPLE_RATIO,
                         interpolation=cv2.INTER_LINEAR)

    faceRects = faceDetector(imSmall, 0)

    if len(faceRects) > 0:
        maxArea = 0
        maxRect = None
        # TODO: test on images with multiple faces
        for face in faceRects:
            if face.area() > maxArea:
                maxArea = face.area()
                maxRect = [face.left(),
                           face.top(),
                           face.right(),
                           face.bottom()
                           ]

        rect = dlib.rectangle(*maxRect)
        scaledRect = dlib.rectangle(int(rect.left() * FACE_DOWNSAMPLE_RATIO),
                                    int(rect.top() * FACE_DOWNSAMPLE_RATIO),
                                    int(rect.right() * FACE_DOWNSAMPLE_RATIO),
                                    int(rect.bottom() * FACE_DOWNSAMPLE_RATIO))

        landmarks = landmarkDetector(im, scaledRect)
        points = dlibLandmarksToPoints(landmarks)
    return points


# Warps an image in a piecewise affine manner.
# The warp is defined by the movement of landmark points specified by pointsIn
# to a new location specified by pointsOut. The triangulation beween points is specified
# by their indices in delaunayTri.
def warpImage(imIn, pointsIn, pointsOut, delaunayTri):
    h, w, ch = imIn.shape
    # Output image
    imOut = np.zeros(imIn.shape, dtype=imIn.dtype)

    # Warp each input triangle to output triangle.
    # The triangulation is specified by delaunayTri
    for j in range(0, len(delaunayTri)):
        # Input and output points corresponding to jth triangle
        tin = []
        tout = []

        for k in range(0, 3):
            # Extract a vertex of input triangle
            pIn = pointsIn[delaunayTri[j][k]]
            # Make sure the vertex is inside the image.
            pIn = constrainPoint(pIn, w, h)

            # Extract a vertex of the output triangle
            pOut = pointsOut[delaunayTri[j][k]]
            # Make sure the vertex is inside the image.
            pOut = constrainPoint(pOut, w, h)

            # Push the input vertex into input triangle
            tin.append(pIn)
            # Push the output vertex into output triangle
            tout.append(pOut)

        # Warp pixels inside input triangle to output triangle.
        warpTriangle(imIn, imOut, tin, tout)
    return imOut


# In[148]:


def getLipsMask(size, lips):
    # Find Convex hull of all points
    hullIndex = cv2.convexHull(np.array(lips), returnPoints=False)
    # Convert hull index to list of points
    hullInt = []
    for hIndex in hullIndex:
        hullInt.append(lips[hIndex[0]])
    # Create mask such that convex hull is white
    mask = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(hullInt), (255, 255, 255))
    return mask


def alphaBlend(alpha, foreground, background):
    fore = np.zeros(foreground.shape, dtype=foreground.dtype)
    fore = cv2.multiply(alpha, foreground, fore, 1 / 255.0)
    alphaPrime = np.ones(alpha.shape, dtype=alpha.dtype) * 255 - alpha
    back = np.zeros(background.shape, dtype=background.dtype)
    back = cv2.multiply(alphaPrime, background, back, 1 / 255.0)
    outImage = cv2.add(fore, back)
    return outImage


def get_angle(p1, p2):
    if p2[1] <= p1[1]:
        y = np.abs(p1[1] - p2[1])
    else:
        y = p1[1] - p2[1]
    x = np.abs(p1[0] - p2[0])
    return np.rad2deg(np.arctan2(y, x))


# In[149]:


#
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor

# Landmark model location
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# Get the face detector
faceDetector = dlib.get_frontal_face_detector()
# The landmark detector is implemented in the shape_predictor class
landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)


# # Give image path

# In[164]:
def main(image_path, color_code):
    im = cv2.imread(image_path)

    imDlib = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    points = getLandmarks(faceDetector, landmarkDetector, imDlib)
    print(points)

    # In[165]:

    lipstick_colors = {"vamptastic_plum": (103, 92, 223)
                       }
    # lipstick_colors_ = color_code      #[10, 50, 250]       # [10, 250, 10]

    # In[166]:

    ## Select points of lips and create "lips mask"
    lips = [points[x] for x in range(48, 68)]
    mouth = [points[x] for x in range(60, 68)]
    clone_lips = im.copy()
    for point in lips:
        cv2.circle(clone_lips, point, 4, (0, 255, 0), -1)
    for point in mouth:
        cv2.circle(clone_lips, point, 4, (255, 0, 0), -1)

    # In[167]:

    contours = [np.asarray(lips, dtype=np.int32)]
    (x, y, w, h) = cv2.boundingRect(contours[0])
    center = (int(x + w / 2), int(y + h / 2))
    mask = getLipsMask(im.shape, lips)
    mouth_mask = getLipsMask(im.shape, mouth)
    mouth_mask = cv2.bitwise_not(mouth_mask)
    # mask = cv2.bitwise_and(mask, mask, mask=mouth_mask[:, :, 0])

    ## Dilate lips mask to include some skin around the mouth
    maskHeight, maskWidth = mask.shape[0:2]
    maskSmall = cv2.resize(mask, (600, int(maskHeight * 600.0 / maskWidth)))
    maskSmall = cv2.dilate(maskSmall, (3, 3))
    maskSmall = cv2.GaussianBlur(maskSmall, (5, 5), 0, 0)
    mask = cv2.resize(maskSmall, (maskWidth, maskHeight))
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    # plt.imshow(mask_rgb)
    # plt.show()
    # def apply_color_to_mask(mask):
    #     # Get random lipstick color
    #     color_name, color = random.choice(list(lipstick_colors.items()))
    #     print(color)
    #     print(type(color))
    #     color = (20, 52, 45)
    #     # print(color)
    #
    #     print('llllllllllllllllllllllllllllllllllllllllllllllllllll')
    #     print("[INFO] Color Name: {}".format(color_name))
    #     b, g, r = cv2.split(mask)
    #     b = np.where(b > 0, color[0], 0).astype('uint8')
    #     g = np.where(g > 0, color[1], 0).astype('uint8')
    #     r = np.where(r > 0, color[2], 0).astype('uint8')
    #     return cv2.merge((b, g, r)), color_name

    # In[168]:

    ## Apply color to mask
    # color_mask, color_name = apply_color_to_mask(mask)

    color_mask = np.zeros_like(mask)

    masked_lips = cv2.bitwise_and(im, im, mask=mask[:, :, 0])
    b, g, r = color_code      #[10, 50, 250]       # [10, 250, 10]
    color_mask[:,:,0]=(masked_lips[:,:,0]/255)*b
    color_mask[:,:,1]=(masked_lips[:,:,1]/255)*g
    color_mask[:,:,2]=(masked_lips[:,:,2]/255)*r

    ## Seamless cloning the mask with the lips. In this case MIXED_CLONE allow to maintain the lips texture
    # masked_lips = cv2.bitwise_and(im, im, mask=mask[:, :, 0])
    output = cv2.seamlessClone(masked_lips, color_mask, mask[:, :, 0], center, cv2.MIXED_CLONE)
    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    plt.imshow(output_rgb)
    plt.show()

    # In[170]:

    ## Alpha Blending
    from PIL import Image

    final = alphaBlend(mask, output, im)
    # cv2.putText(final, "Lipstick Color: {}".format(color_name), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    final_image = np.hstack((im, final))
    final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)

    return final_image


image_path = 'imgs/66.jpg'
color_code = [255, 0, 255]
image_ = main(image_path, color_code)
from matplotlib import pyplot as plt

plt.imshow(image_)
plt.show()

