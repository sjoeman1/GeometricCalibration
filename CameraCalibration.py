import numpy as np
import cv2 as cv
import glob
from scipy.interpolate import griddata
import os

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#chessboard parameters
columns = 6
rows = 9
board_shape = (columns, rows)
cube_size = 22

# prepare object points with cube size, like (0,0,0), (22,0,0), (44,0,0) ....,(132,110,0)
objp = np.zeros((columns * rows, 3), np.float32)
objp[:, :2] = np.mgrid[0:columns * cube_size:cube_size, 0:rows * cube_size:cube_size].T.reshape(-1, 2)

#amount of clicks for the corners
clicks = 0


# #draw a cube on the image given the corners and the projected points
# project 3D points to image plane
def draw_cube(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    #first the x and y origin axis lines
    img = cv.line(img, tuple(imgpts[0]), tuple(imgpts[1]), (255,255,0), 10)
    img = cv.line(img, tuple(imgpts[0]), tuple(imgpts[3]), (0,255,255), 10)

    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255,0,0), 6)

    # then draw z origin axis lines
    img = cv.line(img, tuple(imgpts[0]), tuple(imgpts[4]), (255,0,255), 10)

    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img


# generates a new image from img, with cornerpoints and a cube projected on to it
def Online(img, calibration, corners = None):

    # the 3d points of a cube. 2 * cube_size to make the cube 2 chessboard squares sized
    length = 2 * cube_size
    cube = np.float32([[0, 0, 0], [0, length, 0], [length, length, 0], [length, 0, 0],
                       [0, 0, -length], [0, length, -2 * cube_size], [length, length, -length], [length, 0, -length]])

    # the camera calibration matrices
    ret, mtx, dist, rvecs, tvecs = calibration

    #gray the image
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #find corners
    if corners is None:
        ret, corners = cv.findChessboardCorners(gray, board_shape, flags= cv.CALIB_CB_FAST_CHECK)
        img = cv.drawChessboardCorners(img, board_shape, corners, ret)

    if ret is not False:
        # refine corners
        corners2 = cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

        # find the rotation and translation vectors.
        ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)

        #project the real cube coordinates to image coordinates
        imgpts, jac = cv.projectPoints(cube, rvecs, tvecs, mtx, dist)

        #draw the cube using the coordinates
        img = draw_cube(img, imgpts)

    return img


# draws a cube using the online function, but using the webcam
# using the different calibrations
def realTimeOnline(calibration1, calibration2, calibration3):

    #get the webcam
    vid = cv.VideoCapture(0)
    while True:
        ret_vid, frame = vid.read()
        if not ret_vid:
            print("could not find video input, exiting...")
            break

        # show the videos of the different calibrations
        frame = Online(frame, calibration1)
        cv.imshow('vid calibration1', frame)
        frame = Online(frame, calibration2)
        cv.imshow('vid calibration2', frame)
        frame = Online(frame, calibration3)
        cv.imshow('vid calibration3', frame)

        # abort if escape is pressed
        key = cv.waitKey(1)
        if key % 256 == 27:
            print("aborting")
            break

    vid.release()
    cv.destroyAllWindows()

# use a projective matrix to calculate all the corners in a grid
# this way both the x and y coordinates are taken into account
def interpolateCorners(init_corners, image):

    #calculate the projective matrix
    input_pts = np.float32(init_corners)
    output_pts = np.float32([[500, 500], [500, 0], [0, 0], [0, 500]])
    M = cv.getPerspectiveTransform(output_pts, input_pts)

    # calculate coordinates of grid between 0 and 500 with length of rows and columns, this number does not matter as it is only the size of the unit matrix
    x = np.linspace(0, 500, rows)
    y = np.linspace(0, 500, columns)
    #combine x and y to get a grid of coordinates
    grid = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

    corners = []
    #use the matrix to transform the grid to coordinates on the image
    for point in grid:
        corner = cv.perspectiveTransform(np.array([[point]], dtype=np.float32), M)[0]
        corners.append(corner)

    corners = np.array(corners, ndmin=3, dtype=np.float32)
    return corners


# click event handler
# saves the x and y coordinates of 4 clicks
# need to in order: top left, clockwise
def click_event(event, x, y, flags, params):
    global clicks
    corners = params[0]
    if event == cv.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        corners[clicks] = [x, y]

        clicks += 1
        if clicks == 4:
            clicks = 0
            cv.destroyWindow('img Click corners')


# shows the image where corners need to be manually acquired
def getChessboardCorners(img):
    cv.imshow('img Click corners', img)
    corners = np.zeros((4, 2))
    cv.setMouseCallback('img Click corners', click_event, param= (corners, clicks))
    cv.waitKey(0)
    return True, corners


# collect all the cornerpoints of the images in images
# when corners cant be detected automatically, acquire them manually
# also refines the corner positions of both automatic and manual corners
def Offline(images):

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # get the image and gray them out
    img = cv.imread(images[0])
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners automatically
        ret, corners = cv.findChessboardCorners(gray, board_shape, None)
        if not ret:
            #find chessboard corners manually
            ret, corners = getChessboardCorners(gray)
            corners = interpolateCorners(corners, img)

        # refine the corner positions
        corners = cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

        #check quality of image

        error = reprojectionError(objp, corners, gray)
        print(error)
        # If found to be within the error threshold, add object points, image points (after refining them)
        if error < 0.02:
            objpoints.append(objp)
            imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, board_shape, corners, ret)
        cv.imshow('img', img)
        cv.waitKey(0)

    cv.destroyWindow('img')

    #calibrate the camera using all the points found in all the images
    calibration = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return calibration

# calculate the reprojection error by projecting the 3d points to 2d points and measuring the average distance
def reprojectionError(objectpoints, corners, image):
    mean_error = 0
    objp = []
    imgp = []
    objp.append(objectpoints)
    imgp.append(corners)
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objp, imgp, image.shape[::-1], None, None)

    for i in range(len(objp)):
        imgpoints2, _ = cv.projectPoints(objp[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgp[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error

    return mean_error/len(objp)

# calibrate the camerea using 3 different runs of images
# then display the test image with a cube
# then open camera to real time display cube
def main():
    image = cv.imread(f'{os.getcwd()}\\test_image\\testImageTrue.png')

    images = glob.glob(f'images\\chessImage*.png')
    # camera calibration for all images
    calibration1 = Offline(images)
    img = Online(image, calibration1)
    cv.imshow("calibration 1", img)
    cv.imwrite("cubeCalibration1.png", img)

    images = glob.glob(f'images2\\chessImage*.png')
    # camera calibration for run 2
    calibration2 = Offline(images)
    img = Online(image, calibration2)
    cv.imshow("calibration 2", img)
    cv.imwrite("cubeCalibration2.png", img)

    images = glob.glob(f'images3\\chessImage*.png')
    # camera calibration for run 3
    calibration3 = Offline(images)
    img = Online(image, calibration3)
    cv.imshow("calibration 3", img)
    cv.imwrite("cubeCalibration3.png", img)

    cv.waitKey(0)

    #Online phase
    realTimeOnline(calibration1, calibration2, calibration3)

if __name__ == "__main__":
    main()
