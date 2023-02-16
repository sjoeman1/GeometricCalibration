import numpy as np
import cv2 as cv
import glob
from scipy.interpolate import griddata
import os

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


columns = 6
rows = 7
board_shape = (columns, rows)

clicks = 0

def undistort(img, calibration):
    # undistort an image using a calibration
    ret, mtx, dist, rvecs, tvecs = calibration
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    return dst

def draw(img, corners, imgpts):
    # #draw a cube on the image given the corners and the projected points
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
    return img


def generateImage(img, calibration, corners = None):
    #overlay a cube over an image

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((columns * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    axis = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],
                       [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])

    ret, mtx, dist, rvecs, tvecs = calibration

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #find corners
    if corners is None:
        ret, corners = cv.findChessboardCorners(gray, board_shape, None)
        img = cv.drawChessboardCorners(img, board_shape, corners, ret)

    if ret is not False:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # find the rotation and translation vectors.
        ret, rvecs, tvecs = cv.solvePnP(objp, corners, mtx, dist)

        #project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)

        img = draw(img, corners2, imgpts)

    return img


def Online(images, calibration):
    #do the online part of the assignment
    #draws a cube on each chessboard image and displays it

    vid = cv.VideoCapture(0)
    while True:
        ret_vid, frame = vid.read()
        if not ret_vid:
            print("could not find video input, exiting...")
            break
        frame = generateImage(frame, calibration)
        cv.imshow('vid', frame)
        key = cv.waitKey(1)
        if key % 256 == 27:
            print("aborting")
            break

    vid.release()
    cv.destroyAllWindows()

def interpolateCorners(init_corners, image):
    # use a projective matrix to calculate all the corners in a grid


    #calculate the projective matrix
    input_pts = np.float32(init_corners)
    output_pts = np.float32([[0, 0], [500, 0], [500, 500], [0, 500]])
    M = cv.getPerspectiveTransform(output_pts, input_pts)

    # calculate coordinates of grid between 0 and 1 with length of rows and columns
    x = np.linspace(0, 500, columns)
    y = np.linspace(0, 500, rows)
    #combine x and y to get a grid of coordinates
    grid = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

    corners = []
    #use the matrix to transform the grid to coordinates on the image
    for point in grid:
        corner = cv.perspectiveTransform(np.array([[point]]), M)[0][0]
        corners.append(corner)

    #draw a circle for each corner
    #for corner in corners:
    #   cv.circle(image, (int(corner[0]), int(corner[1])), 5, (0, 0, 255), -1)
    #cv.imshow('corners', image)
    #cv.waitKey(1)
    return corners


def click_event(event, x, y, flags, params):
    global clicks
    corners = params[0]
    print(clicks)
    print(corners)
    if event == cv.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        corners[clicks] = [x, y]

        clicks += 1
        if clicks == 4:
            clicks = 0
            cv.destroyWindow('img Click corners')


def getChessboardCorners(img):
    cv.imshow('img Click corners', img)
    corners = np.zeros((4, 2))
    cv.setMouseCallback('img Click corners', click_event, param= (corners, clicks))
    cv.waitKey(0)
    print(corners)
    return True, corners

def Offline(images):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((columns * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    print(images)

    img = cv.imread(images[0])
    cv.imshow("img", img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, board_shape, None)
        if not ret:
            print(fname)
            ret, corners = getChessboardCorners(gray)
            corners = interpolateCorners(corners, img)
            print("BEEP")
            cv.drawChessboardCorners(img, board_shape, corners, ret)
            cv.imshow("img manual corners", img)
            print(corners)
            cv.waitKey(0)
        # If found, add object points, image points (after refining them)
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, board_shape, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(0)
    cv.destroyWindow('img')
    return cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
def main():
    image = cv.imread(f'{os.getcwd()}\\test_image\\chessImage157True.jpg')

    images = glob.glob(f'{os.getcwd()}\\images\\chessImage*.png')
    # print(images)
    # # camera calibration for all images
    calibration1 = Offline(images)
    img = generateImage(image, calibration1)
    cv.imshow("calibration 2", img)

    images = glob.glob(f'{os.getcwd()}\\images2\\chessImage*.png')
    # camera calibration for run 2
    calibration2 = Offline(images)
    img = generateImage(image, calibration2)
    cv.imshow("calibration 2", img)

    images = glob.glob(f'{os.getcwd()}\\images3\\chessImage*.png')
    # camera calibration for run 3
    calibration3 = Offline(images)
    img = generateImage(image, calibration3)
    cv.imshow("calibration 3", img)

    cv.waitKey(0)

    #Online phase
    Online(images, calibration2)

if __name__ == "__main__":
    main()
