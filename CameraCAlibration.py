import numpy as np
import cv2 as cv
import glob
import os

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


columns = 6
rows = 7
board_shape = (columns, rows)




def click_event(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        params.append([x, y])
        if len(params) == 4:
            cv.destroyWindow('img')


def getChessboardCorners(img):
    cv.imshow("img", img)
    corners = []
    cv.setMouseCallback('img', click_event, param= corners)
    cv.waitKey(0)
    print(corners)
    return True, corners



def main():
    # images = glob.glob('images/chessImage*.jpg')
    # print(images)
    # # camera calibration for all images
    # calibration1 = Offline(images)

    images = glob.glob(f'{os.getcwd()}\\images\\chessImage*.png')
    # camera calibration for run 2
    calibration2 = Offline(images)

    images = glob.glob('images3\\chessImage*.png')
    # camera calibration for run 3
    calibration3 = Offline(images)

    #Online phase
    Online(images, calibration2)

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
        automatic = ret
        if not ret:
            print(fname)
            continue
            ret, corners = getChessboardCorners(gray)
            # TODO: linear interpolation of 4 corner points
        # If found, add object points, image points (after refining them)
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, board_shape, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
    cv.destroyAllWindows()
    return cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


def undistort(image, calibration):
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
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

    ret, mtx, dist, rvecs, tvecs = calibration

    #find corners
    if corners is None:
        ret, corners = cv.findChessboardCorners(img, (7, 6), None)

    if ret is not False:
        # find the rotation and translation vectors.
        ret, rvecs, tvecs = cv.solvePnP(objp, corners, mtx, dist)

        #project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)

        img = draw(img, corners, imgpts)
        cv.imshow('img', img)

def Online(images, calibration):
    vid = cv.VideoCapture(0)
    while True:
        ret_vid, frame = vid.read()
        if not ret_vid:
            print("could not find video input, exiting...")
            break
        generateImage(frame, calibration)

        key = cv.waitKey(1)
        if key % 256 == 27:
            print("aborting")
            break

    vid.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
