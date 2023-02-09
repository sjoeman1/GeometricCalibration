import numpy as np
import cv2 as cv
import glob

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
    images = glob.glob('images/chessImage?.jpg')
    # camera calibration for all images
    ret1, mtx1, dist1, rvecs1, tvecs1 = Offline(images)

    #TODO camera calibration for run 2 and 3

    #TODO online phase

def Offline(images):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((columns * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.


    for fname in images:
        print(fname)
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, board_shape, None)
        automatic = ret
        if not ret:
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


if __name__ == "__main__":
    main()
