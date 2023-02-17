import time

import numpy as np
import cv2 as cv
import glob

vid = cv.VideoCapture(0)
columns = 6
rows = 9
board_shape = (columns, rows)

def main():
    i = 0
    trues = 0
    false = 0
    print("get 25 images, 5 which do not automatically detect corners.")
    print("Take a final test image that has the chessboard tilted significantly close to the image border. Corners have to be found automatically")
    print("press space to take picture, esc to exit")
    while True :
        ret_vid, frame = vid.read()
        if not ret_vid:
            print("could not find video input, exiting...")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, board_shape, flags=cv.CALIB_CB_FAST_CHECK)
        print(ret)
        # frameCorner = cv.drawChessboardCorners(frame, board_shape, corners, ret)
        # cv.imshow('img', frameCorner)
        cv.imshow('img without conrer', frame)
        key = cv.waitKey(1)

        if key%256 == 27:
            print("aborting")
            break
        elif key%256 == 32:
            print(f"writing image {i}")
            if(ret):
                trues +=1
            else:
                false +=1
            print(f"Able to find corners: {ret}, trues: {trues}, false {false}")
            cv.imwrite(f"test_image\\testImage{ret}.png", frame)
            i += 1

    vid.release()
    cv.destroyAllWindows()




if __name__ == "__main__":
    main()