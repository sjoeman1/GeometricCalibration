import numpy as np
import cv2 as cv
import glob

vid = cv.VideoCapture(0)

def main():
    i = 0
    while True :
        ret, frame = vid.read()
        if not ret:
            continue
        cv.imshow('img', frame)
        key = cv.waitKey(1)
        if key%256 == 27:
            print("aborting")
            break
        elif key%256 == 32:
            print(f"writing image {i}")
            cv.imwrite(f"images\\chessImage{i}.jpg", frame)
            i += 1

    vid.release()
    cv.destroyAllWindows()




if __name__ == "__main__":
    main()