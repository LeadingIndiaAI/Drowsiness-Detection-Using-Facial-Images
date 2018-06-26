# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2



def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    #Compute eye aspect ratio
    ear = (A+B)/(2*C)

    return ear

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2],mouth[10])
    B = dist.euclidean(mouth[3],mouth[9])
    C = dist.euclidean(mouth[4],mouth[8])

    mar = (A+B+C)/3

    return mar


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.3 #threshold for blink
EYE_AR_CONSEC_FRAMES = 10 #consecutive considered true
sleep_flag = -1

counter = 0
total = 0

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

print("[INFO] starting video stream thread...")
vs = FileVideoStream(args["video"]).start()
fileStream = True
vs = VideoStream(src=0).start()
time.sleep(1.0)
start_time = time.time()
elapsed_time = start_time

while True:
    # if fileStream and not vs.more():
    #     break

    frame = vs.read()
    frame = imutils.resize(frame, width = 450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray,0)    #dlib’s built-in face detector.

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        mouth = shape[mStart: mEnd]
        mouthEAR = mouth_aspect_ratio(mouth)

        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        if mouthEAR > 25:
            print("You are yawning")
        # else:
        #     print("You are working")
        if ear <  EYE_AR_THRESH:
            counter += 1

            if counter >= EYE_AR_CONSEC_FRAMES:
                if sleep_flag < 0:
                    print("You are sleeping")
                    sleep_flag = 1
            # if total > 0:
            #     elapsed_time  = time.time() - start_time
            #     if elapsed_time > 10:
            #         print("You are sleeping dear")
            #         start_time = time.time()
            #     else:
            #         print("You are working")

        else:
            counter = 0

        cv2.putText(frame, "Blinks: {}".format(total), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
