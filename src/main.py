import cv2
import os
 
camUser = 'admin'
camVerificationCode = 'QDIFHA'
camIp = '10.0.2.119'

RTSP_URL = f'rtsp://{camUser}:{camVerificationCode}@{camIp}:554/H.264'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
video = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

staticBack = None
while True:
    _, frame = video.read()
    motion = 0
    grayScale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    grayScale = cv2.GaussianBlur(grayScale, (21, 21), 0)

    if staticBack is None:
        staticBack = grayScale
        continue
 
    diff_frame = cv2.absdiff(staticBack, grayScale)
 
    thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)
 
    movingObjects,_ = cv2.findContours(thresh_frame.copy(),
                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    minPixelsMoving = 5000
    for contour in movingObjects:
        if cv2.contourArea(contour) < minPixelsMoving:
            continue
        motion = 1
 
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (170, 0, 100), 3)

    cv2.imshow("Kin Vision", frame)
 
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
 
video.release()
cv2.destroyAllWindows()