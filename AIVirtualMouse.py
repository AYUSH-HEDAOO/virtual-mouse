import numpy as np
import cv2
import autopy
import HandTrackingModule as htm
import time
import pyautogui

##########################
wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 7
##########################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# Initialize handwriting recognition variables
writing = False
previous_point = None
recognized_text = ""

try:
    cap = cv2.VideoCapture(0)
except:
    cap = cv2.VideoCapture(1)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
print(wScr, hScr)



# Load the pre-trained KNearest model for character recognition
knn = cv2.ml.KNearest_load("knearest_model.xml")

def recognize_handwriting(point):
    # Extract x and y coordinates from the point
    x, y = point
    
    # Assuming you have preprocessed the handwritten data and extracted features
    # Here, we'll use a dummy feature vector for demonstration
    feature_vector = np.array([[x, y]], dtype=np.float32)
    
    # Use the KNearest model to predict the character
    ret, result, neighbors, dist = knn.findNearest(feature_vector, k=3)
    
    # Convert the predicted label to character
    recognized_char = chr(int(result[0][0]))
    
    return recognized_char


while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # 2. Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # 4. Only Index Finger : Moving Mode
        if fingers[1] == 1 and fingers[2] == 0 and fingers[0] == 0:
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY
            # Check if handwriting mode is activated
            if writing:
                if previous_point is not None:
                    cv2.line(img, previous_point, (int(x1), int(y1)), (255, 0, 0), 5)
                previous_point = (int(x1), int(y1))
                recognized_text += recognize_handwriting(previous_point)

        # 8. Both Index and middle fingers are up : Clicking Mode
        if fingers[1] == 1 and fingers[2] == 1:
            length, img, lineInfo = detector.findDistance(8, 12, img)
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 10, (0, 255, 255), cv2.FILLED)
                autopy.mouse.click()

        # 9. Index and thumb fingers are up: Scrolling Mode
        if fingers[0] == 1 and fingers[1] == 1:
            lengthS, imgS, lineInfoS = detector.findDistance(4, 8, img)
            if lengthS < 40:
                cv2.circle(img, (lineInfoS[4], lineInfoS[5]), 10, (0, 255, 255), cv2.FILLED)
                pyautogui.vscroll(-150)
            if lengthS > 80:
                cv2.circle(img, (lineInfoS[4], lineInfoS[5]), 10, (0, 255, 255), cv2.FILLED)
                pyautogui.vscroll(150)

    # 11. Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 12. Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)

    