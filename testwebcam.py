import cv2

cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

while(True):

    ret, webcamImage = cap.read()
    cv2.imshow('frame', webcamImage)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
