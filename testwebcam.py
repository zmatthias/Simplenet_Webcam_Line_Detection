import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 128)
cap.set(4, 72)

while(True):

    ret, webcamImage = cap.read()
    cv2.imshow('frame', webcamIqmage)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
