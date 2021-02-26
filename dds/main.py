import os
import cv2
import dispenser
import people_counter


video = "C:/Users/jefqu/Desktop/3TI/afstudeerproject/code/QR-code_2/IMG_4270.MOV"
cap = cv2.VideoCapture("C:/Users/jefqu/Desktop/3TI/afstudeerproject/code/QR-code_2/IMG_4270.MOV")
_, frame1 = cap.read()
_, frame = cap.read()
cv2.imwrite("room_photo.jpg", cv2.resize(frame1, (480, 640)))

middelpunt, radius = dispenser.dispenser("room_photo.jpg")
print(middelpunt)
test_punt = (250, 405)

while test_punt[0] > 0 and test_punt[0] < 5000:
    distance = pow(pow((middelpunt[0] - test_punt[0]), 2) + pow((middelpunt[1] - test_punt[1]), 2), (1/2))
    if (distance <= radius):
        print(f"{distance}, interaction")
        break
    else:
        print(distance)
        test_punt = (test_punt[0] - 1, test_punt[1])
        print(test_punt)

# while True:
    # ok, frame2 = cap.read()
    # resize = cv2.resize(frame2, (480, 640))

    # human.detect(resize)
    # cv2.circle(resize, middelpunt, radius, (0, 255, 0), 5)

    # cv2.imshow("Test", resize)

    # press Q to end program
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #    print("I quit!")
    #    break
os.system('people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input C:/Users/jefqu/Desktop/3TI/afstudeerproject/code/video/counting_interactie.mp4')
cap.release()
# cv2.destroyAllWindows()