# Capture the mouse click events in Python and OpenCV
'''
-> draw shape on any image
-> reset shape on selection
-> crop the selection
run the code : python capture_events.py --image image_example.jpg
'''


# import the necessary packages
import cv2

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
ref_point = [(0,0),(0,0)]
drawing = False
clone = None
image = None


# load the image, clone it, and setup the mouse callback function
def dispenser(img):
    global clone , image
    def shape_selection(event, x, y, flags, param):
        # grab references to the global variables
        global ref_point, drawing, clone, image

        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            ref_point[0] = (x,y)
            drawing = True

        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # the cropping operation is finished
            ref_point[1] = (x, y)
            drawing = False

            # draw a rectangle around the region of interest
            cv2.circle(image, ref_point[0], 0, (0, 255, 0), 2)

            cv2.circle(image, ref_point[0], int(pow(pow((ref_point[0][0] - ref_point[1][0]), 2) + pow((ref_point[0][1] - ref_point[1][1]), 2), (1/2))), (0, 255, 0), 2)
            cv2.imshow("image", image)
        elif drawing == True and event == cv2.EVENT_MOUSEMOVE:
            image = clone.copy()
            ref_point[1] = (x, y)
            cv2.circle(image, ref_point[0], int(pow(pow((ref_point[0][0] - ref_point[1][0]), 2) + pow((ref_point[0][1] - ref_point[1][1]), 2), (1/2))), (0, 255, 0), 2)
            

    image = cv2.imread(img)
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", shape_selection)

# keep looping until the 'q' key is pressed
    while True:
    # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            image = clone.copy()

        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break

    # if there are two reference points, then crop the region of interest
    # from the image and display it
    if len(ref_point) == 2:
        # crop_img = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
        selected_image = image.copy()

        cv2.circle(image, ref_point[0], 0, (0, 255, 0), 5)
        cv2.imshow("Selected image", selected_image)
        # cv2.imwrite("dispenser.jpg", crop_img)
        cv2.waitKey(0)
    # close all open windows
    cv2.destroyAllWindows()
    distance = int(pow(pow((ref_point[0][0] - ref_point[1][0]), 2) + pow((ref_point[0][1] - ref_point[1][1]), 2), (1/2)))
    return ref_point[0], distance