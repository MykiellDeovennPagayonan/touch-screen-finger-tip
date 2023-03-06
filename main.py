import mediapipe as mp
import cv2
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

pt1 = None
pt2 = None
drawing = False
pyautogui.FAILSAFE = False

# Set up the hand tracking
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

#  landmarks representing the fingertips
fingertip_indexes = [8, 12, 16, 20]

# Define a function to move the mouse to a target position
def move_mouse(target_position, img_crop_width, img_crop_height):
    screen_width, screen_height = pyautogui.size()
    x, y = target_position[0] / img_crop_width, target_position[1] / img_crop_height
    x, y = x * screen_width, y * screen_height
    pyautogui.moveTo(x, y)


# Define a function to process the output of the hand tracking solution
def process_hands(image):
    # Convert the image to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run the hand tracking solution on the image
    results = hands.process(image)

    # If hands were detected
    if results.multi_hand_landmarks:
        # For each detected hand
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the coordinates of  index
            index_finger = hand_landmarks.landmark[8]
            x, y = int(index_finger.x * image.shape[1]), int(index_finger.y * image.shape[0])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

            # Move the mouse to the position of the index finger
            if pt1 and pt2:
                target_position = (x - pt1[0], y - pt1[1])
                img_crop_width = pt2[0] - pt1[0]
                img_crop_height = pt2[1] - pt1[1]
                if target_position[0] > 0 and target_position[1] > 0 and target_position[0] < img_crop_width and target_position[1] < img_crop_height:
                    move_mouse(target_position, img_crop_width, img_crop_height)


    # Convert the image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image

# Define the callback function for mouse events

def draw_rectangle(event, x, y, flags, params):
    global pt1, pt2, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing the rectangle
        pt1 = (x, y)
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        # If drawing, update the rectangle
        if drawing:
            pt2 = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        # Stop drawing the rectangle
        pt2 = (x, y)
        drawing = False

# Create a named window and set the mouse callback
cv2.namedWindow('frame')
cv2.setMouseCallback('frame', draw_rectangle)

# you guys know this already
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # If the user has defined a rectangle, crop the frame and show the cropped image
    if pt1 and pt2:
        img_crop = frame[min(pt1[1], pt2[1]):max(pt1[1], pt2[1]), min(pt1[0], pt2[0]):max(pt1[0], pt2[0])]
        cv2.imshow('WhiteBoard', img_crop)

    frame = process_hands(frame)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

