import cv2
import mediapipe
import pyautogui

pyautogui.PAUSE = 0  # Remove PyAutoGUI delay

capture_hands = mediapipe.solutions.hands.Hands(
    max_num_hands=1
)

drawing_option = mediapipe.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()

camera = cv2.VideoCapture(0)

pinched = False

while True:

    _, image = camera.read()

    image_height, image_width, _ = image.shape

    image = cv2.flip(image, 1)

    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    output_hands = capture_hands.process(rgb_image)

    all_hands = output_hands.multi_hand_landmarks

    if all_hands:

        for hand in all_hands:

            drawing_option.draw_landmarks(
                image,
                hand
            )

            one_hand_landmarks = hand.landmark

            for id, lm in enumerate(one_hand_landmarks):

                x = int(lm.x * image_width)
                y = int(lm.y * image_height)

                # Index finger tip
                if id == 8:

                    mouse_x = int(
                        (screen_width / image_width) * x
                    )

                    mouse_y = int(
                        (screen_height / image_height) * y
                    )

                    cv2.circle(
                        image,
                        (x, y),
                        10,
                        (0, 255, 255),
                        -1
                    )

                    pyautogui.moveTo(
                        mouse_x,
                        mouse_y
                    )

                    x1 = x
                    y1 = y

                # Thumb tip
                if id == 4:

                    x2 = x
                    y2 = y

                    cv2.circle(
                        image,
                        (x, y),
                        10,
                        (0, 255, 255),
                        -1
                    )

        # Actual distance between thumb and index finger
        dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

        # Click once when pinch begins
        if dist < 30 and not pinched:

            pyautogui.click()

            pinched = True

        # Reset when fingers move apart
        if dist > 40:

            pinched = False

    cv2.imshow(
        "Hand movement video capture",
        image
    )

    # 1 ms instead of 100 ms
    key = cv2.waitKey(1)

    if key == 27:
        break


camera.release()
capture_hands.close()
cv2.destroyAllWindows()