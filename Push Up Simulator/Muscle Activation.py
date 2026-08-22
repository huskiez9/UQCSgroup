import cv2
import mediapipe as mp
import numpy as np
from collections import deque


#Some parameters
CAM_WIDTH = 1280
CAM_HEIGHT = 720

DOWN_ANGLE = 90
UP_ANGLE = 160

ARM_VISIBILITY = 0.30 #Set arm visibility score
BODY_VISIBILITY = 0.20

DOWN_CONFIRM_FRAMES = 3
UP_CONFIRM_FRAMES = 3

ANGLE_HISTORY_SIZE = 3


WHITE = (255, 255, 255)
GREEN = (0, 255, 120)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)

ARM_COLOR = (0, 255, 255)
TORSO_COLOR = (0, 255, 120)
LEG_COLOR = (255, 180, 0)

# ============================== MEDIAPIPE SETUP ==============================
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

#Pose setting
pose = mp_pose.Pose(
    static_image_mode=False, model_complexity=1, smooth_landmarks=True,
    min_detection_confidence=0.45, min_tracking_confidence=0.45
)

#Landmark indices for left and right body parts
BODY = {
    "left":  {"shoulder": 11, "elbow": 13, "wrist": 15, "hip": 23, "knee": 25, "ankle": 27},
    "right": {"shoulder": 12, "elbow": 14, "wrist": 16, "hip": 24, "knee": 26, "ankle": 28}
}


elbow_history = deque(maxlen=ANGLE_HISTORY_SIZE)
hip_history = deque(maxlen=ANGLE_HISTORY_SIZE)
knee_history = deque(maxlen=ANGLE_HISTORY_SIZE)



def calculate_angle(point_a, point_b, point_c):
    ba_x = point_a[0] - point_b[0]
    ba_y = point_a[1] - point_b[1]
    bc_x = point_c[0] - point_b[0]
    bc_y = point_c[1] - point_b[1]

    mag_ba = np.sqrt(ba_x ** 2 + ba_y ** 2)
    mag_bc = np.sqrt(bc_x ** 2 + bc_y ** 2)

    if mag_ba < 1e-6 or mag_bc < 1e-6: #Return none if vector is too small to avoid zero division error
        return None
    dot_product = ba_x * bc_x + ba_y * bc_y #Dot product of vectors BA and BC
    cosine = dot_product / (mag_ba * mag_bc) #Cosine of vectors BA and BC
    return float(np.degrees(np.arccos(cosine))) #Angle in degrees between vectors BA and BC


def get_point(landmark):
    return [landmark.x, landmark.y] #Return the x and y coordinates of a LANDMARK AS A LIST


def get_pixel(landmark, width, height):
    return (int(landmark.x * width), int(landmark.y * height)) #Return the PIXEL COORDINATES of a LANDMARK as tuple


def filter_angle(history, value): #Calculate the median of the angles to "filter out the noise", and filter out None values
    if value is None:
        return None
    history.append(value)
    return float(np.median(history))


def arm_visibility_score(landmarks, side):
    indice_list = BODY[side]
    shoulder = landmarks[indice_list["shoulder"]].visibility
    elbow = landmarks[indice_list["elbow"]].visibility          
    wrist = landmarks[indice_list["wrist"]].visibility                              
    return (shoulder + elbow + wrist) / 3.0


def choose_side(landmarks):
    left_score = arm_visibility_score(landmarks, "left") #Retrieve visibility score for left and right side to determine side to pick
    right_score = arm_visibility_score(landmarks, "right")
    if right_score >= left_score:
        return "right"
    return "left"


def arm_visible(landmarks, side):
    ids = BODY[side]
    shoulder = landmarks[ids["shoulder"]]
    elbow = landmarks[ids["elbow"]]
    wrist = landmarks[ids["wrist"]]
    return (shoulder.visibility >= ARM_VISIBILITY and elbow.visibility >= ARM_VISIBILITY and wrist.visibility >= ARM_VISIBILITY)


def get_elbow_angle(landmarks, side):
    ids = BODY[side]
    shoulder = get_point(landmarks[ids["shoulder"]])
    elbow = get_point(landmarks[ids["elbow"]])
    wrist = get_point(landmarks[ids["wrist"]])
    return calculate_angle(shoulder, elbow, wrist)


def get_body_angles(landmarks, side):
    ids = BODY[side]
    shoulder_lm = landmarks[ids["shoulder"]]
    hip_lm = landmarks[ids["hip"]]
    knee_lm = landmarks[ids["knee"]]
    ankle_lm = landmarks[ids["ankle"]]

    hip_angle = None
    knee_angle = None

    #CALCULATE HIP ANGLE BASED ON SHOULDER, HIP AND KNEE LANDMARKS IF ABOVE BODY_VISIBILITY
    if (shoulder_lm.visibility >= BODY_VISIBILITY and hip_lm.visibility >= BODY_VISIBILITY
            and knee_lm.visibility >= BODY_VISIBILITY):
        hip_angle = calculate_angle(get_point(shoulder_lm), get_point(hip_lm), get_point(knee_lm))

   #CALCULATE KNEE ANGLE BASED ON HIP, KNEE AND ANKLE IF ABOVE BODY_VISIBILITY
    if (hip_lm.visibility >= BODY_VISIBILITY and knee_lm.visibility >= BODY_VISIBILITY
            and ankle_lm.visibility >= BODY_VISIBILITY):
        knee_angle = calculate_angle(get_point(hip_lm), get_point(knee_lm), get_point(ankle_lm))

    return (hip_angle, knee_angle)

def get_chest(frame,landmarks):
    height, width = frame.shape[:2]
    left_shoulder = get_point(landmarks[12])
    right_shoulder = get_point(landmarks[11])
    left_hip = get_point(landmarks[24])
    right_hip = get_point(landmarks[23])

    chest_x = 0.5*(left_shoulder[0]+right_shoulder[0])+ left_shoulder[0]
    chest_y = left_shoulder[1]- 0.33*(left_shoulder[1]+left_hip[1]) 
    chest_coord = (int(chest_x * width), int(chest_y * height))
    return chest_coord

def draw_text(frame, text, position, color=WHITE, scale=0.7, thickness=2):
    x, y = position
    # Black outline
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_PLAIN, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA) #Frame means the image, text is the string to be displayed, position is the coordinates of the text, color is the color of the text, scale is the size of the text...)
    # Main text
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_PLAIN, scale, color, thickness, cv2.LINE_AA)


def draw_active_side(frame, landmarks, side):
    """
    Draws the side of the body which is the most visible (left or right) based on visibility 
    threshold. It first retreives the landsmarks based on the side of the body, then for 
    each landmark, it retrieves the actual mediapipe landmark and converts it to pixel coordinates. 
    Then it stores the pixel coordinates in a dictionary called points. It then draws the lines between
    landmarks to represent the arm, torso, upper leg and lower leg....

    """
    height, width = frame.shape[:2] #Get the height and width of the frame
    landmarks_dict = BODY[side] #Retrieve all the landmarks based on the side (left or right) and store them in dict
    points = {}
    for name, index in landmarks_dict.items(): 
        actual_mediapipe_landmark = landmarks[index]
        points[name] = get_pixel(actual_mediapipe_landmark, width, height) #Get the pixel coordinates of the landmark

    # ARM
    cv2.line(frame, points["wrist"], points["elbow"], (0, 255, 255), 5, cv2.LINE_AA)
    cv2.line(frame, points["elbow"], points["shoulder"], (0, 255, 255), 5, cv2.LINE_AA)

    #CHEST
    chest_point = get_chest(frame,landmarks)
    cv2.circle(frame, chest_point, 12, (0, 0, 255), -1)

    # TORSO
    if landmarks[landmarks_dict["hip"]].visibility >= BODY_VISIBILITY and landmarks[landmarks_dict["shoulder"]].visibility >= BODY_VISIBILITY:
        cv2.line(frame, points["shoulder"], points["hip"], (0, 255, 255), 5, cv2.LINE_AA) #Draw a line from shoulder to points

    # UPPER LEG
    if (landmarks[landmarks_dict["hip"]].visibility >= BODY_VISIBILITY and landmarks[landmarks_dict["knee"]].visibility >= BODY_VISIBILITY):
        cv2.line(frame, points["hip"], points["knee"], LEG_COLOR, 5, cv2.LINE_AA)

    # LOWER LEG
    if (landmarks[landmarks_dict["knee"]].visibility >= BODY_VISIBILITY and landmarks[landmarks_dict["ankle"]].visibility >= BODY_VISIBILITY):
        cv2.line(frame, points["knee"], points["ankle"], LEG_COLOR, 5, cv2.LINE_AA)

    # DRAW JOINTS INDIVIDUALLY IF VISIBLE
    for name, index in landmarks_dict.items():
        visibility = landmarks[index].visibility
        if visibility >= BODY_VISIBILITY:
            cv2.circle(frame, points[name], 7, WHITE, -1, cv2.LINE_AA)
            cv2.circle(frame, points[name], 11, WHITE, 2, cv2.LINE_AA)

    return points


def main():
    cap = cv2.VideoCapture(0)   
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH) #Defne the width and height of camera frame
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT) #Define the width and height of camera frame
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) #Define the buffer size of camera frame

    if not cap.isOpened(): #If camera is not opened...
        print("Sorry, camera couldnt be opened!")
        return

    #Default values for reps, stage, bottom_reached, down_frames and up_frames
    reps = 0
    stage = "UP"
    bottom_reached = False
    down_frames = 0
    up_frames = 0

    if not run_countdown(cap, seconds=5): #Run countdown before starting the pushup tracker
        cap.release()
        cv2.destroyAllWindows()
        return
    
    try:
        while cap.isOpened(): #While camera is opened, read the camera frame and process it
            camera_read_successfully, frame = cap.read() 
            if not camera_read_successfully:
                print("Sorry, camera frame couldnt be read!")
                break

            # MIRROR CAMERA
            frame = cv2.flip(frame, 1) #Flip the frame horizontally
            h, w = frame.shape[:2]

            # MEDIAPIPE
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #Convert
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True

            # POSE DETECTED
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # ALWAYS DRAW THE FULL MEDIAPIPE SKELETON
                # This lets you immediately see whether MediaPipe itself is detecting your body.
                mp_draw.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(100, 100, 100), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(80, 80, 80), thickness=2, circle_radius=2))

                
            
                # PICK BEST SIDE
                side = choose_side(landmarks)
                left_score = arm_visibility_score(landmarks, "left")
                right_score = arm_visibility_score(landmarks, "right")

                #Code for displaying         
                draw_text(frame, f"Side: {side.upper()}", (25, 35), GREEN, 0.6, 2) #Displays side which is drawn by media pipe
                draw_text(frame, f"L_visibility: {left_score:.2f}", (25, 65), WHITE, 0.5, 1)
                draw_text(frame, f"R_visibility: {right_score:.2f}", (25, 90), WHITE, 0.5, 1)

                # ARM AVAILABLE
                if arm_visible(landmarks, side):

                    points = draw_active_side(frame, landmarks, side) 

                    # ELBOW ANGLE
                    raw_elbow_angle = get_elbow_angle(landmarks, side)                
                    filtered_elbow_angle = filter_angle(elbow_history, raw_elbow_angle)

                    # BODY ANGLES
                    (raw_hip_angle, raw_knee_angle) = get_body_angles(landmarks, side)
                    hip_angle = filter_angle(hip_history, raw_hip_angle)
                    knee_angle = filter_angle(knee_history, raw_knee_angle)

                    # SHOW ELBOW ANGLE
                    if filtered_elbow_angle is not None:
                        elbow_x_coord, elbow_y_coord = points["elbow"]
                        draw_text(frame, f"{filtered_elbow_angle:.1f}", (elbow_x_coord + 20, elbow_y_coord - 15), YELLOW, 0.9, 2) #Display the angle SLIGHTLY ABOVE AND RIGHT OF THE ELBOW JOINT

                    # REP LOGIC
                    if filtered_elbow_angle is not None: #If elbow angle detected, let pushup commence
                    
                        if filtered_elbow_angle <= DOWN_ANGLE:
                            down_frames += 1 #I guess 3 frames is enough...
                            if down_frames >= DOWN_CONFIRM_FRAMES:  
                                stage = "DOWN"
                                bottom_reached = True
                                down_frames = 0
                        else:
                            down_frames = 0 #Set down_frames to 0 if elbow_angle is not below down_angle   

                        # RETURN TO TOP
                        if filtered_elbow_angle >= UP_ANGLE and bottom_reached: 
                            up_frames += 1
                            if up_frames >= UP_CONFIRM_FRAMES:
                                reps += 1
                                stage = "UP"
                                bottom_reached = False
                                up_frames = 0
                                print(f"Push-up completed! Total: {reps}")
                        else:
                            up_frames = 0

                    # DISPLAY INFORMATION
                    draw_text(frame, f"REPS: {reps}", (25, 145), GREEN, 1.25, 3)

                    stage_color = GREEN if stage == "UP" else ORANGE
                    draw_text(frame, f"STAGE: {stage}", (25, 185), stage_color, 0.7, 2)

                    if filtered_elbow_angle is not None:
                        draw_text(frame, f"Elbow: {filtered_elbow_angle:.1f}", (25, 225), WHITE, 0.6, 1)

                    if hip_angle is not None:
                        hip_color = GREEN if hip_angle >= 160 else RED
                        draw_text(frame, f"Hip: {hip_angle:.1f}", (25, 255), hip_color, 0.6, 1)
                    else:
                        draw_text(frame, "Hip not visible", (25, 255), ORANGE, 0.55, 1)

                    if knee_angle is not None:
                        knee_color = GREEN if knee_angle >= 160 else ORANGE
                        draw_text(frame, f"Knee: {knee_angle:.2f}", (25, 285), knee_color, 0.6, 1)
                    else:
                        draw_text(frame, "Knee not visible!", (25, 285), ORANGE, 0.55, 1)

                # BODY FOUND BUT ARM NOT RELIABLE
                else:
                    elbow_history.clear()
                    draw_text(frame, "BODY FOUND - SHOW ARM TO CAMERA", (25, h - 35), RED, 0.7, 2)

            else:
                elbow_history.clear()
                hip_history.clear()
                knee_history.clear()
                draw_text(frame, "NO PERSON DETECTED", (25, h - 35), RED, 0.8, 2)

            # DISPLAY
            cv2.imshow("Side Push-Up Tracker", frame)

            # KEYBOARD
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):  # Q = quit
                break

            elif key == ord("r"):  #Reset counts, rep, stage default up, bottom reach false
                reps = 0
                stage = "UP"
                bottom_reached = False
                down_frames = 0
                up_frames = 0
                elbow_history.clear()
                hip_history.clear()
                knee_history.clear() 
                print("[INFO] Counter reset")

    finally:
        cap.release()
        pose.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()