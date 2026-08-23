# A brief summary

An interactive computer vision fitness project built by our hackathon team. Our software uses a webcam to automatically count reps, monitor the user's form and movement based on body part angles, and provide live analysis feedback. 
The project ensures good form by only allowing a full extension and flexion motion with the elbow. The tracker makes workouts easier for users to follow without wearing a physical device or using specialised gym equipment. The user simply needs to positions themselves side-on (perpendicular) towards the camera and can begin exercising while their reps, calories burned and more, is displayed on the screen.  

## Our vision

Our vision is to make personal exercise tracking more accessible, functional and safer. Our software combines a repetition counter, workout timer and a movement analysis tool into one. The elbow angles allow for incorrect form to not be counted allowing for users to recognise bad technique, reducing the risk of injury.  


## Features

- **Automatic rep counting:** Increases a counter when the user completes a full range of motion exercise
- **Form Analysis**: Checks elbow angle and shoulder to heel alignment 
- **Live Visual Feedback:** Displays body part angles and movement stages while the exercise is being completed.
- **Audio Feedback:** Plays a beep after each successful repetition.
- **Workout Controls:** Allows the user to reset, save or quit the session with keyboard controls. 
- **Session Records:**  Saves completed workout records into a database with a date and time. 
- **Workout Timer:** Tracks the total workout time the user has been exercising for.


## How it works

The application uses MediaPipe Pose to identify important body features, including the shoulder, elbow, wrist, hip, knee, ankle and heel. It then uses these coordinates to calculate the user's elbow and body angles. When the elbow bends past the set down threshold, the program recognises the bottom/full extension of an exercise. When the arm is straightened again, it counts one completed repetition. The position must be detected across multiple frames which helps prevent camera noise from creating false repetitions. The program also compares the positions of the shoulder, hip and heel to give the user immediate alignment feedback needed for most workouts. 

## Controls

R — reset the repetition counter.

S — saves the current workout.

B — turn the completion beep sound on or off.

Q — closes the application.

E - ends the current session

A - decreases the volume of the background music

D - increases the volume of the background music

C - continue from instruction screen to main workout

## Hackathon scope

Our first demo will focus on a reliable core interaction:

1. Open the webcam and detect the user's pose.
2. Select and position the body into the most visible side. 
3. Calculate elbow and body alignment angles.
4. Recognise the down and up / extension and flexion stages of a workout.
5. Count completed repetitions and provide immediate results.
6. Save the user's workout result if wanted.

Future versions could support additional exercises, workout goals, set and rest tracking, progress graphs, voice feedback, and personalised form thresholds.


## Technology Used

- Python

- OpenCV

- MediaPipe Pose

- NumPy

- Github

- VSCode


