# Pinch Sketch

A small hackathon demo where a person draws in the air by pinching their thumb
and index finger together in front of a webcam. The app tracks the hand and
draws a line on a canvas at the pinch position while the pinch is held.

## Demo goal

Build a clear, reliable prototype that can:

1. Open the user's webcam.
2. Detect one hand and the tips of the thumb and index finger.
3. Treat the fingers as "pinched" when their distance is below a threshold.
4. Draw connected lines on a canvas at the midpoint of the two fingertips
   while pinched.
5. Stop the current line when the user releases the pinch.
6. Offer a simple clear-canvas button. A colour picker and stroke-width slider
   are nice optional extras.

## Suggested approach

- Use a browser app so it is quick to demo and does not need a separate
  desktop installation.
- Use MediaPipe Hand Landmarker (or MediaPipe Hands) for real-time hand
  landmark detection. This is the ML component of the project.
- Render the camera preview and drawing layer with a HTML canvas. Mirror the
  preview so moving right feels natural.
- Convert the hand-landmark coordinates (normalised from 0 to 1) into canvas
  pixel coordinates. Use the average of the thumb-tip and index-tip positions
  as the drawing point.
- Keep the last point for the active stroke; draw a segment from the last point
  to the new point on every frame. Reset that point when the pinch ends.

## Build priorities

Start with the smallest working version: webcam, landmark overlay, pinch
indicator, then drawing. Make it tolerant of a little tracking jitter by using
a modest pinch threshold and optionally smoothing positions. Do not spend time
on accounts, databases, or a backend unless the core interaction already feels
good.

## Demo script

1. Grant webcam access.
2. Hold a hand up to show the detected landmarks.
3. Pinch thumb and index finger together to begin drawing.
4. Move the pinched fingers to write or sketch in the air.
5. Release to finish the stroke, then use **Clear** to reset the canvas.
