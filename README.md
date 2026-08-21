# Airscript

An educational visual-recognition project built by our hackathon team. It lets
a presenter write and draw in the air while a computer displays their work as
a live digital whiteboard.

The experience is inspired by teaching videos where an instructor can be seen
alongside their handwritten maths, diagrams, and explanations—except the
whiteboard is virtual. The presenter uses hand gestures in front of a camera,
and the audience sees both the presenter and the writing on screen.

## Vision

Make classroom explanations more natural, expressive, and interactive. A
teacher should be able to explain a concept by writing equations, sketching
diagrams, and highlighting ideas without needing to stand in front of a
physical whiteboard or tablet.

## Planned features

- **Air writing:** write letters, numbers, and equations using hand gestures.
- **Air drawing:** sketch lines, shapes, diagrams, and annotations.
- **Eraser:** remove selected strokes or clear the board.
- **Zoom controls:** zoom in and out of the virtual whiteboard.
- **Graphs and visual aids:** display simple graphs and other teaching visuals.
- **Live presenter view:** show the presenter’s camera feed together with the
  virtual whiteboard content.

## How it works

The app uses a webcam and hand tracking to identify the presenter’s hand and
gesture position. When the drawing gesture is active, the app maps the hand’s
position to a digital canvas and creates a stroke. Releasing the gesture ends
the current stroke.

## Hackathon scope

Our first demo will focus on a reliable core interaction:

1. Open the webcam and detect a hand.
2. Use a pinch or similar gesture to start and stop drawing.
3. Draw the hand movement onto an on-screen canvas.
4. Provide a clear/erase control.
5. Show the presenter and the virtual whiteboard together.

Extra features such as zooming, graphs, colours, shapes, and handwriting
recognition can be added after the basic drawing experience works smoothly.
