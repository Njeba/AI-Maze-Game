# 🧠 AI Maze Game

An AI-powered maze game integrating **path planning**, **speech recognition**, and **computer vision** into a single interactive system.
The project demonstrates how multiple AI technologies can work together in real time through autonomous navigation, voice control, and gesture-based interaction.

---

## 🚀 Features

* 🔍 **Autonomous Maze Solving**

  * Uses **Breadth-First Search (BFS)** to find the shortest path

* 🧱 **Procedural Maze Generation**

  * Mazes generated using **Depth-First Search (DFS) recursive backtracking**

* 🎤 **Voice Control**

  * Navigate using commands such as:

    * “Go up”
    * “Go down”
    * “Go left”
    * “Go right”

* ✋ **Gesture Recognition**

  * Real-time hand tracking using **MediaPipe Hands**
  * Custom gesture classification using **K-Nearest Neighbors (KNN)**

* ⌨️ **Keyboard Controls**

  * Supports Arrow Keys and WASD movement

* 🎮 **Difficulty Modes**

  * Easy → 4x4
  * Medium → 8x8
  * Hard → 16x16

---

## 🛠️ Technologies Used

* Python
* OpenCV
* MediaPipe
* Scikit-learn
* SpeechRecognition
* Tkinter
* NumPy

---

## 🧠 AI Concepts Demonstrated

### Path Planning

* Breadth-First Search (BFS)
* Depth-First Search (DFS)

### Machine Learning

* K-Nearest Neighbors (KNN)

### Computer Vision

* Real-time hand landmark detection
* Gesture classification

### Speech Recognition

* Voice command processing using Google Speech API
   *Go Up
   *Go Down
   *Go Left
   *Go Right
---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/ai-maze-game.git
cd ai-maze-game
```

Install dependencies:

```bash
pip install opencv-python mediapipe scikit-learn numpy SpeechRecognition pyaudio
```

Run the game:

```bash
python maze_game.py
```

---

## 🎮 How to Play

1. Select a difficulty level
2. Choose a control mode:

   * Auto Solve
   * Keyboard
   * Voice Control
   * Hand Gesture
3. Navigate from the green start tile to the gold finish tile

---

## 🧪 Training Gesture Controls

The game includes a built-in gesture training system.

1. Open **Train Gestures**
2. Select a label:

   * Up
   * Down
   * Left
   * Right
3. Capture gesture samples
4. Train and save the model

Training data is stored locally in:

```bash
gesture_training_data.json
```

---

## 🔮 Future Improvements

* CNN-based gesture recognition
* Offline speech recognition
* A* pathfinding implementation
* Dynamic obstacle avoidance
* Reinforcement learning agents
* Multiplayer support
* Modern game engine integration

---

## 👨‍💻 Developed By

**Njeba**

---

## 📄 License

This project is licensed under the MIT License.
