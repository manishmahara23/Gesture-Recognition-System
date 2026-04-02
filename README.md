#  Gesture Recognition System
### Built with MediaPipe + OpenCV | Python

---

##  Requirements
- Python 3.8 or above
- Laptop webcam

---

##  Setup (One-time)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the project:**
   ```bash
   python gesture_recognition.py
   ```

3. **Press `Q` to quit.**

---

##  Recognized Gestures

| Gesture        | Description                        |
|----------------|------------------------------------|
| ✊ Fist         | All fingers closed                 |
| 🖐 Open Hand    | All fingers extended               |
| 👍 Thumbs Up   | Only thumb up                      |
| 👎 Thumbs Down | Only thumb down                    |
| ✌️ Peace        | Index + Middle finger extended     |
| 👆 Pointing    | Only index finger extended         |
| 🤘 Rock On     | Index + Pinky extended             |
| 👌 OK          | Thumb + Index tips touching        |
| 🤟 Three       | Index + Middle + Ring extended     |

---

##  How It Works

1. **Webcam captures** live video frames
2. **MediaPipe Hands** detects 21 hand landmarks per hand
3. **Finger state logic** checks if each finger is extended or closed
4. **Gesture classifier** maps the finger states to a gesture name
5. **OpenCV** overlays the result on screen in real-time

---

##  File Structure
```
├── .gitignore
├── README.md               
├── gesture_recognition.py            
└── requirements.txt                 
```

---

##  Tips
- Make sure your hand is **clearly visible** in the frame
- Good **lighting** improves detection accuracy
- Works with **up to 2 hands** simultaneously
- Hold gestures **steadily** for a moment for best results

---

## 🚀 Possible Extensions
- Add custom gestures
- Map gestures to keyboard/mouse controls
- Control media player or presentation slides
- Add gesture history log
