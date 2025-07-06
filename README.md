# 🧠 A-EYE: Intelligent Surveillance System

**A-EYE** is a smart surveillance solution designed to detect suspicious behavior and identify potential thieves in real-time using computer vision and AI. It also predicts the **age**, **gender**, and **race** of individuals based on facial analysis to aid in better profiling and threat assessment.

![]()

---

## 🚀 Features

- 👁️ Real-time detection of **suspicious behavior**
- 🕵️ Identification of **potential thieves**
- 🧓 Age, 👩 Gender, and 🌍 Race prediction via facial recognition
- 📹 Integration with camera feeds for live monitoring
- 🧠 Powered by state-of-the-art AI and deep learning models

---

## 🔍 Use Cases

- Retail theft detection
- Smart building security
- Public safety monitoring (malls, stations, airports)
- Law enforcement support systems

---

## 🛠️ Tech Stack

- Python (OpenCV, TensorFlow/PyTorch)
- Deep Learning for behavior and face analysis
- REST API / WebSocket for integration
- (Optional) Frontend dashboard for visualization



## 🧱 System Architecture

```text
┌────────────┐     ┌─────────────┐     ┌────────────────┐
│ Video Feed │──▶▶│ Preprocessor│──▶▶│ Person Tracker │
└────────────┘     └─────────────┘     └────────────────┘
    │                    │                     │
    ▼                    ▼                     ▼
Source from      Frame resizing,       Detects and assigns
camera or file   FPS control,          unique IDs to each
(RTSP, MP4, etc) background blur,      person across frames
                 ROI masking

                            ┌─────────────┴───────────────┐
                            ▼                             ▼
                  ┌────────────────┐             ┌────────────────────┐
                  │ Behavior Model │             │ Face Detection     │
                  │ (CNN + LSTM)   │             │ + Demographic Model│
                  └────────────────┘             └────────────────────┘
                          ▲                               ▲
    Spatio-temporal motion patterns         Aligned face crops sent to
    analyzed for suspicious cues            deep learning model for:
    like loitering, quick grabs,            - Age group estimation  
    sudden direction changes, etc.          - Gender classification  
                                            - Race prediction

                            ▼                             ▼
                          Output         ◀────────────▶  Profiling
                          Alerts / Logs / Visualization
                          Suspicion score, demographics,
                          visual overlays, and optional
                          REST API integration
```

---

## 📄 License

This project is licensed under the MIT License. See **LICENSE** for more details.



