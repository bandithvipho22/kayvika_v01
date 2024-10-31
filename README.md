# Project Overview
In this project, working on khmer sign recognition, it means the model can recognize the sign on hand gesture and generate the sign into text and speech.
Sign language processing (SLP) is a field of research that focuses on the development
of technologies that can understand and generate sign language. In Cambodia, communication for the deaf and mute community is difficult due to the lack of tools that translate Khmer sign language into text or speech. Existing technologies are designed for other languages, leaving a gap for Khmer speakers. This project aims to develop a real-time system that recognizes Khmer sign language, converts it into text, and generates speech, enhancing communication for the hearing-impaired.
# Problem Statement
There are many research projects on sign detection and recognition systems that focus
on sign language detection. However, most of these systems only work with static sign detection
using images, rather than recognizing continuous sign languages through action recognition.
Moreover, Continuous sign language recognition or action recognition system can better handle
the complexity and variability of natural sign language. In addition, advanced systems that
currently exist are not available for Khmer sign language recognition. Developing an action
recognition system that translates these gestures into text and speech is essential for improving
communication for Cambodia’s deaf and mute community.
# Objective
The primary objectives of this project are:
+ To develop a system that accurately recognizes individual Khmer sign language gestures
using hand tracking and gesture recognition techniques.
+ Integrate recognized gestures into meaningful sentences and generate it into Khmer text
and speech.
+ To optimize the system for efficiency and scalability in real-world scenarios, such as web
application, mobile application and edge devices.
# Output
+ Sign Recognition
+ Recognize sign from hand gesture, then generate it to text and speech

![image](https://github.com/user-attachments/assets/4f83cb29-7520-4392-a72e-9aa4a40a5c17)

# Setup Environment
For Window user, first need to install virtual env
```bash
python -m pip install virtualenv
```
Create environment to run sign recognition code
```bash
python -m venv env
python -m virtualenv env
```
Source Environment, for window
```bash
source env\Scripts\activate
```
or, use this
```bash
.\env\Scripts\Activate.ps1
```
Then, install library
```bash
pip install -r requirements.txt
```
or,
```bash
python -m pip install -r requirements.txt
```
