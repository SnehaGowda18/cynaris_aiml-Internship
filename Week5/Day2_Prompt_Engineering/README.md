# W5D2: Prompt Engineering & System Prompts with Ollama

## 📌 Description

This project demonstrates Prompt Engineering using Ollama and the Llama 3.2 (3B) Large Language Model (LLM). The objective is to understand how system prompts influence the behavior and quality of AI-generated responses. Different system prompts were used to assign various roles such as Teacher, Python Expert, Cybersecurity Mentor, Interviewer, and Travel Guide. The project also compares responses generated with and without system prompts to observe differences in clarity, tone, structure, and relevance.

---

## 🎯 Learning Objectives

- Understand Prompt Engineering concepts.
- Learn how System Prompts affect LLM responses.
- Interact with Ollama through the Python API.
- Generate role-based AI responses.
- Compare responses with and without System Prompts.

---

## 📂 Project Structure

```
Day2_Prompt_Engineering/
│── app.py
│── compare_prompts.py
│── prompt_templates.py
│── requirements.txt
│── README.md
└── outputs/
    ├── teacher_output.txt
    ├── python_expert_output.txt
    ├── cybersecurity_mentor_output.txt
    ├── interviewer_output.txt
    ├── travel_guide_output.txt
    └── comparison.txt
```

---

## 🛠️ Tools & Technologies

- Python 3.11
- Ollama
- Llama 3.2 (3B)
- Requests Library
- Visual Studio Code
- Git
- GitHub
- PowerShell

---

## 📦 Installation

### Install Python Packages

```bash
pip install -r requirements.txt
```

### Check Ollama Installation

```bash
ollama --version
```

### Verify Installed Models

```bash
ollama list
```

---

## ▶️ Running the Project

### Run Role-Based Prompt Examples

```bash
python app.py
```

### Compare Responses With and Without System Prompts

```bash
python compare_prompts.py
```

---

## 📊 Prompt Roles Implemented

- Teacher
- Python Expert
- Cybersecurity Mentor
- Interviewer
- Travel Guide

---

## 📁 Output Files

The generated responses are automatically saved in the **outputs** folder.

- teacher_output.txt
- python_expert_output.txt
- cybersecurity_mentor_output.txt
- interviewer_output.txt
- travel_guide_output.txt
- comparison.txt

---

## 📈 Results

The project demonstrates that adding a System Prompt significantly improves response quality by:

- Defining the AI's role.
- Producing more relevant answers.
- Improving response clarity.
- Controlling response style and tone.
- Making answers more suitable for the intended audience.

---

## ✅ Learning Outcomes

After completing this project, I was able to:

- Understand Prompt Engineering.
- Use System Prompts effectively.
- Generate role-based AI responses.
- Compare responses with and without System Prompts.
- Interact with local LLMs using Ollama.
- Save AI-generated outputs automatically using Python.

---

## 🚀 Future Improvements

- Add more prompt engineering techniques.
- Compare additional LLMs available in Ollama.
- Experiment with temperature and other generation parameters.
- Build a simple web interface using Streamlit or FastAPI.

