# Chemistry Learner

**An interactive chemistry learning tool powered by Python, Pygame, and computer vision.**

Chemistry Learner 是一个基于 **Pygame** 的交互式化学学习应用，集成了 **MediaPipe 手势识别** 与 **OpenCV 视觉处理**，提供无接触式界面操作、化学式渲染与可扩展的学习内容展示。本项目面向教学演示、交互实验、以及化学学习类工具开发。

---

## 📌 Features

* **可交互图形界面（GUI）**
  使用 Pygame 构建菜单导航、内容展示与提示系统。

* **实时手势识别（MediaPipe）**
  利用 MediaPipe Hands 获取手部关键点，实现选择、抓取等基本交互。

* **化学内容渲染**
  支持化学式、下标、反应式的排版与显示。

* **可扩展的查询接口**
  项目预留 API 调用结构，可接入任意知识库或第三方查询服务。

* **跨平台运行**
  除需摄像头外无特殊硬件要求，Windows/macOS/Linux 均可运行。

---

## ⚙️ Requirements

* **Python 3.8+ （推荐Python 3.11.9）**
* **Pygame**
* **OpenCV (opencv-python)**
* **MediaPipe**
* **摄像头设备**

---

## 🛠 Installation

建议使用虚拟环境进行安装：

```bash
# 推荐 Python 版本：3.11.9
# optional: create & activate a venv
# python -m venv venv
# source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate         # Windows

# install dependencies
pip install -U pygame opencv-python mediapipe openai
```

如无需外部查询，可自行移除或替换 `openai`。

---

## 🚀 Running

进入项目根目录后执行：

```bash
python main.py
```

运行后：

1. 程序进入主界面并自动启动摄像头。
2. 使用手势或鼠标进行交互。
3. 在内容界面按 **SPACE** 或 **ESC** 返回上一层菜单。

---

## 🧩 Extending the Project

你可以轻松扩展本项目：

* **添加新的学习章节/化学内容**
* **接入本地化化学数据库**
* **替换为更高性能的手势模型**
* **加入语音输入/语音播报**
* **改造成教育演示软件、虚拟实验交互界面**