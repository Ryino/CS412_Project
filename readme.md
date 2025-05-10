# CS412_Project
# 🎭 Co-SA: Contrastive Self-Attention for Emotion Recognition

This project builds a deep learning model called **Co-SA** to recognize emotions (like Happy, Sad, Angry, and Neutral) from text, audio, and video data. It uses modern techniques like contrastive learning, attention mechanisms, and reinforcement learning to make better emotion predictions.

## 🧠 Purpose

The main aim is to improve how well emotion recognition models work by:

- Learning strong and meaningful patterns from multiple types of input (text, audio, and video).
- Using **contrastive learning** to understand both shared and unique features across the different input types.
- Guiding the learning process with **reinforcement learning** using an actor-critic method.
- Using **self-attention** to focus on the most important parts of time-based data.

## 🚀 How the Model Works

The model includes several key parts:

- **Shared & Differential Embedding Model** (`network.Model`): Learns how to combine and compare features from different types of input.
- **Actor-Critic Modules**: These help the model better align different data types using reinforcement learning strategies.
- **Training Tools**: Includes evaluation and learning rate schedulers to train the model smoothly and adapt to changes.

## 📊 Evaluated Emotions

The model can recognize and classify the following emotions:

- 😊 Happy  
- 😢 Sad  
- 😠 Angry  
- 😐 Neutral  

