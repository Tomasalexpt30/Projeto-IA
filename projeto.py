from transformers import pipeline
from deep_translator import GoogleTranslator
from deepface import DeepFace
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

try:
    emotion_detector_text = pipeline(
        "text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None
    )
except Exception as e:
    print(f"Erro ao carregar o modelo de análise de texto: {e}")
    emotion_detector_text = None

def translate_to_english(text):
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception as e:
        print(f"Erro ao traduzir o texto: {e}")
        return text


def analyze_text_emotion():
    if not emotion_detector_text:
        print("O modelo de análise de texto não está carregado corretamente.")
        return
    
    print("=== Reconhecimento de Emoções (Texto) ===")
    user_input = input("Descreve como te sentes (ex.: 'Estou feliz' ou 'Estou triste e ansioso'): ")


    translated_input = translate_to_english(user_input)
    print(f"\nTexto traduzido para inglês: {translated_input}")

    print("\nProcessando o texto...\n")
    try:
        results = emotion_detector_text(translated_input)
        print("Emoções Detetadas (Texto):")
        sorted_results = sorted(results[0], key=lambda x: x['score'], reverse=True)[:3]
        for result in sorted_results:
            print(f"{result['label']}: {result['score'] * 100:.2f}%")
    except Exception as e:
        print(f"Erro ao processar o texto: {e}")

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Imagem não encontrada ou inválida.")
    
    img = cv2.resize(img, (640, 480))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return img

def analyze_image_emotion():
    print("\n=== Reconhecimento de Emoções (Imagem) ===")
    image_name = input("Insere o nome da imagem (ex.: 'imagem.jpg'): ")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, image_name)

    if not os.path.exists(image_path):
        print("Erro: Caminho para a imagem não encontrado!")
        return

    try:
        img = preprocess_image(image_path)
        result = DeepFace.analyze(
            img_path=image_path,
            actions=["emotion"], 
            enforce_detection=True, 
            detector_backend="mtcnn" 
        )


        if isinstance(result, list):
            result = result[0]
        print("\nEmoções Detetadas (Imagem):")
        emotions = result["emotion"]

        translated_emotions = {
            "angry": "Raiva",
            "disgust": "Nojo",
            "fear": "Medo",
            "happy": "Felicidade",
            "sad": "Tristeza",
            "surprise": "Surpresa",
            "neutral": "Neutro"
        }

        for emotion, score in emotions.items():
            print(f"{translated_emotions.get(emotion, emotion).capitalize()}: {score:.2f}%")

        dominant_emotion = result["dominant_emotion"]
        dominant_emotion_pt = translated_emotions.get(dominant_emotion, dominant_emotion).capitalize()
        print(f"\nEmoção Mais Dominante: {dominant_emotion_pt}")

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f"Emoção Dominante: {dominant_emotion_pt}")
        plt.axis("off")

        colors = {
            "angry": "red",
            "disgust": "green",
            "fear": "purple",
            "happy": "yellow",
            "sad": "blue",
            "surprise": "orange",
            "neutral": "gray"
        }
        emotion_colors = [colors.get(emotion, "black") for emotion in emotions.keys()]

        plt.subplot(1, 2, 2)
        plt.bar([translated_emotions.get(emotion, emotion).capitalize() for emotion in emotions.keys()],
                emotions.values(),
                color=emotion_colors)
        plt.title("Distribuição de Emoções")
        plt.xlabel("Emoções")
        plt.ylabel("Porcentagem")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Erro ao processar a imagem: {e}")

def main():
    while True:
        print("\n=== Sistema de Reconhecimento de Emoções ===")
        print("1. Analisar emoções em texto")
        print("2. Analisar emoções em imagens")
        print("3. Sair")
        choice = input("Escolhe uma opção (1/2/3): ")

        if choice == "1":
            analyze_text_emotion()
        elif choice == "2":
            analyze_image_emotion()
        elif choice == "3":
            print("Obrigado por usar o sistema!")
            break
        else:
            print("Escolha inválida! Tenta novamente.")

if __name__ == "__main__":
    main()
