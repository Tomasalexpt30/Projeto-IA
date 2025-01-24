from transformers import pipeline
from deep_translator import GoogleTranslator
from deepface import DeepFace
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# === Reconhecimento de Emoções em Texto === #
try:
    emotion_detector_text = pipeline(
        "text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None
    )
except Exception as e:
    print(f"Erro ao carregar o modelo de análise de texto: {e}")
    emotion_detector_text = None

# Função para traduzir texto para inglês
def translate_to_english(text):
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception as e:
        print(f"Erro ao traduzir o texto: {e}")
        return text

# Função para processar texto
def analyze_text_emotion():
    if not emotion_detector_text:
        print("O modelo de análise de texto não está carregado corretamente.")
        return

    print("=== Reconhecimento de Emoções (Texto) ===")
    user_input = input("Descreve como te sentes (ex.: 'Estou feliz' ou 'Estou triste e ansioso'): ")

    # Traduzir o texto para inglês
    translated_input = translate_to_english(user_input)
    print(f"\nTexto traduzido para inglês: {translated_input}")

    print("\nProcessando o texto...\n")
    try:
        results = emotion_detector_text(translated_input)

        # Mostrar as 3 emoções mais relevantes
        print("Emoções Detetadas (Texto):")
        sorted_results = sorted(results[0], key=lambda x: x['score'], reverse=True)[:3]
        for result in sorted_results:
            print(f"{result['label']}: {result['score'] * 100:.2f}%")
    except Exception as e:
        print(f"Erro ao processar o texto: {e}")

# === Reconhecimento de Emoções em Imagens === #
# Função para pré-processar imagens
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Imagem não encontrada ou inválida.")

    # Redimensionar e converter para RGB
    img = cv2.resize(img, (640, 480))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalização (opcional, dependendo do modelo usado)
    img = img / 255.0

    return img

# Função para processar imagens usando DeepFace
def analyze_image_emotion():
    print("\n=== Reconhecimento de Emoções (Imagem) ===")
    image_name = input("Insere o nome da imagem (ex.: 'imagem.jpg'): ")

    # Caminho padrão para a pasta onde o script está localizado
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, image_name)

    if not os.path.exists(image_path):
        print("Erro: Caminho para a imagem não encontrado!")
        return

    try:
        # Pré-processar a imagem
        img = preprocess_image(image_path)

        # Usar o DeepFace para analisar emoções na imagem
        result = DeepFace.analyze(
            img_path=image_path,
            actions=["emotion"],  # Analisar apenas emoções
            enforce_detection=True,  # Forçar a deteção de rostos
            detector_backend="mtcnn"  # Backend para deteção de rosto
        )

        # Garantir que o resultado seja tratado como uma lista (se múltiplas imagens forem processadas)
        if isinstance(result, list):
            result = result[0]

        # Obter as emoções detetadas
        print("\nEmoções Detetadas (Imagem):")
        emotions = result["emotion"]

        # Traduzir emoções para português
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

        # Mostrar a emoção mais dominante
        dominant_emotion = result["dominant_emotion"]
        dominant_emotion_pt = translated_emotions.get(dominant_emotion, dominant_emotion).capitalize()
        print(f"\nEmoção Mais Dominante: {dominant_emotion_pt}")

        # Exibir a imagem analisada
        plt.figure(figsize=(10, 5))

        # Mostrar a imagem
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f"Emoção Dominante: {dominant_emotion_pt}")
        plt.axis("off")

        # Gráfico de barras das emoções com cores personalizadas
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

        # Mostrar tudo
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Erro ao processar a imagem: {e}")

# === Sistema Principal === #
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

# Executar o sistema
if __name__ == "__main__":
    main()
