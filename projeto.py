from transformers import pipeline
from deep_translator import GoogleTranslator
from deepface import DeepFace
import os
import cv2
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
        # Carregar e processar a imagem com OpenCV
        img = cv2.imread(image_path)
        img = cv2.resize(img, (640, 480))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Usar o DeepFace para analisar emoções na imagem
        result = DeepFace.analyze(img_path=image_path, actions=["emotion"], enforce_detection=False)

        # Garantir que o resultado seja tratado como uma lista (se múltiplas imagens forem processadas)
        if isinstance(result, list):
            result = result[0]

        # Obter as emoções detetadas
        print("\nEmoções Detetadas (Imagem):")
        for emotion, score in result["emotion"].items():
            print(f"{emotion.capitalize()}: {score:.2f}%")

        # Mostrar a emoção mais dominante
        dominant_emotion = result["dominant_emotion"].capitalize()
        print(f"\nEmoção Mais Dominante: {dominant_emotion}")

        # Exibir a imagem analisada
        plt.imshow(img)
        plt.title(f"Emoção Dominante: {dominant_emotion}")
        plt.axis("off")
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
