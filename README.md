# Identificação de Placas de Veículos com YOLOv8 e OCR

## 📝 Descrição
Este projeto implementa um sistema de **detecção e reconhecimento de placas de veículos** utilizando **YOLOv8** para identificar a placa na imagem e **EasyOCR** para extrair os caracteres.

## ⚙️ Tecnologias Utilizadas
- **Python** → Linguagem principal
- **YOLOv8** (Ultralytics) → Para detecção das placas
- **EasyOCR** → Para reconhecimento de caracteres
- **OpenCV** → Para manipulação de imagens
- **Matplotlib** → Para exibição das imagens

## 📂 Estrutura do Projeto
```
/
├── identificação_de_placas.py   # Código principal
├── dataset_placas/              # Dataset de treinamento
│   ├── images/                  # Imagens de placas
│   ├── labels/                  # Arquivos de anotações YOLO
│   ├── data.yaml                # Configuração do dataset
└── README.md                    # Documentação do projeto
```

## 📥 Instalação
Antes de rodar o projeto, instale as dependências necessárias:
```bash
pip install ultralytics opencv-python easyocr numpy matplotlib pyyaml
```

## 📌 Uso
### 1️⃣ **Carregar e Detectar Placas com YOLOv8**
```python
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Carregar o modelo YOLOv8 pré-treinado
model = YOLO('yolov8n.pt')
image_path = '/content/placas-vermelhas.jpg'
image = cv2.imread(image_path)

# Detectar placas
results = model(image)

# Desenhar caixas de detecção
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 3)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
```

### 2️⃣ **Treinar YOLOv8 para Melhorar Detecção**
```python
# Treinar o modelo YOLOv8
model.train(data='/content/dataset_placas/data.yaml', epochs=50, imgsz=640, batch=16, name='yolov8_placas')
```

### 3️⃣ **Testar o Modelo Treinado**
```python
# Carregar o modelo treinado
model = YOLO('/content/runs/detect/yolov8_placas/weights/best.pt')
image_path = '/content/exemplo-modelo-nova-placa-carro.jpg'
image = cv2.imread(image_path)
results = model(image)

# Exibir a imagem com a detecção
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
```

### 4️⃣ **Recortar a Placa e Aplicar OCR**
```python
import easyocr

# Criar leitor OCR
reader = easyocr.Reader(['en'])
placa_texto_easyocr = reader.readtext('/content/placa_detectada.jpg', detail=0)

# Exibir a placa detectada
print('Placa:', placa_texto_easyocr)
```

## 🤝 Contribuição
Sinta-se à vontade para contribuir! Faça um **fork** do repositório, abra uma **issue** ou envie um **pull request** com melhorias.

---
