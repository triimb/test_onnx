Voici un **README** clair, professionnel et bien structuré pour `onnx_detector/`. Il met en avant l'architecture extensible, les fonctionnalités, et des exemples concrets d'utilisation. 🚀  

---

# 🏗 **ONNX Detector**  

`onnx_detector` est une bibliothèque performante et extensible permettant d’inférer des modèles de détection d’objets au format ONNX. Elle inclut **le préprocessing, l’inférence, le postprocessing** (NMS, etc.), ainsi qu’un système avancé de gestion des modèles et résultats.  

## ✨ **Fonctionnalités**  
✅ **Support de multiples modèles ONNX** (YOLO, Faster R-CNN, SSD, etc.)  
✅ **Gestion des modèles avec différentes précisions** (FP32, FP16, INT8...)  
✅ **Pipeline complet** : Preprocessing, Inférence, Postprocessing  
✅ **NMS (Non-Maximum Suppression) configurable**  
✅ **Sauvegarde automatique des images inférées avec bounding boxes**  
✅ **Configuration flexible via YAML**  

---

## 📂 **Structure du Projet**  
```
📁 onnx_detector/
├── detector.py           # Pipeline principal d’inférence
├── preprocessing/        # Pré-traitement des images
│   ├── __init__.py
│   ├── transforms.py     # Redimensionnement, normalisation, etc.
├── inference/            # Gestion de l’inférence ONNX
│   ├── __init__.py
│   ├── onnx_runner.py    # Chargement et exécution des modèles ONNX
├── postprocessing/       # Post-traitement des résultats
│   ├── __init__.py
│   ├── nms.py            # Suppression des boxes redondantes (NMS)
│   ├── visualization.py  # Dessin des bounding boxes sur les images
├── storage/              # Gestion des modèles et des résultats
│   ├── __init__.py
│   ├── model_registry.py # Stockage des modèles ONNX
│   ├── result_saver.py   # Sauvegarde des images annotées
├── configs/              # Configuration en YAML
│   ├── detector_config.yaml
└── tests/                # Tests unitaires
```

---

## 🚀 **Installation**  

### 🔹 **Prérequis**  
- **Python 3.8+**  
- **ONNX Runtime** : `pip install onnxruntime`  
- **OpenCV** : `pip install opencv-python`  
- **Pillow** : `pip install pillow`  
- **Matplotlib** (pour le plot des résultats) : `pip install matplotlib`  

### 🔹 **Installation de la bibliothèque**  
```bash
git clone https://github.com/votre-repo/onnx_detector.git
cd onnx_detector
pip install -r requirements.txt
```

---

## 🛠 **Utilisation**  

### 📌 **1️⃣ Chargement d’un Modèle et Inférence sur une Image**  
```python
from onnx_detector.detector import ONNXDetector

detector = ONNXDetector("models/yolov8_fp16.onnx")  # Chargement du modèle
boxes, labels, scores = detector.run_inference("images/test.jpg")  # Inférence
```

---

### 📌 **2️⃣ Inférence avec Sauvegarde des Résultats**  
```python
from onnx_detector.detector import ONNXDetector

detector = ONNXDetector("models/yolov8_fp16.onnx")
detector.run_and_save("images/test.jpg", save_path="results/yolov8_fp16/")
```
✅ **L’image avec les boxes détectées est automatiquement sauvegardée**  

---

### 📌 **3️⃣ Configuration avec YAML**  
Vous pouvez configurer le détecteur avec un fichier `configs/detector_config.yaml` :  

```yaml
model:
  path: "models/yolov8_fp16.onnx"
  precision: "fp16"

preprocessing:
  resize: [640, 640]
  normalize: True

postprocessing:
  nms_threshold: 0.4
  confidence_threshold: 0.5

output:
  save_results: True
  save_dir: "results/yolov8_fp16/"
```

Puis exécutez :  
```python
detector.run_from_config("configs/detector_config.yaml", "images/test.jpg")
```

---

## 🔥 **Pourquoi `onnx_detector` ?**  
| Fonctionnalité         | ONNX Detector ✅ | Autres 🔻  |
|------------------------|------------------|------------|
| Facilité d'utilisation | ✅ Interface simple  | ❌ Parfois complexe |
| Multiples modèles      | ✅ YOLO, Faster R-CNN, etc.  | ❌ Parfois limité |
| Post-processing        | ✅ NMS configurable  | ❌ Fixé dans le code |
| Sauvegarde des résultats | ✅ Automatique | ❌ Manuel |

---

## 🧑‍💻 **Contribuer**  
1. **Fork** le projet  
2. **Crée une branche** (`git checkout -b feature-ma-feature`)  
3. **Ajoute tes modifications** (`git commit -m "Ajout d'une nouvelle fonctionnalité"`)  
4. **Push** (`git push origin feature-ma-feature`)  
5. **Ouvre une Pull Request** 🚀  

---

## 📜 **Licence**  
Ce projet est sous licence **MIT**.  

---

## ❓ **Support**  
Si vous avez des questions, ouvrez une **issue** ou contactez-moi ! 🚀  

---

Que penses-tu de ce README ? Tu veux ajouter des détails ? 😊