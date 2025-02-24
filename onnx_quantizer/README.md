Voici un **README** complet, professionnel et clair pour `onnx_quantizer/`. Il suit les bonnes pratiques en termes de structure et d’explication, avec des **exemples concrets** pour une compréhension rapide. 🚀  

---  

# 🏗 **ONNX Quantizer**  

`onnx_quantizer` est une bibliothèque modulaire permettant de quantifier des modèles ONNX afin d’optimiser leurs performances en réduisant la taille et en accélérant l’inférence, tout en minimisant la perte de précision.  

## ✨ **Fonctionnalités**  
✅ **Quantification Automatique** : FP32 → FP16, INT8 avec calibration, etc.  
✅ **Prise en charge de plusieurs backends** : ONNX Runtime, TensorRT, OpenVINO...  
✅ **Facilité d’utilisation** : Interface simple avec support des fichiers YAML pour la configuration.  
✅ **Extensible** : Ajoutez vos propres méthodes de quantification facilement.  
✅ **Gestion des modèles** : Sauvegarde et chargement des modèles quantifiés automatiquement.  

---

## 📂 **Structure du Projet**  
```
📁 onnx_quantizer/
├── quantizer.py           # Pipeline principal de quantification
├── backends/              # Gestion des backends de quantification
│   ├── __init__.py
│   ├── onnxruntime_q.py   # Quantification avec ONNX Runtime
│   ├── tensorrt_q.py      # Quantification avec TensorRT
│   ├── openvino_q.py      # Quantification avec OpenVINO
├── calibration/           # Gestion des datasets de calibration
│   ├── __init__.py
│   ├── data_loader.py     # Chargement des images de calibration
│   ├── statistics.py      # Calcul des stats pour la calibration INT8
├── storage/               # Stockage et récupération des modèles quantifiés
│   ├── __init__.py
│   ├── model_registry.py  # Gestion des modèles ONNX quantifiés
├── configs/               # Configuration en YAML
│   ├── quantization.yaml  # Paramètres de quantification
└── tests/                 # Tests unitaires et validation
```

---

## 🚀 **Installation**  

### 🔹 **Prérequis**  
- **Python 3.8+**  
- **ONNX Runtime** : `pip install onnxruntime`  
- **ONNX** : `pip install onnx`  
- **TensorRT (optionnel)** : Suivre [la doc officielle](https://developer.nvidia.com/tensorrt)  
- **OpenVINO (optionnel)** : Suivre [la doc officielle](https://docs.openvino.ai/latest/)  

### 🔹 **Installation de la bibliothèque**  
```bash
git clone https://github.com/votre-repo/onnx_quantizer.git
cd onnx_quantizer
pip install -r requirements.txt
```

---

## 🛠 **Utilisation**  

### 📌 **1️⃣ Quantification Simple en FP16**  
```python
from onnx_quantizer.quantizer import ONNXQuantizer

quantizer = ONNXQuantizer("models/yolov8.onnx")
quantized_model = quantizer.quantize("fp16")  # Quantifie en FP16
```

---

### 📌 **2️⃣ Quantification INT8 avec Calibration**  
```python
from onnx_quantizer.quantizer import ONNXQuantizer

quantizer = ONNXQuantizer("models/yolov8.onnx")
quantized_model = quantizer.quantize("int8", calibration_dataset="datasets/calibration/")
```

---

### 📌 **3️⃣ Chargement d’un Modèle Quantifié**  
```python
from onnx_quantizer.storage.model_registry import ModelRegistry

registry = ModelRegistry()
model_path = registry.get_model("yolov8", "int8")
print(f"Modèle quantifié chargé : {model_path}")
```

---

## ⚙ **Configuration avec YAML**  
Vous pouvez configurer les options de quantification avec un fichier `configs/quantization.yaml` :  

```yaml
quantization:
  precision: "int8"
  backend: "onnxruntime"
  calibration:
    dataset_path: "datasets/calibration/"
    num_samples: 100
```

Puis exécutez :  
```python
quantizer.quantize_from_config("configs/quantization.yaml")
```

---

## 📊 **Pourquoi utiliser `onnx_quantizer` ?**  
| Fonctionnalité         | ONNX Quantizer ✅ | Autres 🔻  |
|------------------------|------------------|------------|
| Facilité d'utilisation | ✅ Interface simple  | ❌ Souvent complexe |
| Multiples backends     | ✅ ONNX, TensorRT, OpenVINO  | ❌ Parfois restreint |
| Extensibilité          | ✅ Facile à ajouter un backend  | ❌ Dur à modifier |
| Performance            | ✅ Optimisé pour l'inférence  | 🔄 Variable |

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

**Qu’en penses-tu ? Tu veux ajouter d’autres détails ?** 💡