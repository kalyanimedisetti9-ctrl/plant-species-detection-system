# Plant Species Detection System – Academic Project Documentation 📘

**Project Title:** Plant Species Detection System  
**Academic Level:** Diploma / Undergraduate Project  
**Domain:** Artificial Intelligence, Computer Vision, Deep Learning, Web Development  
**Institution Reference:** Aditya Polytechnic College  

---

## 1. Abstract

Botanical identification plays a vital role in agriculture, forestry, pharmacological research, and environmental conservation. Traditional plant identification relies on manual inspection of physical floras, taxonomic dichotomous keys, and expert consultation, which is labor-intensive and error-prone for non-specialists.

The **Plant Species Detection System** is an automated, web-based intelligent platform engineered using **Python, Django, OpenCV, TensorFlow/Keras, and MySQL**. The system accepts digital photographs of leaves and flora, applies OpenCV image preprocessing (color space conversion, noise reduction, standard resizing, and pixel normalization), and feeds the preprocessed tensor to a Deep Convolutional Neural Network (MobileNetV2 transfer learning). The model classifies the specimen, yielding the common name, official Latin scientific binomial (*e.g., Mangifera indica, Azadirachta indica*), biological family, confidence percentage, and cultivation/medicinal characteristics.

---

## 2. System Architecture & Methodology

```text
┌─────────────────────────┐
│       User Client       │ (Web Browser / Smartphone / Webcam)
└────────────┬────────────┘
             │ HTTP POST (Image Upload)
             ▼
┌─────────────────────────┐
│     Django Backend      │ (URL Routing, Form Validation, Auth, Media Storage)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   OpenCV Preprocessor   │ (cv2.imdecode, cvtColor BGR->RGB, resize 224x224, /255.0)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  TensorFlow / Keras     │ (MobileNetV2 CNN, GlobalAvgPool, Dense Softmax)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Taxonomy Database     │ (PlantSpecies ORM: Scientific Name, Family, Care)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│      Result Display     │ (Confidence Progress Bar, Botanical Profile, History)
└─────────────────────────┘
```

### Mathematical Formulation of Deep Learning Inference:
1. **Input Normalization:**
   $$X_{\text{norm}} = \frac{X_{\text{raw}}}{255.0} \quad \text{where } X_{\text{norm}} \in [0.0, 1.0]^{1 \times 224 \times 224 \times 3}$$

2. **Convolutional Feature Extraction:**
   $$F = \text{MobileNetV2}(X_{\text{norm}})$$

3. **Softmax Output Probability Distribution:**
   $$P(y = i \mid X) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}} \quad \text{for } i = 1, \dots, K$$

4. **Prediction Selection:**
   $$\hat{y} = \arg\max_{i} P(y = i \mid X), \quad \text{Confidence} = P(y = \hat{y} \mid X) \times 100\%$$

---

## 3. Database Schema (Entity-Relationship)

### 1. `PlantSpecies` Table
| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | BigAutoField | Primary Key | Auto-increment unique ID |
| `common_name` | VARCHAR(150) | Unique, Not Null | English / Vernacular name (*e.g., Mango*) |
| `scientific_name` | VARCHAR(200) | Not Null | Latin binomial (*e.g., Mangifera indica*) |
| `family` | VARCHAR(150) | Nullable | Botanical family (*e.g., Anacardiaceae*) |
| `description` | TEXT | Not Null | Detailed morphological description |
| `care_instructions`| TEXT | Nullable | Sunlight, soil, water guidelines |
| `medicinal_uses` | TEXT | Nullable | Health & ecological values |
| `image` | VARCHAR(100) | Nullable | Path to reference image |
| `created_at` | DATETIME | Auto now add | Timestamp |

### 2. `Detection` Table
| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | BigAutoField | Primary Key | Auto-increment unique ID |
| `user_id` | ForeignKey(User)| Nullable | User foreign key (supports guest scans) |
| `uploaded_image` | VARCHAR(100) | Not Null | Path to user uploaded image |
| `predicted_species`| VARCHAR(150)| Not Null | Common name from ML model |
| `scientific_name` | VARCHAR(200)| Not Null | Botanical Latin scientific name |
| `confidence` | FLOAT | Not Null | Prediction probability percentage |
| `plant_species_ref_id`| ForeignKey(PlantSpecies)| Nullable | Link to botanical record |
| `detected_at` | DATETIME | Auto now add | Timestamp |

### 3. `ContactMessage` Table
| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | BigAutoField | Primary Key | Auto-increment unique ID |
| `name` | VARCHAR(120) | Not Null | Sender name |
| `email` | VARCHAR(254) | Not Null | Sender email address |
| `subject` | VARCHAR(200) | Not Null | Message topic |
| `message` | TEXT | Not Null | Inquiry body |
| `is_resolved` | BOOLEAN | Default False | Admin resolution flag |
| `created_at` | DATETIME | Auto now add | Timestamp |

---

## 4. Hardware & Software Specifications

### Hardware Requirements:
* **Processor:** Intel Core i3 / AMD Ryzen 3 or higher (2.0 GHz+)
* **RAM:** 4 GB minimum (8 GB recommended for deep learning inference)
* **Storage:** 2 GB free disk space
* **Display:** $1366 \times 768$ minimum resolution (responsive on all smartphones)
* **Camera:** Built-in webcam or smartphone camera (optional for live snapshots)

### Software Requirements:
* **Operating System:** Windows 10/11, Ubuntu Linux, or macOS
* **Python Runtime:** Python 3.10 to 3.14
* **Web Framework:** Django 6.x
* **Deep Learning:** TensorFlow 2.x, Keras 3.x
* **Image Processing:** OpenCV 4.x (`opencv-python`), Pillow, NumPy
* **Database:** MySQL 8.x / SQLite 3
* **Web Browser:** Google Chrome, Mozilla Firefox, Microsoft Edge, Safari

---

## 5. Testing & Verification

| Test Case ID | Test Description | Expected Result | Actual Result | Status |
|---|---|---|---|---|
| **TC-01** | Upload valid JPG/PNG image | Image uploaded and previewed | Preview displayed correctly | **PASS** |
| **TC-02** | Upload non-image or invalid format | Validation error message displayed | Error handled gracefully | **PASS** |
| **TC-03** | OpenCV Image Preprocessing | Resize to $224 \times 224$ & normalize to $[0,1]$ | Shape `(1, 224, 224, 3)` produced | **PASS** |
| **TC-04** | TensorFlow ML Inference | Correct species & scientific name returned | Output mapped to Latin binomial | **PASS** |
| **TC-05** | User Registration & Login | Account created & authenticated session | Session established | **PASS** |
| **TC-06** | Detection History Logging | Scans saved in user profile | Table & card views updated | **PASS** |
| **TC-07** | Download History CSV Report | CSV spreadsheet downloaded | Formatted CSV file exported | **PASS** |
| **TC-08** | REST API (`/api/detect/`) | JSON response with prediction data | Status 200 OK + JSON | **PASS** |
| **TC-09** | Admin Portal Management | Admin CRUD on species and detections | Admin interface functional | **PASS** |
| **TC-10** | Mobile Responsive Layout | No horizontal scroll on mobile screens | Layout wraps cleanly | **PASS** |

---

## 6. Conclusion & Future Scope

The **Plant Species Detection System** successfully delivers an accurate, responsive, and intuitive web solution for automated botanical identification. The application combines the computing power of convolutional neural networks with a user-friendly dark green and white interface.

### Future Enhancements:
1. **Foliage Disease Diagnostics:** Identifying fungal, bacterial, and pest infections on leaves.
2. **Offline Mobile Application:** Quantizing the Keras model to TensorFlow Lite (TFLite) for offline smartphone scanning in remote agricultural zones.
3. **Multilingual Botanical Support:** Providing common names and care guidelines in regional Indian languages (Hindi, Telugu, Tamil, etc.).
