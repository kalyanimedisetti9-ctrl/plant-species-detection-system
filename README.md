# Plant Species Detection System 🌿

A complete, responsive, full-stack web application designed for automated **Plant Species Detection and Botanical Identification** using **Artificial Intelligence, OpenCV Image Processing, and Django**.

Developed as an academic **Diploma Project** demonstrating the integration of computer vision, deep convolutional neural networks, relational databases, and modern web interfaces.

---

## 🌟 Key Features

* **AI-Powered Image Recognition:** Powered by TensorFlow and Keras deep neural networks (MobileNetV2 transfer learning).
* **OpenCV Preprocessing Pipeline:** Real-time BGR-to-RGB conversion, target resizing (224×224), and pixel intensity normalization.
* **Botanical Taxonomy & Scientific Names:** Displays official Latin binomial nomenclature (*e.g., Rosa rubiginosa*), biological family, care guidelines, and medicinal/ecological properties.
* **Modern Dark Green & White UI:** Designed with an aesthetic dark green theme, subtle animations, and responsive cards.
* **Green Glow Interactive Buttons:** Dynamic glowing hover effects with smooth transitions on all primary actions.
* **Drag-and-Drop Image Upload:** Interactive dropzone with instant preview, format validation (`JPG`, `JPEG`, `PNG`), and loading spinner.
* **User Authentication & Dashboard:** Complete user registration, login, personal dashboard, and detection history logs with delete capabilities.
* **Botanical Species Catalog:** Searchable catalog with family filtering and detailed botanical profile views.
* **Custom Django Admin Panel:** Manage users, plant species catalog, detection history records, and contact inquiries with image thumbnails.
* **Dual Database Support:** Native MySQL configuration with zero-config fallback to SQLite.

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **Backend Framework** | Python 3, Django 6.x |
| **Machine Learning** | TensorFlow 2.x, Keras 3.x |
| **Computer Vision** | OpenCV (`cv2`), NumPy, Pillow |
| **Database** | MySQL (with PyMySQL) / SQLite fallback |
| **Frontend UI** | HTML5, CSS3, JavaScript (ES6+), Bootstrap 5, Bootstrap Icons |

---

## 🏗️ Project Architecture & Ontology

```text
User (Student / Farmer / Researcher)
  │
  └── uploads
        ↓
   Plant Image (JPG, JPEG, PNG)
        │
        └── processed by
                 ↓
        OpenCV Preprocessing (BGR→RGB, Resize 224x224, Normalize [0,1])
                 │
                 └── input to
                        ↓
        TensorFlow / Keras Model (MobileNetV2 / CNN Backbone)
                 │
                 └── generates
                        ↓
        Prediction (Softmax Probability Distribution & Top Candidates)
                 │
                 └── identifies
                        ↓
        Plant Species (Common Name, Latin Scientific Name, Family, Care)
                 │
                 └── displayed as
                        ↓
        Result Page & Saved Detection History
```

---

## 📁 Project Directory Structure

```text
PlantSpeciesDetectionSystem/
│
├── manage.py                   # Django management script
├── seed_database.py            # Automated database seeding & model setup
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── plant_detection/            # Django project core
│   ├── __init__.py             # PyMySQL connector hook
│   ├── settings.py             # App settings, DB config & media routing
│   ├── urls.py                 # Main URL router
│   ├── wsgi.py                 # WSGI production gateway
│   └── asgi.py                 # ASGI gateway
│
├── detection/                  # Main web application
│   ├── models.py               # PlantSpecies, Detection, ContactMessage
│   ├── views.py                # View handlers & prediction flow
│   ├── urls.py                 # Application routing
│   ├── forms.py                # Upload, Registration & Contact forms
│   ├── admin.py                # Customized Django admin interface
│   ├── preprocessing.py        # OpenCV image preprocessing module
│   ├── apps.py                 # App configuration
│   │
│   ├── templates/detection/    # Bootstrap 5 HTML templates
│   │   ├── base.html           # Master layout (Navbar, Footer, Modals)
│   │   ├── home.html           # Landing page with hero, features & CTA
│   │   ├── about.html          # Project objectives & tech stack
│   │   ├── upload.html         # Drag-and-drop detection upload interface
│   │   ├── result.html         # Prediction card & botanical data
│   │   ├── species.html        # Searchable species catalog
│   │   ├── species_detail.html # Comprehensive species profile
│   │   ├── how_it_works.html   # 6-step workflow & architecture
│   │   ├── contact.html        # Contact inquiry form
│   │   ├── login.html          # User login
│   │   ├── register.html       # User registration
│   │   ├── dashboard.html      # User control panel
│   │   └── history.html        # Personal detection history table
│   │
│   └── static/detection/       # Static assets
│       ├── css/style.css       # Dark green theme & glowing buttons
│       └── js/script.js        # Drag-and-drop, preview & loading overlay
│
├── ml_model/                   # Machine Learning Module
│   ├── dataset/                # Training / Testing image folders
│   │   ├── train/
│   │   └── test/
│   ├── training/
│   │   └── train_model.py      # TensorFlow model training script
│   ├── trained_model/
│   │   ├── plant_species_model.keras  # Saved neural network model
│   │   └── class_indices.json         # Class index mappings
│   └── prediction.py           # Inference engine & taxonomy resolver
│
└── media/                      # Dynamic user media
    ├── uploaded_images/        # User detection uploads
    └── species_images/         # Catalog reference images
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
* Python 3.10+ installed
* MySQL Server (Optional, SQLite is ready out of the box)

### 2. Clone / Open Directory
```bash
cd "plant species detection system"
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Database Setup & Initialization
Run the automated initialization script to apply database migrations, generate the deep learning model, and seed botanical species data:

```bash
python seed_database.py
```

### 5. Start the Development Server
```bash
python manage.py runserver
```

Open your browser and navigate to:
**`http://127.0.0.1:8000/`**

---

## 🔑 Default Credentials

### Administrator Account
* **URL:** `http://127.0.0.1:8000/admin/`
* **Username:** `admin`
* **Password:** `admin123`

### Demo Student Account
* **URL:** `http://127.0.0.1:8000/login/`
* **Username:** `student`
* **Password:** `student123`

---

## 🗄️ MySQL Database Setup (Optional)

To connect the application to a local MySQL server:

1. Create a MySQL database:
```sql
CREATE DATABASE plant_species_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

2. Set environment variables (or update `plant_detection/settings.py`):
```bash
# Windows PowerShell
$env:USE_MYSQL="True"
$env:DB_NAME="plant_species_db"
$env:DB_USER="root"
$env:DB_PASSWORD="your_password"
$env:DB_HOST="localhost"
$env:DB_PORT="3306"
```

3. Run migrations and seed data:
```bash
python seed_database.py
```

---

## 🧪 Model Training & Dataset Customization

To train or fine-tune the model with your custom plant dataset:

1. Place your dataset in `ml_model/dataset/train/<species_name>/` and `ml_model/dataset/test/<species_name>/`.
2. Run the training script:
```bash
python ml_model/training/train_model.py --epochs 25 --batch_size 32
```
3. The newly trained model will be automatically saved to `ml_model/trained_model/plant_species_model.keras` and picked up by the web application.

---

## 📜 License & Academic Attribution
Developed for academic submission and diploma demonstration.
All rights reserved © 2026.
