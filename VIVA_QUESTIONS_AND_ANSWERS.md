# Plant Species Detection System – Viva & Academic Defense Guide 🎓

This guide contains frequently asked questions, architectural explanations, and technical justifications for your **Diploma / Final Project Demonstration & Viva Exam**.

---

## 1. Project Overview & Objectives

### Q1: What is the main objective of your project?
> **Answer:** The objective is to automate the identification of plant species from digital photographs using computer vision (OpenCV) and deep learning (TensorFlow/Keras Convolutional Neural Networks). The system eliminates the need for manual botanical reference searches, outputs Latin binomial scientific names (*e.g., Rosa rubiginosa*), confidence scores, biological families, and cultivation/medicinal characteristics in real-time.

### Q2: Who are the target beneficiaries of this system?
> **Answer:** 
> 1. **Botany & Agriculture Students / Researchers:** For rapid taxonomic study and specimen indexing.
> 2. **Farmers & Agriculturalists:** For weed vs. crop identification and plant health awareness.
> 3. **Gardeners & Plant Enthusiasts:** For proper indoor and outdoor plant care guidance.

---

## 2. Architecture & Design Patterns

### Q3: What design pattern does Django follow?
> **Answer:** Django follows the **MVT (Model-View-Template)** architecture:
> * **Model (`models.py`):** Defines the database structure (`PlantSpecies`, `Detection`, `ContactMessage`).
> * **View (`views.py`):** Contains the business logic, handles image requests, invokes OpenCV preprocessing and ML inference, and formats context data.
> * **Template (`templates/`):** Presentation layer using HTML5, CSS3, JavaScript, and Bootstrap 5 with template inheritance (`base.html`).

### Q4: Explain the application ontology and data flow.
> **Answer:** 
> $$\text{User} \xrightarrow{\text{Uploads}} \text{Plant Image} \xrightarrow{\text{Preprocesses}} \text{OpenCV} \xrightarrow{\text{Tensor Batch}} \text{TensorFlow CNN} \xrightarrow{\text{Inference}} \text{Softmax Prediction} \xrightarrow{\text{Taxonomy Match}} \text{Result View \& DB Log}$$

---

## 3. Computer Vision & OpenCV Preprocessing

### Q5: Why is OpenCV used instead of simply passing the raw image to the neural network?
> **Answer:** 
> Raw images vary wildly in dimensions, color encodings, aspect ratios, and pixel intensity scales. OpenCV (`cv2`) standardizes the input data:
> 1. **Color Conversion (`cv2.cvtColor`):** Converts BGR format to standard RGB.
> 2. **Dimensional Standardisation (`cv2.resize`):** Resizes images to exactly $224 \times 224$ pixels using area interpolation.
> 3. **Pixel Normalization:** Divides pixel values by $255.0$ to scale intensity from $[0, 255]$ into $[0.0, 1.0]$, preventing gradient explosion and accelerating inference convergence.
> 4. **Batch Dimension Expansion (`np.expand_dims`):** Expands the tensor shape from $(224, 224, 3)$ to $(1, 224, 224, 3)$ matching Keras batch tensor input expectations.

---

## 4. Machine Learning & Neural Networks

### Q6: What neural network model is used and why?
> **Answer:** The system uses **Transfer Learning with MobileNetV2 / Deep Convolutional Neural Networks (CNN)**. MobileNetV2 uses depthwise separable convolutions, making it lightweight, memory-efficient, and exceptionally fast for real-time web inference without sacrificing classification accuracy.

### Q7: What is the purpose of the Softmax activation function?
> **Answer:** The output layer uses the **Softmax** activation function:
> $$\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$
> It converts raw network logits into a normalized probability distribution across all $K$ plant classes where $\sum P_i = 1.0$. The class with the highest probability represents the predicted species, and its value multiplied by 100 yields the **Confidence Percentage**.

### Q8: How does the system handle alternative candidate predictions?
> **Answer:** By sorting the probability vector (`np.argsort(confidence_scores)[::-1]`), the system extracts the top 3-4 candidate species with their respective confidence scores, allowing users to inspect close taxonomic alternatives.

---

## 5. Database & Security

### Q9: How is the database organized?
> **Answer:** 
> * `PlantSpecies`: Stores common name, Latin scientific name, biological family, description, care guidelines, medicinal benefits, and reference imagery.
> * `Detection`: Foreign key link to `User` (or null for guest scans), path to uploaded photo, predicted common name, scientific name, confidence score, and timestamp.
> * `ContactMessage`: Persists user feedback and inquiries.

### Q10: What security mechanisms are implemented?
> **Answer:**
> 1. **CSRF Tokens (`{% csrf_token %}`):** Protects POST submissions against Cross-Site Request Forgery.
> 2. **File Validation:** Restricts uploads to permitted extensions (`.jpg`, `.jpeg`, `.png`) and caps file size at 10MB to prevent denial-of-service.
> 3. **Password Hashing:** Uses PBKDF2 with SHA-256 password hashing.
> 4. **SQL Injection Prevention:** Uses Django's parameterized Object Relational Mapper (ORM).
> 5. **User Privacy:** Authenticated users can only view and manage their personal detection history records.

---

## 6. Future Enhancements

### Q11: How can this system be expanded in real-world agricultural scenarios?
> **Answer:**
> 1. **Plant Disease Detection:** Extending the classifier to identify specific foliage leaf diseases (e.g. bacterial spot, powdery mildew).
> 2. **Edge Device Deployment:** Running the quantized TFLite model on Raspberry Pi or drone-mounted cameras for autonomous field crop scanning.
> 3. **Geo-tagging & Climate Correlation:** Storing GPS coordinates of scans to map plant biodiversity and invasive species spreading patterns.
