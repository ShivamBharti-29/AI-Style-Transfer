# 🎨 AI Neural Style Transfer

An AI-powered web application that transforms ordinary images into artistic masterpieces using **Neural Style Transfer**, **VGG-19**, and **Adaptive Instance Normalization (AdaIN)**. Users can upload a content image and a style image, adjust the style intensity, and generate high-quality stylized artwork through an intuitive Flask-based interface.

---

# 🚀 Features

* 🖼️ **Neural Style Transfer**

  * Transfers artistic styles from one image to another using Adaptive Instance Normalization (AdaIN).

* 🎨 **Custom Style Strength Control**

  * Adjust style intensity using an interactive slider.

* 📤 **Image Upload**

  * Upload custom content and style images directly from the browser.

* ⚡ **Fast Image Generation**

  * Generates stylized images using a pretrained VGG-19 encoder and a custom-trained decoder.

* 📥 **Download Results**

  * Download generated artwork in a single click.

* 🖼️ **Artwork Gallery**

  * View previously generated images in a gallery.
  * Delete unwanted generated images.

* 📱 **Responsive User Interface**

  * Modern Bootstrap-based UI with image previews and interactive controls.

* 🧠 **Example Showcase**

  * Includes sample content, style, and generated images for quick demonstration.

---

# 🛠️ Tech Stack

### Frontend

* HTML5
* CSS3
* Bootstrap 5
* JavaScript

### Backend

* Python
* Flask
* Flask-WTF

### Deep Learning

* PyTorch
* TorchVision
* VGG-19
* Adaptive Instance Normalization (AdaIN)

### Image Processing

* Pillow (PIL)

### Utilities

* NumPy
* tqdm
* Gunicorn

---

# 📂 Project Structure

```text
AI-Style-Transfer/
│
├── content_data/
├── style_data/
├── examples/
├── experiment/
├── models/
│   └── decoder_2.pth
├── static/
│   ├── uploads/
│   └── outputs/
├── style_data/
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── gallery.html
│   ├── features.html
│   ├── contact.html
│   ├── pricing.html
│   └── signin.html
├── utils/
│   ├── models.py
│   └── utils.py
├── app.py
├── train.py
├── requirements.txt
├── vgg_normalised.pth
└── README.md
```

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/ShivamBharti-29/AI-Style-Transfer.git
cd AI-Style-Transfer
```

---

## 2. Create a virtual environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Download Model Weights

Place the following pretrained model files inside the project directory.

```text
models/
└── decoder_2.pth

vgg_normalised.pth
```

---

## 5. Run the application

```bash
python app.py
```

The application will start on:

```text
http://127.0.0.1:5000
```

---

# 🧠 Model Training

Train the decoder network using:

```bash
python train.py
```

The training pipeline includes:

* Content image preprocessing
* Style image preprocessing
* Adaptive Instance Normalization (AdaIN)
* Content Loss
* Style Loss
* Adam Optimizer
* Learning Rate Scheduler
* Model Checkpoint Saving
* Output Image Visualization

---

# 📸 Application Workflow

1. Upload a content image.
2. Upload a style image.
3. Adjust the style intensity.
4. Click **Transfer Style**.
5. VGG-19 extracts content and style features.
6. AdaIN blends both feature representations.
7. Decoder reconstructs the stylized image.
8. Download the generated artwork or view it in the gallery.

---

# 🧠 Deep Learning Pipeline

```text
Content Image          Style Image
       │                     │
       ▼                     ▼
     VGG-19 Encoder     VGG-19 Encoder
             │
             ▼
Adaptive Instance Normalization (AdaIN)
             │
             ▼
      Custom Decoder Network
             │
             ▼
      Stylized Output Image
```

---

# 📦 Dependencies

* Python
* Flask
* Flask-WTF
* PyTorch
* TorchVision
* Pillow
* NumPy
* tqdm
* Gunicorn

---

# 🎯 Future Improvements

* Multiple style blending
* Higher-resolution image generation
* Batch image processing
* User authentication
* Cloud storage for generated artwork
* GPU acceleration
* Mobile-responsive optimization
* Docker deployment
* Real-time video style transfer

---

# 👨‍💻 Author

**Shivam Bharti**

GitHub:
https://github.com/ShivamBharti-29

---

# 📄 License

This project is intended for educational and portfolio purposes.
