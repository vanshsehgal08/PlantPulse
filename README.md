# 🌿 Plant Pulse - AI-Powered Plant Disease Detection & Management

<div align="center">

![Plant Pulse](https://img.shields.io/badge/Plant%20Pulse-AI%20Powered-green)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Advanced machine learning technology for accurate disease identification, expert guidance, and actionable insights.**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Project Structure](#-project-structure) • [Technologies](#-technologies)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Screenshots & Capabilities](#-screenshots--capabilities)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Model Information](#-model-information)
- [Features in Detail](#-features-in-detail)
- [Development](#-development)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

**Plant Pulse** is an intelligent plant disease detection and management system that leverages cutting-edge AI technology to help farmers, gardeners, and agricultural professionals identify plant diseases from leaf images. The application provides instant, accurate disease detection with visual explanations, expert guidance, interactive analytics, and cost estimation tools.

### Key Highlights

- 🧠 **Deep Learning Models**: Trained on thousands of plant disease images across multiple crop types
- ⚡ **Real-time Detection**: Get instant predictions with confidence scores
- 🔍 **Visual Explanations**: Grad-CAM heatmaps show exactly where the AI detects disease indicators
- 💬 **AI Expert Consultation**: Ask questions and get expert advice powered by Google Gemini
- 📊 **Interactive Analytics**: Track disease patterns, trends, and confidence over time
- 💰 **Cost Calculator**: Estimate treatment costs based on detected diseases
- 🎨 **Modern UI**: Beautiful, responsive interface with dark theme and smooth animations

---

## ✨ Features

### 🔬 Smart Disease Detection
- Upload multiple leaf images for batch processing
- Live camera capture support for instant on-site detection
- Support for JPG, JPEG, and PNG formats
- Real-time AI-powered disease identification
- Top 3 disease candidates with confidence percentages
- Detailed confidence distribution charts

### 🎨 AI Visual Insights (Grad-CAM)
- Visual heatmaps for every uploaded image
- Red-highlighted regions showing disease indicators
- Side-by-side comparison of original and AI-processed images
- Transparent AI explanations for each detection

### 💬 Expert Chat Assistant
- AI-powered consultation using Google Gemini
- Context-aware responses based on detected diseases
- Questions about causes, symptoms, treatments, and prevention
- Conversation history tracking
- Quick action buttons for common questions

### 📈 Interactive Analytics Dashboard
- **Disease Frequency Chart**: Bar chart showing frequency of each detected disease
- **Average Confidence by Disease**: Visualize model confidence across different diseases
- **Confidence Trends**: Line chart tracking confidence over time
- **Disease Distribution Analysis**: Scatter plot showing disease patterns
- **Summary Statistics**: Total analyses, unique diseases, average/highest confidence

### 💰 Treatment Cost Calculator
- Automatic disease detection from recent predictions
- Manual disease selection for planning
- Multiple treatment types: Organic, Chemical, Integrated
- Flexible scale options: Acres, Plants, Square Meters, Hectares
- Customizable parameters: Labor costs, material cost multipliers
- Cost breakdown visualization (Material vs Labor)
- Treatment recommendations based on detected disease

### 📚 Analysis History
- Track all previous predictions
- View detection history with confidence scores
- Export results to CSV format
- Easy access to past analyses

---

## 🖼️ Screenshots & Capabilities

### Supported Plant Types
The model is trained to detect diseases across multiple plant categories:

- **Fruits**: Apple, Blueberry, Cherry, Grape, Orange, Peach, Raspberry, Strawberry
- **Vegetables**: Corn (Maize), Pepper (Bell), Potato, Squash, Tomato
- **Legumes**: Soybean

### Disease Detection Coverage
Over **38 disease classes** including:
- Fungal diseases (Scab, Rust, Mildew, Blight)
- Bacterial diseases (Bacterial Spot, Citrus Greening)
- Viral diseases (Mosaic Virus, Leaf Curl Virus)
- Pest-related issues (Spider Mites)
- Healthy plant identification

---

## 🚀 Installation

### Prerequisites

- **Python 3.8** or higher
- **pip** package manager
- **Git** (optional, for cloning)

### Step-by-Step Installation

1. **Clone or download the repository**:
   ```bash
   git clone https://github.com/yourusername/PlantPulse.git
   cd PlantPulse
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   # Windows
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # macOS/Linux
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify model files**:
   - Ensure `best_model.keras` or `my_model_24.keras` exists in:
     - `models/` directory (project root), OR
     - `frontend/models/` directory

5. **Run the application**:
   ```bash
   streamlit run frontend/app.py
   ```

6. **Access the application**:
   - The app will open automatically in your default browser
   - Default URL: `http://localhost:8501`


## 📖 Usage Guide

### Disease Prediction

1. **Navigate to Predict Page**: Click "Disease Prediction" from the home page or use the sidebar

2. **Upload Images**:
   - **File Upload**: Drag and drop or click to browse for leaf images
   - **Camera Capture**: Use the live camera feature to capture images directly
   - Supports multiple images for batch processing

3. **Run Analysis**: Click "Run AI Analysis" button

4. **View Results**:
   - Disease detection results table with confidence scores
   - Top 3 candidate diseases for selected image
   - Interactive confidence distribution chart
   - Grad-CAM visual insights for all uploaded images
   - Expert guidance on treatment and prevention

5. **Export Results**: Download results as CSV for record-keeping

### Interactive Charts

1. Navigate to "Interactive Charts" tab in Predict page
2. View comprehensive analytics:
   - Disease frequency across all analyses
   - Average confidence trends
   - Time-series confidence tracking
   - Distribution analysis
3. Use interactive features:
   - Hover for detailed tooltips
   - Zoom and pan on charts
   - Explore patterns in your data

### Treatment Cost Calculator

1. Navigate to "Cost Calculator" tab in Predict page
2. Select disease (automatically uses recent detection or manual selection)
3. Configure parameters:
   - Treatment type (Organic/Chemical/Integrated)
   - Scale (Acres/Plants/Square Meters/Hectares)
   - Labor cost per hour
   - Material cost multiplier
4. View cost breakdown and recommendations

### Expert Chat

1. Navigate to Chat page
2. Ask questions about:
   - Disease symptoms and identification
   - Treatment methods and options
   - Prevention strategies
   - Plant care and management
3. View conversation history
4. Use quick action buttons for common questions

---

## 📁 Project Structure

```
PlantPulse/
├── frontend/
│   ├── app.py                 # Main Streamlit application (Home page)
│   ├── pages/
│   │   ├── 1_Predict.py       # Disease prediction page
│   │   └── 2_Chat.py         # Expert chat assistant page
│   ├── utils/
│   │   ├── model.py          # Model loading and image preprocessing
│   │   ├── gradcam.py        # Grad-CAM visualization implementation
│   │   └── guidance.py      # Expert guidance data for diseases
│   ├── models/
│   │   └── my_model_24.keras # Trained deep learning model
│   └── assets/
│       └── styles.css        # Custom CSS styling
├── models/                    # Alternative model location
│   ├── best_model.keras
│   └── my_model_24.keras
├── data/                      # Training data (not included in repo)
│   ├── train/                # Training dataset
│   └── valid/                # Validation dataset
├── requirements.txt           # Python dependencies
├── render.yaml               # Render deployment configuration
├── DEPLOYMENT.md             # Deployment guide
├── RENDER_CHECKLIST.md       # Pre-deployment checklist
└── README.md                 # This file
```

---

## 🔧 Technologies Used

### Core Framework
- **Streamlit** - Web application framework for Python
- **Python 3.8+** - Programming language

### Machine Learning & AI
- **TensorFlow** - Deep learning framework
- **Keras** - High-level neural network API
- **NumPy** - Numerical computing
- **scikit-learn** - Machine learning utilities

### Computer Vision & Image Processing
- **OpenCV** - Image processing and computer vision
- **Pillow (PIL)** - Image manipulation
- **Matplotlib** - Visualization and Grad-CAM heatmaps

### Data Handling & Visualization
- **Pandas** - Data manipulation and analysis
- **Altair** - Interactive data visualization
- **NumPy** - Array operations

### AI Chat
- **Google Generative AI (Gemini)** - Large language model for expert consultation

### UI/UX
- **Custom CSS** - Modern dark theme with animations
- **Streamlit Components** - Camera input, file uploaders, charts

---

## 🧠 Model Information

### Architecture
- **Type**: Deep Convolutional Neural Network (CNN)
- **Framework**: TensorFlow/Keras
- **Input Size**: 128x128 RGB images
- **Output**: Multi-class classification (38+ disease classes)

### Training Data
- Trained on thousands of plant disease images
- Multiple crop types and disease categories
- Data augmentation for improved generalization
- Validation split for model evaluation

### Performance
- High accuracy on diverse plant disease datasets
- Confidence scores for each prediction
- Top-3 candidate ranking for ambiguous cases

### Model Files
- Primary: `best_model.keras` or `my_model_24.keras`
- Location: `models/` or `frontend/models/` directory
- Size: ~90 MB (use Git LFS for repository storage)

---

## 🎯 Features in Detail

### 1. Smart Disease Detection

The core feature uses a deep learning model trained on extensive plant disease datasets. When you upload a leaf image:

- **Image Preprocessing**: Automatic resizing, normalization, and batch preparation
- **Model Inference**: Real-time prediction using optimized TensorFlow model
- **Result Interpretation**: Top diseases ranked by confidence with human-readable labels
- **Batch Processing**: Handle multiple images simultaneously

### 2. Grad-CAM Visual Insights

Gradient-weighted Class Activation Mapping (Grad-CAM) provides visual explanations:

- **Heatmap Generation**: Highlights regions where the AI detected disease indicators
- **Color Coding**: Red areas indicate high disease probability regions
- **Transparency Overlay**: Original image overlaid with heatmap for easy comparison
- **Per-Image Analysis**: Individual heatmaps for each uploaded image

### 3. Interactive Analytics

Comprehensive data visualization tools:

- **Real-time Updates**: Charts update as you run more predictions
- **Interactive Elements**: Hover tooltips, zoom, pan capabilities
- **Multi-View Analysis**: Frequency, trends, distributions, and summaries
- **Export Capabilities**: Download analysis history as CSV

### 4. Treatment Cost Calculator

Practical tool for treatment planning:

- **Disease-Specific Pricing**: Different costs for organic, chemical, and integrated approaches
- **Scalable Calculations**: Works for small gardens to large farms
- **Labor Cost Integration**: Includes time estimates and hourly rates
- **Visual Breakdown**: Pie charts showing material vs labor costs
- **Recommendations**: Expert guidance tailored to selected disease

### 5. Expert Chat Assistant

AI-powered consultation system:

- **Context Awareness**: Uses detected diseases to provide relevant advice
- **Comprehensive Knowledge**: Covers identification, treatment, and prevention
- **Conversation Memory**: Maintains context across questions
- **Quick Actions**: Pre-built questions for common scenarios

---
