# VLM for Jetson Orin Nano

This repository contains a Vision-Language Model (VLM) interface built using Gradio and designed to run on the Jetson Orin Nano.

## Setup Instructions

### 1. Set up Jetson Orin Nano
To get started with the Jetson Orin Nano, follow the official NVIDIA setup guide:

- [Set up Jetson Orin Nano](https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit)

### 2. Install PyTorch
To install the appropriate version of PyTorch for Jetson Orin Nano, download the corresponding `.whl` file based on your Jetpack version from the following link:

- [Download PyTorch `.whl` file](https://developer.nvidia.com/embedded/downloads#?search=torch)

After downloading the correct `.whl` file for your system, install it using `pip`:

```bash
pip install /path/to/your/downloaded_file.whl
```

### 3. Clone the Repository
Clone this repository to your local machine or Jetson Orin Nano:

```bash
git clone https://github.com/your-username/VLM-Interface-for-Jetson-Orin-Nano.git
cd VLM-Interface-for-Jetson-Orin-Nano
```

### 4. Install Dependencies
To install the necessary dependencies, including Gradio, Pillow, PyTorch, and Hugging Face Transformers, run the following command:

```bash
pip install -r requirements.txt
```

### 5. Running the Interface
After installing the dependencies, you can run the Gradio interface for the VLM model:

```bash
python vlm_gradio_app.py
```
