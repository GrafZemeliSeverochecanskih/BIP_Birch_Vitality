python -m venv venv
venv\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install transformers timm datasets accelerate

pip install numpy Pillow scikit-learn matplotlib jupyter