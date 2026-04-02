```
python -m venv venv
venv\Scripts\activate
```

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

```
pip install transformers timm datasets accelerate
```

```
pip install numpy Pillow scikit-learn matplotlib jupyter
```

```
python src/predict.py --images path/to/folder --checkpoint outputs/vit_base_patch16_224_attention/checkpoints/best_model.pt
```