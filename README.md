# 🧠 Neural Networks from Scratch — NumPy · PyTorch · TensorFlow

> **CMPE-256 | Deep Learning Assignment** · San José State University  
> **Due:** Sunday Feb 22, 2026 · **Points:** 200

---

## 🎬 Video Walkthroughs *(MOST IMPORTANT)*

| # | Colab | Description | Video |
|---|-------|-------------|-------|
| A | NumPy from Scratch | Manual backprop · tf.einsum · 4D plot | [▶ Watch](https://youtu.be/eSP1MW2Jxck) |
| B | PyTorch Scratch | Raw tensors · no nn.Linear | [▶ Watch](YOUR_YOUTUBE_LINK_B) |
| C | PyTorch nn.Module | Class-based · built-in layers | [▶ Watch](YOUR_YOUTUBE_LINK_C) |
| D | PyTorch Lightning | LightningModule · Trainer | [▶ Watch](YOUR_YOUTUBE_LINK_D) |
| E-i | TF Low-Level | tf.Variable · GradientTape | [▶ Watch](YOUR_YOUTUBE_LINK_E1) |
| E-ii | TF Built-in Layers | Dense layers · GradientTape loop | [▶ Watch](YOUR_YOUTUBE_LINK_E2) |
| E-iii | TF Functional API | Input→Dense graph · Model · fit | [▶ Watch](YOUR_YOUTUBE_LINK_E3) |
| E-iv | TF Sequential | Sequential · compile · fit | [▶ Watch](YOUR_YOUTUBE_LINK_E4) |

> **How to add your links:** Replace each `YOUR_YOUTUBE_LINK_X` above with your actual YouTube URL.  
> Example: `[▶ Watch](https://youtu.be/abc123xyz)`

---

## 📂 Repository Structure

```
neural-networks-from-scratch/
├── README.md
└── colabs/
    ├── a_numpy_3layer_neural_net.ipynb       ← NumPy only · tf.einsum · manual backprop
    ├── b_pytorch_from_scratch.ipynb          ← PyTorch raw tensors · no built-in layers
    ├── c_pytorch_nn_module.ipynb             ← PyTorch nn.Module · class-based
    ├── d_pytorch_lightning.ipynb             ← PyTorch Lightning · Trainer
    ├── e1_tensorflow_scratch.ipynb           ← TF low-level · no Keras
    ├── e2_tensorflow_builtin_layers.ipynb    ← TF Dense + GradientTape
    ├── e3_tensorflow_functional.ipynb        ← TF Functional API
    └── e4_tensorflow_sequential.ipynb        ← TF Sequential API
```

---


---

## 🧮 The Equation (All Colabs Use This)

```python
y = 2*x1**2 + 3*x2*x3 + np.sin(x1*x2) + 0.5*x3**2 + noise
```

**3 variables** x₁, x₂, x₃ ∈ [−2, 2] — extends the 2-variable class example.  
**4D visualization**: PCA (scikit-learn) reduces inputs 3→2 components → 3D scatter + color = y value.

---

## 🏗️ Architecture (Consistent Across All 8 Colabs)

```
Input(3) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(16, Tanh) → Output(1, Linear)
```

**Total parameters:** 4,785 · **Optimizer:** Adam lr=0.001 · **Loss:** MSE · **Epochs:** 1000

---

## 📋 Colab Descriptions

### Colab A — NumPy Only from Scratch
**File:** `a_numpy_3layer_neural_net.ipynb`

| Section | What it shows |
|---------|---------------|
| Data generation | 3-var equation, N=1000 samples |
| 4D plot | PCA → 3D scatter + color=y (4th dim) |
| `init_weights()` | He initialization: `scale = sqrt(2/fan_in)` |
| `forward_pass()` | **`tf.einsum('bi,ij->bj', X, W)`** — required by assignment |
| `backward_pass()` | Chain rule: `dZ3 = dA3*(1-tanh²(Z3))`, `dZ2 = dA2*(Z2>0)` |
| `AdamOptimizer` | m, v moments + bias correction + parameter update |
| Training loop | 1000 epochs, mini-batch=64 |
| Results | Loss curve (log scale) + predicted vs actual scatter |

---

### Colab B — PyTorch from Scratch (No Built-in Layers)
**File:** `b_pytorch_from_scratch.ipynb`

| Section | What it shows |
|---------|---------------|
| Parameters | `W1 = torch.randn(3,64).requires_grad_(True)` |
| `forward(X)` | `torch.relu(X @ W1 + b1)` — plain function, not a class |
| Autograd | `loss.backward()` — automatic gradients |
| Manual Adam | Update inside `torch.no_grad()`, then `p.grad.zero_()` |

---

### Colab C — PyTorch nn.Module
**File:** `c_pytorch_nn_module.ipynb`

| Section | What it shows |
|---------|---------------|
| `class NeuralNet(nn.Module)` | `__init__` + `forward` |
| Training loop | `zero_grad → forward → loss → backward → step` |
| `ReduceLROnPlateau` | LR halves when val_loss plateaus |
| Save/Load | `state_dict()` and `load_state_dict()` |

---

### Colab D — PyTorch Lightning
**File:** `d_pytorch_lightning.ipynb`

| Section | What it shows |
|---------|---------------|
| `RegressionDM` | `LightningDataModule` |
| `NeuralNetLightning` | `training_step`, `validation_step`, `configure_optimizers` |
| `Trainer` | `EarlyStopping`, `ModelCheckpoint` |
| `trainer.fit()` | Replaces entire manual training loop |

---

### Colab E-i — TensorFlow Low-Level
**File:** `e1_tensorflow_scratch.ipynb`

| Section | What it shows |
|---------|---------------|
| `tf.Variable` | Mutable weight tensors |
| `@tf.function` | JIT compilation speedup |
| `tf.GradientTape` | `tape.gradient(loss, tvars)` |
| Manual Adam | `var.assign_sub(lr * m_hat / (tf.sqrt(v_hat) + eps))` |

---

### Colab E-ii — TF Built-in Layers + GradientTape
**File:** `e2_tensorflow_builtin_layers.ipynb`

| Section | What it shows |
|---------|---------------|
| Layer objects | `Dense(64, 'relu', kernel_initializer='he_normal')` |
| `all_vars` | `l1.trainable_variables + l2.trainable_variables + ...` |
| GradientTape | `optimizer.apply_gradients(zip(grads, all_vars))` |

---

### Colab E-iii — TF Functional API
**File:** `e3_tensorflow_functional.ipynb`

| Section | What it shows |
|---------|---------------|
| Functional syntax | `Dense(64)(inputs)` — double parentheses |
| `Model(inputs, outputs)` | Graph finalized into model |
| `model.compile + fit` | Automatic training with callbacks |
| `EarlyStopping` | Stops + restores best weights |

---

### Colab E-iv — TF Sequential
**File:** `e4_tensorflow_sequential.ipynb`

| Section | What it shows |
|---------|---------------|
| `Sequential([...])` | All layers in one list |
| `compile + fit` | One-line training |
| 3 result plots | Loss curves + scatter + residual histogram |
| `ModelCheckpoint` | Auto-saves best weights |

---

## 📊 Expected Results

| Colab | Test MSE |
|-------|---------|
| A — NumPy Scratch | ~0.015 |
| B — PyTorch Scratch | ~0.012 |
| C — PyTorch Module | ~0.010 |
| D — Lightning | ~0.010 |
| E-i — TF Scratch | ~0.013 |
| E-ii — TF Built-in | ~0.010 |
| E-iii — TF Functional | ~0.010 |
| E-iv — TF Sequential | ~0.010 |

---

## 🛠️ Quick Setup

```bash
git clone https://github.com/YOUR_USERNAME/neural-networks-from-scratch
cd neural-networks-from-scratch
pip install numpy tensorflow torch pytorch-lightning scikit-learn matplotlib
jupyter notebook colabs/
```

---

*CMPE-256 Large Scale Data Analytics · San José State University*
