import os, glob, random, numpy as np, cv2, tensorflow as tf
from tensorflow.keras import layers, models

# CONFIG
IMG_SIZE = 256
BATCH = 8
EPOCHS = 6
PAYLOAD = 0.15
DATA_DIR = "data/natural"   # relative to project root
SAVE_PATH = "backend/models/steg_cnn.h5"

# utils
def load_gray(p):
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Failed to read "+p)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return img

def lsb_embed(img, rate=0.15, rng=None):
    h,w = img.shape
    n = int(h*w*rate)
    rng = np.random.default_rng() if rng is None else rng
    out = img.copy()
    idx = rng.choice(h*w, size=n, replace=False)
    r = idx // w; c = idx % w
    bits = rng.integers(0,2,size=n, dtype=np.uint8)
    out[r, c] = (out[r, c] & 0xFE) | bits
    return out

HPF = np.array([[0,0,-1,0,0],
                [0,-1,2,-1,0],
                [-1,2,-4,2,-1],
                [0,-1,2,-1,0],
                [0,0,-1,0,0]], dtype=np.float32)

def build_model():
    inp = layers.Input((IMG_SIZE, IMG_SIZE, 1))
    # fixed HPF conv
    hpf = layers.Conv2D(1, 5, padding="same", use_bias=False, trainable=False)(inp)
    # set weights after build
    model_tmp = models.Model(inp, hpf)
    model_tmp.get_layer(index=1).set_weights([HPF[:,:,None,None]])
    x = hpf
    x = layers.BatchNormalization()(x)
    x = layers.Activation("tanh")(x)
    for f in [16,32,64]:
        x = layers.Conv2D(f, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.AveragePooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = models.Model(inp, out)
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m

# prepare paths
paths = glob.glob(os.path.join(DATA_DIR, "*"))
if len(paths) < 10:
    print("Please add at least 10 images into", DATA_DIR)
    raise SystemExit(1)

def sample_pair():
    p = random.choice(paths)
    cov = load_gray(p)
    ste = lsb_embed(cov, PAYLOAD)
    if random.random() < 0.5:
        img, y = cov, 0
    else:
        img, y = ste, 1
    img = img.astype(np.float32)/127.5 - 1.0
    img = np.expand_dims(img, -1)
    return img, np.array(y, dtype=np.int32)

def generator(batch=BATCH):
    while True:
        xs=[]; ys=[]
        for _ in range(batch):
            x,y = sample_pair()
            xs.append(x); ys.append(y)
        yield np.stack(xs, axis=0), np.stack(ys, axis=0)

model = build_model()
steps = 200
model.fit(generator(), steps_per_epoch=steps, epochs=EPOCHS)
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
model.save(SAVE_PATH)
print("Saved model to", SAVE_PATH)
