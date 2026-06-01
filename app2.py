import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import gdown
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.applications.efficientnet import preprocess_input
# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="AI Hairstyle Recommendation",
    page_icon="✂️",
    layout="wide"
)

# =========================================
# LOAD MODEL
# =========================================
IMG_SIZE = (224, 224)
NUM_CLASSES = 3
NUM_FEATURES = 9
class_names = ['ovale', 'round', 'square']

# =========================================
# BUILD MODEL (HARUS SAMA DENGAN TRAINING)
# =========================================
def build_model():

    # IMAGE BRANCH
    image_input = tf.keras.Input(shape=(224, 224, 3))
    x = preprocess_input(image_input)

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # LANDMARK BRANCH
    landmark_input = tf.keras.Input(shape=(NUM_FEATURES,))
    g = tf.keras.layers.Dense(128, activation='relu')(landmark_input)
    g = tf.keras.layers.BatchNormalization()(g)
    g = tf.keras.layers.Dropout(0.3)(g)
    g = tf.keras.layers.Dense(128, activation='relu')(g)
    g = tf.keras.layers.Dropout(0.3)(g)
    g = tf.keras.layers.Dense(64, activation='relu')(g)

    # FUSION
    combined = tf.keras.layers.Concatenate()([x, g])
    combined = tf.keras.layers.BatchNormalization()(combined)
    combined = tf.keras.layers.Dense(256, activation='relu')(combined)
    combined = tf.keras.layers.Dropout(0.4)(combined)
    combined = tf.keras.layers.Dense(128, activation='relu')(combined)
    combined = tf.keras.layers.Dropout(0.3)(combined)

    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(combined)

    model = tf.keras.Model(inputs=[image_input, landmark_input], outputs=outputs)
    

    return model


# =========================================
# LOAD WEIGHTS (INI KUNCI FIX KAMU)
# =========================================
file_id = "1lEzpxmTFMRSVl0piMtWHW7L8XPrqS8ry"
model_path = "best_val_acc_run_1.weights.h5"
url = f"https://drive.google.com/uc?export=download&id={file_id}"

if not os.path.exists(model_path):
    gdown.download(url,model_path,quite=False)

@st.cache_resource
def load_model():
    model = build_model()
    model.load_weights(model_path)   # <<< INI PENTING
    return model

model = load_model()


# =========================================
# MEDIAPIPE SETUP
# =========================================
base_options = python.BaseOptions(
    model_asset_path='face_landmarker.task'
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    min_face_detection_confidence=0.5,
    num_faces=1
)

face_mesh = vision.FaceLandmarker.create_from_options(options)

# =========================================
# CONFIG
# =========================================
IMG_SIZE = 224
class_names = ['ovale', 'round', 'square']

LANDMARKS = {
    "left_cheek": 234,
    "right_cheek": 454,
    "chin": 152,
    "forehead": 10,
    "left_jaw": 172,
    "right_jaw": 397,
    "left_forehead": 127,
    "right_forehead": 356
}

# =========================================
# RECOMMENDATION SYSTEM (PASTE YOUR FULL DATA HERE)
# =========================================
recommendations = {
    "ovale":{
        "medium":{
            "medium":{
                "straight":[
                    {"name": "Bro Flow", "image": r"assets\ovale\medium-medium-straight\bro flow.png"},
                    {"name": "Classic Side Part", "image": r"assets\ovale\medium-medium-straight\classic side part.png"},
                    {"name": "Slick Back", "image": r"assets\ovale\medium-medium-straight\slick back.png"},
                ],
                "wavy":[
                    {"name": "Bro Flow Wavy", "image": r"assets\ovale\medium-medium-wavy\Bro Flow Wavy.png"},
                    {"name": "Curtain Cut", "image": r"assets\ovale\medium-medium-wavy\curtain.png"},
                    {"name": "Layered Waves", "image": r"assets\ovale\medium-medium-wavy\Layered waves.png"},
                ],
                "curly":[
                    {"name": "Curly Shag", "image": r"assets\ovale\medium-medium-curly\curly shag.png"},
                    {"name": "Layered Curlt Top", "image": r"assets\ovale\medium-medium-curly\layered curly top.png"},
                    {"name": "Medium Curlt Fringe", "image": r"assets\ovale\medium-medium-curly\medium curly fringe.png"},
                ]
            },
            "short":{
                "straight":[
                    {"name": "Crew Cut", "image": r"assets\ovale\medium-short-straight\crew cut.png"},
                    {"name": "French Crop", "image": r"assets\ovale\medium-short-straight\french crop.png"},
                    {"name": "Textured Cut", "image": r"assets\ovale\medium-short-straight\textured crop.png"},
                ],
                "wavy":[
                    {"name": "Messy Fringe", "image": r"assets\ovale\medium-short-wavy\messy fringe.png"},
                    {"name": "Short Wavy Quiff", "image": r"assets\ovale\medium-short-wavy\short wavy quiff.png"},
                    {"name": "Textured Cut", "image": r"assets\ovale\medium-short-wavy\textured cut.png"},
                ],
                "curly":[
                    {"name": "Curly Crop", "image": r"assets\ovale\medium-short-curly\Curly crop,.png"},
                    {"name": "Short Curly Fade", "image": r"assets\ovale\medium-short-curly\short Curly fade.png"},
                    {"name": "Textured Curly Cut", "image": r"assets\ovale\medium-short-curly\textured Curly cut.png"},
                ]
            }
        },
        "long":{
            "long":{
                "straight":[
                    {"name": "Layered Shoulder", "image": r"assets\ovale\long-long-straight\layered shoulder length.png"},
                    {"name": "Long Bro Flow Cut", "image": r"assets\ovale\long-long-straight\long bro flow.png"},
                    {"name": "Long Slick Back", "image": r"assets\ovale\long-long-straight\long slick back.png"},
                ],
                "wavy":[
                    {"name": "Flow Cut", "image": r"assets\ovale\long-long-wavy\flow cut.png"},
                    {"name": "Layered Surfer Cut", "image": r"assets\ovale\long-long-wavy\layered surfer cut.png"},
                    {"name": "Long Wavy Bro Flow", "image": r"assets\ovale\long-long-wavy\long wavy bro flow.png"},
                ],
                "curly":[
                    {"name": "Curly Flow Cut", "image": r"assets\ovale\long-long-curly\curly flow.png"},
                    {"name": "Curly Shoulder Crop", "image": r"assets\ovale\long-long-curly\curly shoulder length.png"},
                    {"name": "Long Curly Shag", "image": r"assets\ovale\long-long-curly\long curly shag.png"},
                ]
            },
            "medium":{
                "straight":[
                    {"name": "Brow Flow Cut", "image": r"assets\ovale\long-medium-straight\brow flow.png"},
                    {"name": "Classic Side Part", "image": r"assets\ovale\long-medium-straight\classic side part.png"},
                    {"name": "Slick Back", "image": r"assets\ovale\long-medium-straight\slick back.png"},
                ],
                "wavy":[
                    {"name": "Curtain Cut", "image": r"assets\ovale\long-medium-wavy\curtain cut.png"},
                    {"name": "Layered Waves", "image": r"assets\ovale\long-medium-wavy\layered waves.png"},
                    {"name": "Wavy Bro Flow Cut", "image": r"assets\ovale\long-medium-wavy\wavy bro flow cut.png"},
                ],
                "curly":[
                    {"name": "Curly Fringe", "image": r"assets\ovale\long-medium-curly\curly fringe.png"},
                    {"name": "Curly Shag", "image": r"assets\ovale\long-medium-curly\curly shag.png"},
                    {"name": "Layered Cut Top", "image": r"assets\ovale\long-medium-curly\layered curl top.png"},
                ]
            },
            "short":{
                "straight":[
                    {"name": "Crew Cut", "image": r"assets\ovale\long-short-straight\crew cut.png"},
                    {"name": "High Fade Crop", "image": r"assets\ovale\long-short-straight\high fade crop.png"},
                    {"name": "Ivy League Cut", "image": r"assets\ovale\long-short-straight\ivy league.png"},
                ],
                "wavy":[
                    {"name": "Messy Crop", "image": r"assets\ovale\long-short-wavy\messy crop.png"},
                    {"name": "Short Wavy Quiff", "image": r"assets\ovale\long-short-wavy\short wavy quiff.png"},
                    {"name": "Textured Crop", "image": r"assets\ovale\long-short-wavy\textured crop.png"},
                ],
                "curly":[
                    {"name": "Curly Crop", "image": r"assets\ovale\long-short-curly\curly crop.png"},
                    {"name": "Curly Fade", "image": r"assets\ovale\long-short-curly\curly fade.png"},
                    {"name": "Curly Fringe", "image": r"assets\ovale\long-short-curly\curly fringe.png"},
                ]
            }
        },
    },
    "round":{
        "medium":{
            "medium":{
                "straight":[
                    {"name": "layered quiff", "image": r"assets\round\medium-medium-straight\layered quiff.png"},
                    {"name": "Medium Pompadour", "image": r"assets\round\medium-medium-straight\medium pompadour.png"},
                    {"name": "Side Swept Undercut", "image": r"assets\round\medium-medium-straight\side swept undercut.png"},
                ],
                "wavy":[
                    {"name": "Textured Curtain", "image": r"assets\round\medium-medium-wavy\textured curtain.png"},
                    {"name": "Voluminous Side Swept", "image": r"assets\round\medium-medium-wavy\voluminous side swept.png"},
                    {"name": "Wavy Qiff", "image": r"assets\round\medium-medium-wavy\wavy quiff.png"},
                ],
                "curly":[
                    {"name": "Curl High Volume Top", "image": r"assets\round\medium-medium-curly\curly high volume top.png"},
                    {"name": "Curly Layered Cut", "image": r"assets\round\medium-medium-curly\curly layered cut.png"},
                    {"name": "Curly Side Sweep", "image": r"assets\round\medium-medium-curly\curly side sweep.png"},
                ]
            },
            "short":{
                "straight":[
                    {"name": "High Fade Quiff", "image": r"assets\round\medium-short-straight\high fade quiff.png"},
                    {"name": "Pompadour Fade", "image": r"assets\round\medium-short-straight\pompadour fade.png"},
                    {"name": "Side Part Fade", "image": r"assets\round\medium-short-straight\side part fade.png"},
                ],
                "wavy":[
                    {"name": "Faux Hawk Fade", "image": r"assets\round\medium-short-wavy\faux hawk fade.png"},
                    {"name": "High Volume Quiff", "image": r"assets\round\medium-short-wavy\high volume quiff.png"},
                    {"name": "Textured Pompadour", "image": r"assets\round\medium-short-wavy\textured pompadour.png"},
                ],
                "curly":[
                    {"name": "Curly Faux Hawk", "image": r"assets\round\medium-short-curly\curly faux hawk.png"},
                    {"name": "Curly High Top", "image": r"assets\round\medium-short-curly\curly high top.png"},
                    {"name": "Curly Undercut", "image": r"assets\round\medium-short-curly\curly undercut.png"},
                ]
            }
        },
        "long":{
            "long":{
                "straight":[
                    {"name": "Long Pompadour Flow", "image": r"assets\round\long-long-straight\long pompadour flow.jpeg"},
                    {"name": "Long Side Sweep", "image": r"assets\round\long-long-straight\long side sweep.jpeg"},
                    {"name": "Layered Flow Cut", "image": r"assets\round\long-long-straight\layered flow cut.jpeg"},
                ],
                "wavy":[
                    {"name": "Flow Cut", "image": r"assets\round\long-long-wavy\flow cut.jpeg"},
                    {"name": "Layered Surfer Cut", "image": r"assets\round\long-long-wavy\layered surfer cut.jpeg"},
                    {"name": "Long Textured Waves", "image": r"assets\round\long-long-wavy\long textured waves.jpeg"},
                ],
                "curly":[
                    {"name": "Curly Side Sweep Long", "image": r"assets\round\long-long-curly\curly side sweep long.jpeg"},
                    {"name": "Curly Volume Flow", "image": r"assets\round\long-long-curly\curly volume flow.jpeg"},
                    {"name": "Long Layered Curl", "image": r"assets\round\long-long-curly\long layered curl.jpeg"},
                ]
            },
            "medium":{
                "straight":[
                    {"name": "Layered Pompadour", "image": r"assets\round\long-medium-straight\layered pompadour.jpeg"},
                    {"name": "Medium Quiff", "image": r"assets\round\long-medium-straight\medium quiff.jpeg"},
                    {"name": "Side Swept Undercut", "image": r"assets\round\long-medium-straight\side swept undercut.jpeg"},
                ],
                "wavy":[
                    {"name": "Textured Curtain", "image": r"assets\round\long-medium-wavy\textured curtain.jpeg"},
                    {"name": "Voluminius Side Sweep", "image": r"assets\round\long-medium-wavy\voluminous side sweep.jpeg"},
                    {"name": "Wavy Quiff", "image": r"assets\round\long-medium-wavy\wavy quiff.jpeg"},
                ],
                "curly":[
                    {"name": "Curly Layered Cut", "image": r"assets\round\long-medium-curly\curly layered cut.jpeg"},
                    {"name": "Curly Side Sweep", "image": r"assets\round\long-medium-curly\curly side sweep.jpeg"},
                    {"name": "Curl Volume Top", "image": r"assets\round\long-medium-curly\curly volume top.jpeg"},
                ]
            },
            "short":{
                "straight":[
                    {"name": "High Fade Quiff", "image": r"assets\round\long-short-straight\high fade quiff.png"},
                    {"name": "Pompadour Fade", "image": r"assets\round\long-short-straight\pompadour fade.png"},
                    {"name": "Side Part Fade", "image": r"assets\round\long-short-straight\side part fade.png"},
                ],
                "wavy":[
                    {"name": "Faux Hawk Fade", "image": r"assets\round\long-short-wavy\faux hawk fade.png"},
                    {"name": "High Volume Quiff", "image": r"assets\round\long-short-wavy\high volume quiff.png"},
                    {"name": "Wavy Pompadour", "image": r"assets\round\long-short-wavy\wavy pompadour.png"},
                ],
                "curly":[
                    {"name": "Curly High Top", "image": r"assets\round\long-short-curly\curly high top.png"},
                    {"name": "Curly Faux Hawk", "image": r"assets\round\long-short-curly\curly faux hawk.png"},
                    {"name": "Curly Undercut", "image": r"assets\round\long-short-curly\curly undercut.png"},
                ]
            }
            }
    },
    "square":{
        "medium":{
            "medium":{
                "straight":[
                    {"name": "Comb Over Fade", "image": r"assets\square\medium-medium-straight\comb over fade.jpeg"},
                    {"name": "Medium Brush Up", "image": r"assets\square\medium-medium-straight\medium brush up.jpeg"},
                    {"name": "Modern Side Part", "image": r"assets\square\medium-medium-straight\modern side part.jpeg"},
                ],
                "wavy":[
                    {"name": "Layered Sweep Back", "image": r"assets\square\medium-medium-wavy\layered sweep back.jpeg"},
                    {"name": "Medium Wavy Side Part", "image": r"assets\square\medium-medium-wavy\medium wavy side part.jpeg"},
                    {"name": "Wavy Brush Back", "image": r"assets\square\medium-medium-wavy\wavy brush back.jpeg"},
                ],
                "curly":[
                    {"name": "Curly Bro Flow", "image": r"assets\square\medium-medium-curly\curly bro flow.jpeg"},
                    {"name": "Curly Brush Back", "image": r"assets\square\medium-medium-curly\curly brush back.jpeg"},
                    {"name": "Medium Curly Crop", "image": r"assets\square\medium-medium-curly\medium curly crop.jpeg"},
                ]
            },
            "short":{
                "straight":[
                    {"name": "Buzz Cut Fade", "image": r"assets\square\medium-short-straight\buzz cut fade.jpeg"},
                    {"name": "Crew Cut", "image": r"assets\square\medium-short-straight\crew cut.jpeg"},
                    {"name": "Military High Fade", "image": r"assets\square\medium-short-straight\military high fade.jpeg"},
                ],
                "wavy":[
                    {"name": "Short Wavy Crew Cut", "image": r"assets\square\medium-short-wavy\short wavy crew cut.jpeg"},
                    {"name": "Taper Fade Waves", "image": r"assets\square\medium-short-wavy\taper fade waves.jpeg"},
                    {"name": "Textured Side Part", "image": r"assets\square\medium-short-wavy\textured side part.jpeg"},
                ],
                "curly":[
                    {"name": "Curly Caesar Cut", "image": r"assets\square\medium-short-curly\curly caesar cut.jpeg"},
                    {"name": "Curly Crew Cut", "image": r"assets\square\medium-short-curly\curly crew cut.jpeg"},
                    {"name": "Short Curly Fade", "image": r"assets\square\medium-short-curly\short curly fade.jpeg"},
                ]
            }
        },
        "long":{
            "long":{
                "straight":[
                    {"name": "Long Brush Back", "image": r"assets\square\long-long-straight\long brush back.jpeg"},
                    {"name": "Long Side Part", "image": r"assets\square\long-long-straight\long side part.jpeg"},
                    {"name": "Viking Style Layer", "image": r"assets\square\long-long-straight\viking style layer.jpeg"},
                ],
                "wavy":[
                    {"name": "Long Wavy Brush Back", "image": r"assets\square\long-long-wavy\long wavy brush back.jpeg"},
                    {"name": "Shoulder Length", "image": r"assets\square\long-long-wavy\shoulder length.jpeg"},
                    {"name": "Viking Waves", "image": r"assets\square\long-long-wavy\viking waves.jpeg"},
                ],
                "curly":[
                    {"name": "Curly Viking Style", "image": r"assets\square\long-long-curly\curly viking style.jpeg"},
                    {"name": "Layered Long Curl", "image": r"assets\square\long-long-curly\layered long curl.jpeg"},
                    {"name": "Long Curly Bro Flow", "image": r"assets\square\long-long-curly\long curly bro flow.jpeg"},
                ]
            },
            "medium":{
                "straight":[
                    {"name": "Brush Back", "image": r"assets\square\long-medium-straight\brush back.jpeg"},
                    {"name": "Comb Over Fade", "image": r"assets\square\long-medium-straight\comb over fade.jpeg"},
                    {"name": "Modern Side Part", "image": r"assets\square\long-medium-straight\modern side part.jpeg"},
                ],
                "wavy":[
                    {"name": "Layered Sweep Back", "image": r"assets\square\long-medium-wavy\layered sweep back.jpeg"},
                    {"name": "Medium Wavy Side Part", "image": r"assets\square\long-medium-wavy\medium wavy side part.jpeg"},
                    {"name": "Wavy Brush Back", "image": r"assets\square\long-medium-wavy\wavy brush back.jpeg"},
                ],
                "curly":[
                    {"name": "Curly Bro Flow", "image": r"assets\square\long-medium-curly\curly bro flow.jpeg"},
                    {"name": "Curly Brush Back", "image": r"assets\square\long-medium-curly\curly brush back.jpeg"},
                    {"name": "Medium Curly Crop", "image": r"assets\square\long-medium-curly\medium curly crop.jpeg"},
                ]
            },
            "short":{
                "straight":[
                    {"name": "High Fade Crop", "image": r"assets\square\long-short-straight\high fade crop.png"},
                    {"name": "Military Cut", "image": r"assets\square\long-short-straight\military cut.jpeg"},
                    {"name": "Buzz Cut Fade", "image": r"assets\square\long-short-straight\buzz cut fade.jpeg"},
                ],
                "wavy":[
                    {"name": "Short Wavy Fade", "image": r"assets\square\long-short-wavy\short wavy fade.jpeg"},
                    {"name": "Taper Fade Waves", "image": r"assets\square\long-short-wavy\taper fade waves.jpeg"},
                    {"name": "Textured Side Part", "image": r"assets\square\long-short-wavy\textured side part.jpeg"},
                ],
                "curly":[
                    {"name": "Curly Caesar Cut", "image": r"assets\square\long-short-curly\curly caesar.jpeg"},
                    {"name": "Curly Crew Cut", "image": r"assets\square\long-short-curly\curly crew cut.jpeg"},
                    {"name": "Curly Fade", "image": r"assets\square\long-short-curly\curly fade.jpeg"},
                ]
            }
        }
    },
}

# =========================================
# PREPROCESS IMAGE
# =========================================
def preprocess_image(image):
    image = np.array(image)

    rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    resized = cv2.resize(gray_3ch, (224, 224))
    resized = resized.astype("float32")

    return np.expand_dims(resized, axis=0)

# =========================================
# LANDMARK FEATURE EXTRACTION
# =========================================
def get_point(landmarks, idx):
    lm = landmarks[idx]
    return np.array([lm.x, lm.y])

def safe_div(a, b):
    return a / b if b != 0 else 0

def extract_features(image):
    image_np = np.array(image)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    )

    result = face_mesh.detect(mp_image)

    if not result.face_landmarks:
        return None

    face = result.face_landmarks[0]

    pts = {name: get_point(face, idx) for name, idx in LANDMARKS.items()}

    face_width = np.linalg.norm(pts["left_cheek"] - pts["right_cheek"])
    face_height = np.linalg.norm(pts["forehead"] - pts["chin"])
    jaw_width = np.linalg.norm(pts["left_jaw"] - pts["right_jaw"])

    ratio_wh = safe_div(face_width, face_height)
    ratio_jaw = safe_div(jaw_width, face_width)

    mid = (pts["left_cheek"] + pts["right_cheek"]) / 2
    upper = np.linalg.norm(pts["forehead"] - mid)
    lower = np.linalg.norm(mid - pts["chin"])

    ratio_ul = safe_div(upper, lower)

    forehead_width = np.linalg.norm(pts["left_forehead"] - pts["right_forehead"])
    ratio_fj = safe_div(forehead_width, jaw_width)

    features = np.array([
        face_width,
        face_height,
        jaw_width,
        ratio_wh,
        ratio_jaw,
        ratio_ul,
        forehead_width,
        ratio_fj,
        0.0
    ], dtype=np.float32)

    return np.expand_dims(features, axis=0)

# =========================================
# UI
# =========================================
st.title("✂️ AI Hairstyle Recommendation")
st.write("Multi-Input Face Shape Classification + Hairstyle Recommendation")

st.divider()

# INPUT STYLE
col1, col2, col3 = st.columns(3)

with col1:
    current_hair = st.selectbox("Current Hair Length", ["medium","long"])

if current_hair =="long":
    desired_options = ["short","medium","long"]
else:
    desired_options = ["short","medium"]
with col2:
    desired_hair = st.selectbox("Desired Hair Length", desired_options)

with col3:
    hair_type = st.selectbox("Hair Type", ["straight", "wavy", "curly"])

st.divider()

# INPUT IMAGE
input_method = st.radio("Input Method", ["Upload Image", "Camera"])

image = None

if input_method == "Upload Image":
    file = st.file_uploader("Upload", type=["jpg", "png", "jpeg"])
    if file:
        image = Image.open(file)
else:
    cam = st.camera_input("Camera")
    if cam:
        image = Image.open(cam)

# =========================================
# PREDICTION
# =========================================
if image is not None:

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Input Image")

    with col2:

        with st.spinner("Processing..."):

            img_input = preprocess_image(image)
            landmark_input = extract_features(image)

            if landmark_input is None:
                st.error("Face not detected")
                st.stop()

            pred = model.predict([img_input, landmark_input])[0]
            idx = np.argmax(pred)
            label = class_names[idx]

        st.success(f"Face Shape: {label.upper()}")

        for i, c in enumerate(class_names):
            st.write(f"{c}: {pred[i]:.2%}")
            st.progress(float(pred[i]))

    st.divider()

    # =========================================
    # RECOMMENDATION
    # =========================================
    st.subheader("Recommended Hairstyles")

    try:
        styles = recommendations[label][current_hair][desired_hair][hair_type]

        cols = st.columns(len(styles))

        for col, s in zip(cols, styles):
            with col:
                st.image(s["image"])
                st.markdown(f"**{s['name']}**")

    except:
        st.warning("No recommendation for this combination.")