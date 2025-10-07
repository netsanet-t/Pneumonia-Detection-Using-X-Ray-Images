import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
# Import MobileNetV2 for a more realistic placeholder example for transfer learning
from tensorflow.keras.applications import MobileNetV2 
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# --- Constants ---
IMG_SIZE = 224
MODEL_PATH = "xray_classification.keras"
CORRECT_INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3) # Expected by the model (H, W, C=3)
CLASS_LABELS = ["Normal", "Pneumonia"]

# --- Quick Fix: Model Rebuilding Function (MUST BE EDITED) ---
# 🛑 CRITICAL: The weights failed to load because the architecture below does not match the 
# structure saved in 'xray_classification.keras'. You MUST replace the content of this 
# function with the *exact* architecture of your original model.
@st.cache_resource
def build_model_placeholder(input_shape):
    """
    Rebuilds the model structure with the correct input shape (3 channels).
    
    This is currently a MobileNetV2-based placeholder. Replace it with your actual model.
    """
    st.info("Rebuilding model structure to enforce 3-channel input... **(Please ensure this architecture matches your saved model!)**")
    
    # --- Example: Transfer Learning Architecture (REPLACE THIS SECTION!) ---
    # 1. Load the pre-trained base model (e.g., MobileNetV2, VGG16, etc.)
    # If your original model was based on a Keras Application, use it here.
    base_model = MobileNetV2(
        input_shape=input_shape,  # Correct input shape (224, 224, 3)
        include_top=False,        # Exclude the classifier layers from the original model
        weights=None              # IMPORTANT: Set weights to None because we load them later from the .keras file
    )
    
    # 2. Add your custom classification head on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model_rebuilt = Model(inputs=base_model.input, outputs=outputs)
    
    # 3. Compile with the same settings used during original training
    model_rebuilt.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    
    # --- END OF REPLACEABLE SECTION ---
    return model_rebuilt
    
# --- Model Loading Logic (Implementing the Quick Fix) ---
try:
    # 1. Attempt the Quick Fix: Rebuild model structure and load weights
    rebuilt_model = build_model_placeholder(CORRECT_INPUT_SHAPE)
    
    # Check if the weight file exists before attempting to load
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    # Load weights into the rebuilt structure
    # Added by_name=True to help match weights to layers even if the order changed slightly.
    rebuilt_model.load_weights(MODEL_PATH, skip_mismatch=False, by_name=True) 
    model = rebuilt_model
    st.success("✅ Model successfully loaded by rebuilding the 3-channel structure and loading weights by name.")

except FileNotFoundError as e:
    st.error(f"FATAL ERROR: {e}")
    st.stop()
except Exception as e:
    # If loading weights fails (likely due to wrong architecture definition or layer name mismatch)
    st.error(f"""
        🔴 FATAL MODEL ERROR DURING WEIGHT LOADING: {e}
        
        The model structure defined in `build_model_placeholder` does not exactly match 
        the weights saved in `{MODEL_PATH}`. This is the common cause of the weight mismatch error.
        
        **Action Required:** You must edit the function to perfectly mirror your original 
        CNN or Transfer Learning architecture, including all layers and their names.
    """)
    st.stop()


# --- Preprocessing function ---
def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Resizes the image to 224x224, converts to numpy array, normalizes,
    and ensures 3 channels before adding the batch dimension.
    """
    # 1. Resize the image
    img = image.resize((IMG_SIZE, IMG_SIZE))
    
    # 2. Convert to numpy array and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Check and enforce 3 channels (H, W, C)
    if img_array.ndim == 2:
        # Grayscale (H, W), reshape to (H, W, 1)
        img_array = np.expand_dims(img_array, axis=-1)

    if img_array.shape[-1] == 1:
        # If 1 channel, duplicate it three times for the 3-channel model input
        img_array = np.concatenate([img_array] * 3, axis=-1)
    
    # 3. Assert the final channel count is 3 and size is correct
    if img_array.shape != CORRECT_INPUT_SHAPE:
        st.error(f"Internal Preprocessing Error: Expected shape {CORRECT_INPUT_SHAPE}, got {img_array.shape}")
        st.stop()

    # 4. Add batch dimension (1, H, W, C)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# --- Streamlit UI ---
st.title("Pneumonia Detection from Chest X-ray")
st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload a Chest X-ray Image (JPG/PNG)", 
    type=["jpg", "jpeg", "png"],
    help="Upload an image file to check for Pneumonia."
)

if uploaded_file is not None:
    try:
        # Load the image and explicitly convert it to RGB (3 channels)
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded X-ray", use_column_width=True)

        # Preprocess the image
        with st.spinner('Preprocessing image and making prediction...'):
            img_array = preprocess_image(image)
            
            # Predict
            # model.predict returns an array like [[0.987]]
            prediction_result = model.predict(img_array)
            prediction = prediction_result[0][0]

        st.markdown("---")
        
        # Determine label and confidence
        if prediction > 0.5:
            label = CLASS_LABELS[1] # Pneumonia
            confidence = prediction
            color = "red"
            icon = "🦠"
        else:
            label = CLASS_LABELS[0] # Normal
            confidence = 1 - prediction
            color = "green"
            icon = "😌"
            
        st.subheader(f"{icon} Prediction: :{color}[{label}]")
        st.write(f"Confidence: **{confidence * 100:.2f}%**")

    except Exception as e:
        st.error(f"An unexpected error occurred during image processing or prediction: {e}")
        st.exception(e)
