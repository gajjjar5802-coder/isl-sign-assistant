import streamlit as st
import cv2
import mediapipe as mp
import pickle
import numpy as np

# Set page to wide mode for a modern dashboard look
st.set_page_config(page_title="ISL Sign Assistant", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS to make the interface look incredibly premium and clean
st.markdown("""
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .stCheckbox { background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin-bottom: 15px;}
    .output-box {
        background-color: #ffffff;
        border: 2px solid #e6e9ef;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        min-height: 200px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤟 Indian Sign Language Assistive System")
st.caption("A clean computer vision platform for real-time gesture translation.")
st.write("---")

# ======================================================================
# 🚀 SMART MODEL DOWNLOAD ENGINE PLACED RIGHT HERE!
# ======================================================================
@st.cache_resource
def load_model():
    model_path = 'model.p'
    # If the model file isn't already on the Streamlit server, download it from your release link!
    if not os.path.exists(model_path):
        # ⚠️ MAKE SURE TO PASTE YOUR COPIED GITHUB RELEASE LINK ADDRESS BELOW!
        url = "https://github.com/gajjjar5802-coder/isl-translator/releases/download/v1.0/model.p" 
        with st.spinner("Downloading AI Model Brain from cloud... Please wait..."):
            urllib.request.urlretrieve(url, model_path)
            
    with open(model_path, 'rb') as f:
        return pickle.load(f)['model']

model = load_model()
# --- UPDATED: Max hands changed to 2 ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Create clean side-by-side columns
col1, col2 = st.columns([5, 4])

with col1:
    st.markdown("### 📹 Video Stream")
    run_app = st.checkbox("🔌 Toggle Webcam Feed", value=False)
    FRAME_WINDOW = st.image([])

with col2:
    st.markdown("### 📋 System Translation")
    
    if 'sentence' not in st.session_state:
        st.session_state.sentence = ""
    
    # Clean UI Card for Outputs
    text_placeholder = st.empty()
    
    st.write(" ")
    if st.button("🗑️ Clear Output Display", use_container_width=True):
        st.session_state.sentence = ""

if run_app:
    cap = cv2.VideoCapture(0)
    last_predicted = None
    counter = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        predicted_char = "No Hand Detected"
        data_aux = [] # Reset feature list for every frame

        if results.multi_hand_landmarks:
            # --- START OF NEW 3D TWO-HAND LOGIC ---
            # Process up to 2 hands for data_aux
            for hand_landmarks in results.multi_hand_landmarks[:2]:
                # Draw landmarks on the live feed
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
                )
                
                x_list = [lm.x for lm in hand_landmarks.landmark]
                y_list = [lm.y for lm in hand_landmarks.landmark]
                z_list = [lm.z for lm in hand_landmarks.landmark]
                
                min_x, max_x = min(x_list), max(x_list)
                min_y, max_y = min(y_list), max(y_list)
                min_z, max_z = min(z_list), max(z_list)
                
                # Normalize and collect 3D coordinates (X, Y, Z)
                for lm in hand_landmarks.landmark:
                    norm_x = (lm.x - min_x) / (max_x - min_x if max_x != min_x else 1)
                    norm_y = (lm.y - min_y) / (max_y - min_y if max_y != min_y else 1)
                    norm_z = (lm.z - min_z) / (max_z - min_z if max_z != min_z else 1)
                    data_aux.extend([norm_x, norm_y, norm_z])

                # Draw individual clean bounding boxes for tracking visual feedback
                x_min, y_min = int(min_x * W), int(min_y * H)
                x_max, y_max = int(max_x * W), int(max_y * H)
                cv2.rectangle(frame, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (0, 255, 0), 2)

            # PADDING: If only 1 hand is visible, add 63 zeros so array length stays exactly 126
            if len(results.multi_hand_landmarks) == 1:
                data_aux.extend([0.0] * 63)
            # --- END OF NEW 3D TWO-HAND LOGIC ---

        # Make predictions only if feature length matches 126
        if len(data_aux) == 126:
            try:
                prediction = model.predict([data_aux])
                predicted_char = str(prediction[0])
                
                # Smooth prediction lock (requires 12 frames of stability)
                if predicted_char == last_predicted:
                    counter += 1
                    if counter == 12:
                        if not st.session_state.sentence.endswith(predicted_char):
                            st.session_state.sentence += f"{predicted_char}"
                else:
                    last_predicted = predicted_char
                    counter = 0

                # Put predicted text on frame near the first detected hand
                first_hand_x = int(results.multi_hand_landmarks[0].landmark[0].x * W)
                first_hand_y = int(results.multi_hand_landmarks[0].landmark[0].y * H)
                cv2.putText(frame, predicted_char, (first_hand_x, first_hand_y - 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
            except:
                pass

        # Update the beautifully styled markdown box on the side
        text_placeholder.markdown(f"""
            <div class="output-box">
                <p style="margin-bottom:5px; color:#6b7280; font-size:14px; font-weight:bold; text-transform:uppercase; tracking-wide;">Active Character</p>
                <h1 style="margin-top:0; color:#1f2937; font-size:48px; font-weight:800;">{predicted_char}</h1>
                <hr style="margin: 15px 0; border: 0; border-top: 1px solid #e6e9ef;">
                <p style="margin-bottom:5px; color:#6b7280; font-size:14px; font-weight:bold; text-transform:uppercase;">Generated Text Sequence</p>
                <h3 style="margin-top:0; color:#059669; font-size:24px; font-family:monospace; background-color:#f0fdf4; padding:10px; border-radius:8px; border:1px solid #bbf7d0;">{st.session_state.sentence if st.session_state.sentence else '...'}</h3>
            </div>
        """, unsafe_allow_html=True)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
