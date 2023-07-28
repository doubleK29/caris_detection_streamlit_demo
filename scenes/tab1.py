import streamlit as st
import cv2
import time
import numpy as np
import os
from stqdm import stqdm


def tab1_scene(model):
    # Center align zoom image
    st.markdown(
        """
    <style>
        button[title^=Exit]+div [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
    # Upload file
    uploaded_file = st.file_uploader(label="Upload single image", key="single_upload")

    col1, _, col2 = st.columns([1, 4, 1])

    if uploaded_file is not None:
        with col1:
            a = st.button("Predict", key="image_upload")
        if a:
            image_byte = uploaded_file.read()
            np_array = np.frombuffer(image_byte, np.uint8)

            # Make sure input file is an image
            try:
                byte2arr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(byte2arr, cv2.COLOR_BGR2RGB)
            except Exception:
                st.error(
                    "ERROR: The input file must be an image!\
                        Please browser file again."
                )
                st.stop()

            # Save image before predicting
            saved_folder = "saved/"
            if not os.path.exists(saved_folder):
                os.makedirs(saved_folder, exist_ok=True)
            if len(os.listdir(saved_folder)) == 0:
                newest_number_image = ""
            else:
                newest_number_image = len(os.listdir(saved_folder)) + 1
            img_path = os.path.join(saved_folder, f"img_test_{newest_number_image}.jpg")
            cv2.imwrite(img_path, img)

            # Predict
            start_time = time.time()
            pbar = stqdm(
                model.predict(
                    source=img_path,
                    imgsz=960,
                    save=True,
                    conf=0.4,
                    save_txt=False,
                    device="cpu",
                ),
                desc="Predicting...",
            )
            running_time = time.time() - start_time  # Print predicted taken time

            for _ in pbar:
                time.sleep(0.5)

            # Show predicted time
            st.success(f"Prediction finished in {round(running_time, 2)}s", icon="✅")

            # Load image from runs/segment/predict# folder
            runs_folder = "runs/segment"
            runs_lastest_num = len(os.listdir(os.path.join(runs_folder)))
            if runs_lastest_num == 1:
                runs_lastest_num = ""
            predicted_img_name = os.listdir(
                os.path.join(runs_folder, f"predict{runs_lastest_num}")
            )[0]
            predicted_img_path = os.path.join(
                runs_folder, f"predict{runs_lastest_num}", predicted_img_name
            )
            predicted_img = cv2.imread(predicted_img_path)

            # Display images with grid size n_cols x n_rows
            n_cols = 2
            n_rows = 1
            rows = [st.container() for _ in range(n_rows)]
            cols_per_row = [r.columns(n_cols) for r in rows]
            cols = [column for row in cols_per_row for column in row]

            for image_index in range(2):
                if image_index % 2 == 0:
                    cols[image_index].image(img, caption="Original image")
                else:
                    cols[image_index].image(predicted_img, caption="Predicted image")

            # Download single image
            # Convert to bytes before save
            with col2:
                with open(predicted_img_path, "rb") as f:
                    predicted_img_bytes = f.read()
                st.download_button(
                    "Download", data=predicted_img_bytes, file_name="predicted_img.jpg"
                )
