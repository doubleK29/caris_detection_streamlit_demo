import streamlit as st
from utils import check_folder
import time
import os
from utils import extract_zip, create_zip_file
import shutil
from stqdm import stqdm


def tab2_scene(model):
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
    # Upload folder of images
    upload_file = st.file_uploader(
        "Upload a zip file", type=["zip"], key="folder_upload"
    )
    col3, _, col4 = st.columns([1, 0.5, 0.5])

    if upload_file is not None:
        with col3:
            b = st.button("Predict", key="folder_predict")
        if b:
            # Check folder before predicting
            save_folder = "./saved"
            extract_folder = (upload_file.name).split(".")[0]
            folder_path = os.path.join(save_folder, extract_folder)

            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
            os.makedirs(folder_path, exist_ok=True)

            # Extract the images to the save folder
            extract_zip(upload_file, save_folder)

            # Check valid folder
            check_folder(folder_path)

            # Predict
            start_time = time.time()
            pbar = stqdm(
                model.predict(
                    source=folder_path,
                    imgsz=960,
                    save=True,
                    conf=0.4,
                    save_txt=False,
                ),
                desc="Predicting...",
            )
            running_time = round(time.time() - start_time, 2)

            for _ in pbar:
                time.sleep(0.3)

            # Show predicted time
            st.success(f"Prediction finished in {round(running_time, 2)}s", icon="✅")

            # Load image from runs/segment/predict# folder
            runs_folder = "runs/segment"
            runs_lastest_num = len(os.listdir(os.path.join(runs_folder)))
            if runs_lastest_num == 1:
                runs_lastest_num = ""
            runs_latest_folder = os.path.join(runs_folder, f"predict{runs_lastest_num}")

            image_bytes = []
            for image_file in os.listdir(runs_latest_folder):
                predicted_image_path = os.path.join(runs_latest_folder, image_file)
                origin_image_path = os.path.join(folder_path, image_file)

                with open(origin_image_path, "rb") as file:
                    origin_img = file.read()
                    image_bytes.append(origin_img)

                with open(predicted_image_path, "rb") as file:
                    predicted_img = file.read()
                    image_bytes.append(predicted_img)

            # Display images with grid size n_cols x n_rows
            n_cols = 2
            n_rows = 1 + len(image_bytes) // int(n_cols)
            rows = [st.container() for _ in range(n_rows)]
            cols_per_row = [r.columns(n_cols) for r in rows]
            cols = [column for row in cols_per_row for column in row]

            for image_index, img in enumerate(image_bytes):
                if image_index % 2 == 0:
                    # Original image
                    caption_index = image_index // 2 + 1
                    cols[image_index].image(
                        img, caption=f"Original image {caption_index}"
                    )
                else:
                    # Predicted image
                    caption_index = (image_index - 1) // 2 + 1
                    cols[image_index].image(
                        img, caption=f"Predicted image {caption_index}"
                    )

            # Download folder image
            # Download button to save the runs_latest_folder as a ZIP file
            output_zip_file = os.path.join(save_folder, "predicted_folder.zip")
            create_zip_file(runs_latest_folder, output_zip_file)

            # Read the ZIP file as bytes
            with col4:
                with open(output_zip_file, "rb") as file:
                    zip_file_bytes = file.read()

                st.download_button(
                    "Download as zip file",
                    data=zip_file_bytes,
                    file_name="predicted_folder.zip",
                    key="folder_download",
                )
