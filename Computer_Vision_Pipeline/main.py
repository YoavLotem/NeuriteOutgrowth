from Computer_Vision_Pipeline.experiment_utils import extract_data_from_plate_images

if __name__ == "__main__":
    folder = r'SHSY5Y 8 day diff 5 drugs 020621/Hila_diff shsy5y_cal_DAPI_principle1_SHSY5Y 5diff cal DAPI 5 drugs_2021.06.02.13.58.15'
    saving_folder = r'experiments results with graphs/validation experiments/Hila_diff shsy5y_cal_DAPI_principle1_SHSY5Y 5diff cal DAPI 5 drugs_2021.06.02.13.58.15'
    extract_data_from_plate_images(folder, saving_folder)


