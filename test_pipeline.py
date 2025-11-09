from src.data_pipeline.extract import extract_data
from src.data_pipeline.transform import transform_data
from src.data_pipeline.load import load_data

if __name__ == "__main__":
    # 1️⃣ ดึงข้อมูล
    df = extract_data("DiseaseAndSymptoms.csv")

    # 2️⃣ แปลงข้อมูล (clean + encode + split)
    split_dfs, mapping = transform_data(df)

    # 3️⃣ เซฟข้อมูลที่ผ่านการแปลง
    load_data((split_dfs, mapping), "data/output")
