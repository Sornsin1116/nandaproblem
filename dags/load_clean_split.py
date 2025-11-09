from __future__ import annotations
import os, pickle, logging, pendulum
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.models.taskinstance import TaskInstance
import pandas as pd

from src.data_pipeline.extract import extract_data
from src.data_pipeline.transform import transform_data
from src.data_pipeline.load import load_data

logging.basicConfig(level=logging.INFO)

TEMP_DIR = "/tmp/airflow_data"
FINAL_PATH = "data/processed"
DATA_FILE = "DiseaseAndSymptoms.csv"

def airflow_extract(ti: TaskInstance | None = None):
    os.makedirs(TEMP_DIR, exist_ok=True)
    df = extract_data(DATA_FILE)
    tmp_file = os.path.join(TEMP_DIR, "raw_data.pkl")
    df.to_pickle(tmp_file)
    ti.xcom_push(key="data_file_path", value=tmp_file)

def airflow_transform(ti: TaskInstance):
    raw_path = ti.xcom_pull(task_ids="extract_task", key="data_file_path")
    df = pd.read_pickle(raw_path)
    split_data = transform_data(df)
    transformed_file = os.path.join(TEMP_DIR, "transformed.pkl")
    with open(transformed_file, "wb") as f:
        pickle.dump(split_data, f)
    ti.xcom_push(key="data_file_path", value=transformed_file)

def airflow_load(ti: TaskInstance):
    transformed_file = ti.xcom_pull(task_ids="transform_task", key="data_file_path")
    with open(transformed_file, "rb") as f:
        split_data = pickle.load(f)
    os.makedirs(FINAL_PATH, exist_ok=True)
    load_data(split_data, FINAL_PATH)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": pendulum.duration(minutes=5),
}

with DAG(
    dag_id="disease_pipeline",
    default_args=default_args,
    schedule=None,
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    tags=["disease", "ml", "etl"]
) as dag:

    setup = BashOperator(
        task_id="setup_temp",
        bash_command=f"mkdir -p {TEMP_DIR}"
    )

    extract_task = PythonOperator(
        task_id="extract_task",
        python_callable=airflow_extract
    )

    transform_task = PythonOperator(
        task_id="transform_task",
        python_callable=airflow_transform
    )

    load_task = PythonOperator(
        task_id="load_task",
        python_callable=airflow_load
    )

    cleanup = BashOperator(
        task_id="cleanup",
        bash_command=f"rm -rf {TEMP_DIR}",
        trigger_rule="all_done"
    )

    setup >> extract_task >> transform_task >> load_task >> cleanup
