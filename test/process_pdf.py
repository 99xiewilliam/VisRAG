import requests
import os

pdf_dir = "/home/xwh/VisRAG/dataset"
for pdf_file in os.listdir(pdf_dir):
    if pdf_file.endswith(".pdf"):
        with open(f"{pdf_dir}/{pdf_file}", "rb") as f:
            response = requests.post(
                "http://localhost:8000/api/v1/index/pdf",
                files={"pdf": f},
                data={"pdf_name": pdf_file}
            )
        print(f"{pdf_file}: {response.json()}")