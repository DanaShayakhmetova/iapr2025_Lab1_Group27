# Chocolate Detection Project - Group 27

Hello, this is our chocolate detection project, Group 27.

---

## Before running `main.py`

Please ensure that inside the `data` folder, you have added the following folders:

- `train`
- `test`
- `references`

---

## Folder Organization

For `main.py` to successfully generate `submission.csv`, the project folder structure should be:


```
final_project_group27/
├── classification_model/
│ ├── model.pkl
│ ├── pca.pkl
│ └── scaler.pkl
├── data/
│ ├── references/ # 13 JPGs of reference chocolates
│ ├── train/ # train JPGs
│ ├── test/ # test JPGs
│ └── train_labels/ # txt files with bounding box info for train images
├── py_scripts/
│ ├── check.py
│ └── classification.py # trains the model and saves it for main.py
├── src/
│ ├── environment.yml
│ ├── README.md # provided by TAs
│ ├── requirements.txt
│ └── train.csv # provided by TAs
├── main.py # main script that creates submission.csv
├── report.pdf # detailed report
└── README.md # this file you're reading
```


---

## Ready to Run

If your files are organized as above, you are ready to run our script.

---

## After Running `main.py`

You will also get a folder called `yolo_bounding_boxes` containing:

- `chocolate_segments/`  
  Extracted segments of chocolates from train images

- `segment_labels.csv`  
  Corresponding chocolate labels for each extracted segment
