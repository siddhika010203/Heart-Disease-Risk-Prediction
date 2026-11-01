# Heart Disease Risk Prediction (SPSS + Python) - Python part

Educational project only. Not a diagnostic tool.

## Folder layout
```
heart_disease_project/
├── heart_disease_project.py
├── requirements.txt
├── README.md
├── data/
│   └── processed.cleveland.data   <- put your dataset file here
└── outputs/                       <- created automatically
```

## Run in VS Code (Windows)
Open the folder in VS Code (File > Open Folder), then Terminal > New Terminal:
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python heart_disease_project.py
```
macOS / Linux:
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 heart_disease_project.py
```
If PowerShell blocks activation, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

Then select the interpreter: Ctrl+Shift+P > "Python: Select Interpreter" > the `venv` one.

## Options
```
python heart_disease_project.py --seed 42 --test-size 0.25 --tree-depth 4
python heart_disease_project.py --data "C:\path\to\processed.cleveland.data"
```
If the data file is missing the script tries to download it from UCI.

## Outputs (./outputs)
test_metrics.csv, missing_values.csv, descriptive_statistics.csv, target_frequency.csv,
crosstabs.txt, logistic_coefficients.csv, fig7-fig13 PNG charts.
