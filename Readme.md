# NeuroMotion — Explainable Infant Movement Analysis

NeuroMotion is a research prototype for exploring infant spontaneous movement analysis, model evaluation, explainability, and kinematic perturbation experiments related to early cerebral palsy (CP) risk research.

The platform combines saved model evaluation results, skeletal movement visualizations, Grad-CAM joint attribution, and recorded perturbation experiments in an interactive web interface.

> **Disclaimer:** NeuroMotion is a research prototype, not a clinically validated diagnostic system. Its outputs must not be used to diagnose cerebral palsy or make medical decisions.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation and Setup](#installation-and-setup)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Understanding the Visualizations](#understanding-the-visualizations)
- [Research Methodology](#research-methodology)
- [Current Implementation and Limitations](#current-implementation-and-limitations)
- [Future Enhancements](#future-enhancements)
- [Troubleshooting](#troubleshooting)

## Project Overview

NeuroMotion provides a visual interface for examining saved results from an infant movement analysis research pipeline based on the DeepInMotion-Y pretrained Spatial-Temporal Graph Convolutional Network (ST-GCN) and research on explainability through kinematic perturbations.

The project focuses on three main questions:

1. **Model evaluation:** How well does the saved baseline model distinguish between the target movement classes?
2. **Explainability:** Which skeletal joints receive higher attribution scores from the saved Grad-CAM explanations?
3. **Perturbation analysis:** How do recorded model outputs and explanation scores change when selected joint coordinates or kinematic values are perturbed?

The application presents existing experiment results through a React frontend and a FastAPI backend. The displayed results should be interpreted according to their data source and the implementation status of each feature.

## Features

### 1. Research Overview

Displays saved baseline model evaluation results, including:

- Validation accuracy
- Balanced accuracy
- ROC–AUC
- Number of movement features analyzed
- Confusion matrix
- Class-wise precision, recall, and F1 scores
- Ranked baseline feature importance
- Research interpretation notes

These metrics describe the saved evaluation results. They do not, by themselves, establish clinical reliability.

### 2. Model Insights

Provides a dedicated workspace for examining model-related findings and saved explanatory information, depending on the data and components available in the project.

### 3. Movement Analysis

Includes an interactive skeletal visualization with:

- A temporal-window selector
- A frame slider for navigating the available movement sequence
- Joint colors representing saved Grad-CAM attribution scores
- A color scale ranging from lower to higher attribution
- A joint movement plot showing horizontal and vertical image-coordinate changes over time
- An option to download extracted keypoints as a CSV file

The skeleton visualization uses the available keypoint coordinates and saved window-level attribution scores. Unless a separate inference pipeline is connected, changing the displayed frame does not calculate a new Grad-CAM explanation for that individual frame.

### 4. Kinematic Perturbation Experiment

Displays saved perturbation experiment results, including:

- Selected joint
- Model or experiment identifier
- Baseline output
- Selected perturbation output
- Change from baseline
- Perturbation-response graph
- Table of recorded perturbation strengths and outputs

The perturbation selector displays recorded experiment values. It does not necessarily execute a new model inference when the selection changes.

### 5. Interactive Grad-CAM Joint Heatmap

Displays joint attribution scores for the available temporal windows.

The visualization includes:

- Window selection
- Joint names and indices
- Attribution scores
- Ranking of joints by score
- Relative score visualization

Higher displayed attribution indicates a larger relative score within the selected window. It is not a direct measurement of disease severity, clinical risk, or causal importance.

### 6. Grad-CAM vs. Perturbation Sensitivity

Compares saved Grad-CAM attribution scores with changes observed in recorded perturbation experiments.

The analysis includes:

- Pearson correlation
- Spearman rank correlation
- Number of matched joints
- Number of numeric pairs used
- Scatter plot
- Joint-wise attribution and perturbation summaries

This comparison explores whether joints with higher explanation scores also show greater output sensitivity under the available perturbations. A correlation alone does not establish causality or validate the explanation method.

### 7. Research Explorer

Provides additional analysis of saved research outputs, such as joint attribution scores, consistency, velocity perturbations, and temporal-window sensitivity, where supported by the available data.

## Technology Stack

### Frontend

- React
- Vite
- JavaScript
- Recharts for charts and data visualization
- Lucide icons
- CSS for layout and styling

### Backend

- Python
- FastAPI
- Uvicorn
- Pandas for tabular research data
- REST API endpoints for delivering saved results to the frontend

### Research and Model Components

- DeepInMotion-Y pretrained ST-GCN
- Grad-CAM joint attribution
- Skeletal keypoint sequences
- Kinematic perturbation experiments
- CSV-based experimental results

The research model and experiment-generation environment may have dependencies beyond those required to run the web application.

## Project Structure

The expected structure is approximately:

```text
NeuroMotion/
├── README.md
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   └── data/
│       └── saved research result files
│
└── frontend/
    ├── package.json
    ├── index.html
    ├── public/
    │   └── movement_keypoints.csv
    └── src/
        ├── App.jsx
        ├── GRADCAMSkeleton.jsx
        ├── ResearchExplorer.jsx
        └── other frontend components
```

This is a reference structure. Keep the filenames and folders that actually exist in your repository; not every listed file is guaranteed to be present.

## Prerequisites

Install the following software before running the application:

- Python 3.10 or another Python version supported by your backend dependencies
- Node.js and npm
- Visual Studio Code or another editor
- Git, if you want to clone or version-control the repository

Check your installations:

```powershell
python --version
python -m pip --version
node --version
npm --version
```

If your model-reproduction scripts require a particular Python version, follow their original environment requirements separately.

## Installation and Setup

### Step 1: Open the project

Open the NeuroMotion repository folder in Visual Studio Code.

Open a terminal from the project root.

### Step 2: Set up the backend

Navigate to the backend folder:

```powershell
cd backend
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the virtual environment's Python directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Create `requirements.txt` inside the `backend` directory if it does not already exist.

For the basic API dependencies, use:

```text
fastapi
uvicorn[standard]
pandas
```

Install them:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Important:** This is the minimal dependency list for the basic API setup, not a verified complete list of every dependency in the repository. Add any additional packages imported by `main.py` or required by your backend implementation.

### Step 3: Set up the frontend

Open a second terminal in Visual Studio Code.

Navigate to the frontend directory from the project root:

```powershell
cd frontend
```

Install the dependencies declared in `package.json`:

```powershell
npm install
```

If `package.json` is missing, check that you are in the correct frontend directory before proceeding.

## Running the Application

Keep the backend and frontend running in separate terminals.

### Terminal 1: Start the backend

From the project root:

```powershell
cd backend
```

Activate the virtual environment if necessary:

```powershell
.\.venv\Scripts\Activate.ps1
```

Start FastAPI:

```powershell
python -m uvicorn main:app --reload
```

The backend should be available at:

- API base URL: `http://127.0.0.1:8000`
- Swagger documentation: `http://127.0.0.1:8000/docs`
- ReDoc documentation: `http://127.0.0.1:8000/redoc`

Swagger allows you to inspect the available endpoints and test their responses.

If your backend entry point or folder structure differs, adjust the command accordingly.

### Terminal 2: Start the frontend

From the project root:

```powershell
cd frontend
npm run dev
```

Vite will print the frontend URL in the terminal. It is commonly:

`http://localhost:5173`

Open that URL in your browser.

### Stopping the application

Press `Ctrl+C` in each terminal to stop the frontend and backend.

### Restarting the application later

Once the dependencies are installed, the usual startup sequence is:

**Backend terminal**

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python -m uvicorn main:app --reload
```

**Frontend terminal**

```powershell
cd frontend
npm run dev
```

If the virtual environment is already active, you do not need to activate it again.

## API Documentation

FastAPI automatically generates interactive API documentation from the endpoints defined in `main.py`.

Open:

`http://127.0.0.1:8000/docs`

The application uses API endpoints to retrieve available saved results, including baseline evaluation metrics, joint attribution scores, perturbation experiment data, and movement keypoints.

The exact endpoint names depend on the backend implementation. Check Swagger for the authoritative list.

### Testing an endpoint

1. Start the backend.
2. Open Swagger.
3. Expand the endpoint you want to test.
4. Select **Try it out**, if available.
5. Execute the request.
6. Inspect the HTTP status code and JSON response.

A successful Swagger response confirms that the endpoint returned data for that request. It does not guarantee that the frontend is calling the same URL, reading the response schema correctly, or rendering the result successfully.

### If Swagger works but the frontend does not

Check the following:

- Confirm the frontend API base URL.
- Confirm that the frontend fetches the correct endpoint.
- Check that the response field names match what the component expects.
- Inspect the browser's Developer Tools → Console and Network tabs.
- Inspect the backend terminal for errors.
- Confirm that CORS settings permit requests from the frontend origin.
- Confirm that the expected CSV and saved experiment files are present.

If the frontend uses a Vite environment variable such as `VITE_API_BASE_URL`, verify its value and restart the Vite development server after changing it.

## Understanding the Visualizations

### Model evaluation metrics

**Validation accuracy** is the fraction of evaluated records classified correctly.

**Balanced accuracy** averages recall across the target classes, making it more informative than ordinary accuracy when class frequencies differ.

**ROC–AUC** summarizes how well the model separates classes across classification thresholds. Its interpretation depends on the positive class, evaluation data, and metric implementation.

**Precision** measures the fraction of predictions assigned to a class that are correct.

**Recall** measures the fraction of actual members of a class that are identified.

**F1 score** combines precision and recall using their harmonic mean.

**Confusion matrix** displays correct and incorrect predictions by actual and predicted class. Always check which class each row and column represents.

### Grad-CAM attribution

Grad-CAM uses gradients and intermediate model activations to estimate which parts of a model's representation contribute to a selected target output.

In this project, saved joint attribution scores are associated with skeletal joints and temporal windows. The visual interface maps these scores onto the corresponding skeleton or joint heatmap.

These are model explanations, not proof that a joint causes a particular outcome.

### Kinematic perturbations

A kinematic perturbation deliberately modifies a selected movement variable, such as a joint coordinate, angle, or velocity, and examines the resulting change in a recorded or recomputed model output.

A perturbation factor below `1.0` scales the selected value down; a factor above `1.0` scales it up. The precise operation depends on the experiment implementation.

The output change is typically computed as:

`Perturbed output − Baseline output`

The magnitude of that difference indicates how much the output changed under the tested perturbation. The direction indicates whether the numerical output increased or decreased.

The exact meaning of the output must be verified against the model's output definition. Do not assume that every displayed scalar is a calibrated CP-risk probability.

### Temporal-window sensitivity

The window-sensitivity panel compares the observed changes across temporal windows.

A window with a larger mean or maximum absolute change is more sensitive under the specific recorded perturbations. A very small change means the output was nearly unchanged under those tested conditions.

A small perturbation response does not necessarily mean that the corresponding movement segment is clinically unimportant.

### Correlation analysis

Pearson correlation measures linear association between paired numerical values.

Spearman correlation measures association between their ranks.

Both depend on the number and quality of the matched observations. When only a few joints are matched, the resulting correlation is exploratory and may change substantially with additional data.

## Research Methodology

The research workflow represented by the platform is:

1. Obtain or load a movement sequence and its skeletal keypoints.
2. Use an existing trained model or saved experiment outputs.
3. Inspect the saved baseline evaluation results.
4. Examine the available Grad-CAM joint attribution scores.
5. Select joints and perturbation factors used in the recorded experiments.
6. Compare perturbed outputs with the baseline.
7. Examine sensitivity across temporal windows.
8. Compare attribution rankings with perturbation sensitivity.
9. Interpret findings while documenting limitations and the provenance of the data.

The web interface is primarily a visualization and analysis layer. Whether a step executes new model inference or displays previously saved results must be checked in the relevant code.

## Current Implementation and Limitations

The project currently includes a web interface for viewing saved evaluation results, skeletal keypoints, joint attribution, and recorded perturbation analyses.

The following distinctions are important:

- **Saved results versus new inference:** Displaying stored experiment results is not equivalent to running the trained model again.
- **Frame-level versus window-level explanations:** Reusing a window-level score across multiple frames does not produce a new frame-specific Grad-CAM explanation.
- **Visualization versus tracking:** Rendering existing keypoints does not mean that the frontend performs pose estimation or tracks joints from raw video.
- **Perturbation display versus experiment execution:** A selector that switches between recorded factors does not necessarily execute the perturbation pipeline.
- **Research metrics versus clinical validation:** Evaluation scores do not establish that the model is suitable for clinical screening or diagnosis.
- **Correlation versus causation:** A relationship between attribution and perturbation response does not prove that the highlighted joints are causal biomarkers.
- **Reproducibility:** Results depend on the saved datasets, model checkpoints, preprocessing, configuration, and experimental procedures used to generate them.

Some enhancements, including live video processing, on-demand inference, automated report generation, and full experiment orchestration, may require additional implementation.

## Future Enhancements

The following are proposed extensions beyond the current visualization workflow. They should be treated as future work until implemented and validated.

### 1. Live model inference

Connect the frontend to a backend inference service that accepts a movement sequence, runs the trained model, and returns its output and metadata.

This would distinguish newly computed predictions from saved experiment results.

### 2. Frame-specific explanations

Compute explanations for individual frames or explicitly defined temporal windows rather than repeating the same saved attribution scores across every frame.

This would allow researchers to investigate how model attention changes throughout a movement sequence.

### 3. Raw-video processing

Integrate a validated pose-estimation pipeline that extracts skeletal coordinates from uploaded videos, with appropriate quality checks and preprocessing.

This would extend the platform beyond previously extracted keypoint files.

### 4. Reproducible experiment execution

Allow authorized researchers to configure joint selection, perturbation type, perturbation factor, and temporal window, then run experiments and save the configuration and outputs.

Record model versions, preprocessing parameters, timestamps, and random seeds where applicable.

### 5. Expanded perturbation methods

Extend the experiments to compare coordinate scaling, velocity modification, angular modification, and other justified kinematic interventions.

Ensure that modifications preserve valid skeletal structure and distinguish realistic movements from artificial or implausible inputs.

### 6. Robustness and stability analysis

Repeat experiments across multiple sequences, windows, perturbation strengths, and model checkpoints.

Report variation and uncertainty instead of relying on a single result or a small number of joints.

### 7. Improved explanation evaluation

Evaluate explanations using multiple complementary criteria, including:

- Attribution stability
- Perturbation sensitivity
- Temporal consistency
- Joint-ranking consistency
- Comparison with independent movement-analysis evidence, where available

Do not assume that any single criterion establishes explanation faithfulness.

### 8. Statistical analysis

Add confidence intervals, suitable significance tests, effect sizes, and participant-level analysis where the dataset and study design permit them.

Account for repeated measurements and avoid treating correlated windows from the same sequence as independent participants.

### 9. Dataset and experiment management

Add structured storage for sequences, model versions, experiment configurations, results, and metadata.

Provide traceability between each displayed chart and the source data that produced it.

### 10. Exportable research reports

Generate downloadable reports containing figures, tables, configuration details, and interpretation notes. Clearly label saved results versus newly computed outputs.

### 11. User access and privacy

Add authentication, role-based access, secure file handling, and appropriate retention controls before storing sensitive research or video data in a shared environment.

### 12. Independent clinical validation

Any future clinical use would require appropriate datasets, independent evaluation, bias and generalization studies, clinical oversight, and relevant regulatory and ethical review.

These steps are prerequisites to considering clinical use; they are not implied by the current prototype.

## Troubleshooting

### `ModuleNotFoundError` when starting the backend

Activate the correct virtual environment and install the missing dependency.

```powershell
python -m pip install -r requirements.txt
```

If the file does not include the missing package, add the appropriate dependency before reinstalling.

### `No module named uvicorn`

```powershell
python -m pip install "uvicorn[standard]"
```

### `No module named fastapi`

```powershell
python -m pip install fastapi
```

### `No module named pandas`

```powershell
python -m pip install pandas
```

### Frontend dependency or script errors

Check that you are inside `frontend` and that `package.json` exists.

```powershell
npm install
npm run dev
```

### API requests fail from the browser

Check the backend process, API URL, endpoint path, CORS configuration, and browser Network tab.

### Data or charts are missing

Verify that the expected files exist in the paths used by the backend and frontend. Confirm that the JSON or CSV schema matches the fields used by the visualization components.

### Port already in use

Start the backend on another port if needed:

```powershell
python -m uvicorn main:app --reload --port 8001
```

If you change the backend port, update the frontend API base URL accordingly.

## Acknowledgements

This project builds on the DeepInMotion-Y research and pretrained ST-GCN components, alongside research on explainability through kinematic perturbations.

Please consult the original research publications and repository licenses when redistributing model files, datasets, or derived research artifacts.

## Project Status

**Status:** Research prototype under development.

NeuroMotion is intended to support exploration, visualization, and evaluation of explainable infant movement analysis methods. Its displayed results require appropriate research interpretation and independent validation before any broader application.
