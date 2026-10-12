# ANA500 Micro-Project 4: Adult Income Deep Learning Classification

**Author:** Valentin Diaconu  
**Course:** ANA500 - Data Science  
**Project stage:** Acquire, Prepare, Analyze, Report, and Act

## Project Overview

This project examines whether demographic, educational, and employment characteristics can classify whether an adult reports annual income above $50,000. Four approaches are evaluated on the same test data:

1. Majority-class baseline
2. Logistic regression
3. Dense multilayer perceptron (MLP)
4. Long short-term memory (LSTM) network

The project extends earlier exploratory work with the Adult Income dataset and demonstrates a complete data-science workflow, including data acquisition, preparation, modeling, evaluation, reporting, and interpretation.

## Important Methodological Disclosure

The Adult Income dataset is **cross-sectional**, not time-series data. It contains no timestamps and no natural sequence of observations.

For the LSTM demonstration, the 89 encoded predictors are reshaped into an analyst-defined feature sequence. These sequence positions represent encoded variables rather than time. Consequently, the LSTM is included only as a recurrent-neural-network architecture comparison. Its results must not be interpreted as evidence of temporal learning or time-series forecasting.

This limitation affects whether the project fully satisfies a requirement specifically calling for time-series analysis.

## Research Question

Can deep neural networks use demographic, educational, and employment characteristics to classify whether an adult reports annual income above $50,000?

## Hypothesis

The MLP and LSTM are expected to outperform the majority-class baseline because nonlinear layers can represent interactions among the available predictors. However, the LSTM may not outperform the MLP because the encoded predictors have no natural temporal order.

## Dataset

The analysis uses the [UCI Adult dataset](https://archive.ics.uci.edu/dataset/2/adult), which was derived from 1994 U.S. Census data.

- Original observations: 32,561
- Variables: 15
- Exact duplicates removed: 24
- Prepared observations: 32,537
- Positive class (`>50K`): 24.09%
- Encoded predictors: 89

Missing categorical values originally represented by `?` are retained as an explicit `Unknown` category instead of automatically deleting all incomplete records. Plausible extreme numerical values are also retained because they may contain relevant predictive information.

## Data Preparation

The workflow:

- Trims categorical text using Pandas string-dtype-compatible code.
- Replaces `?` entries with an explicit `Unknown` category.
- Removes exact duplicate records.
- Creates a binary income target.
- Uses stratified training, validation, and test partitions.
- Fits all preprocessing operations only on the training data.
- Imputes numerical variables with the training-set median.
- Standardizes numerical predictors.
- Imputes categorical variables with the most frequent category when needed.
- One-hot encodes categorical variables while handling unseen categories.

### Data Split

| Partition | Records | Percent above $50K |
|---|---:|---:|
| Training | 22,775 | 24.09% |
| Validation | 4,881 | 24.09% |
| Test | 4,881 | 24.09% |

## Models

### Majority-Class Baseline

Predicts the most common income class and provides a minimum performance reference.

### Logistic Regression

Provides a conventional linear classification benchmark with balanced class weights.

### Dense MLP

Uses two hidden layers with ReLU activation, dropout, L2 regularization, Adam optimization, binary cross-entropy loss, class weighting, and early stopping.

### Feature-Sequence LSTM

Reshapes each encoded feature vector to `(89, 1)` and processes it as an artificial sequence. The network demonstrates an LSTM implementation but does not model chronological information.

## Test-Set Results

| Model | Accuracy | Precision | Recall | F1 score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Majority baseline | 75.91% | 0.00% | 0.00% | 0.00% | 50.00% |
| Logistic regression | 81.09% | 57.16% | 85.88% | 68.64% | 91.08% |
| **Dense MLP** | **82.26%** | **58.83%** | **87.84%** | **70.46%** | **91.79%** |
| Feature-sequence LSTM | 73.47% | 46.84% | 75.00% | 57.67% | 81.80% |

The Dense MLP produced the strongest overall test performance. Logistic regression performed similarly and remained a competitive, more interpretable benchmark. The LSTM performed less effectively, which is consistent with the absence of a meaningful temporal or sequential structure.

## Conclusions and Recommended Action

- Use the Dense MLP as the strongest predictive benchmark from this experiment.
- Retain logistic regression when interpretability, computational efficiency, or auditability is prioritized.
- Do not treat the LSTM results as time-series evidence.
- Evaluate classification thresholds according to the practical costs of false positives and false negatives before deployment.
- Validate the selected approach on newer and more representative data before operational use.
- Review performance and error rates across demographic groups because the dataset is historical and may reflect sampling or social inequities.

Model outputs describe predictive associations and do not establish that any characteristic causes higher or lower income.

## Repository Contents

```text
.
├── README.md
├── ANA500_Micro_Project_4_Adult_Income_Deep_Learning_Executed.ipynb
├── ANA500_Micro_Project_4_Adult_Income_Deep_Learning.pptx
├── ANA500_Micro_Project_4_Adult_Income_Deep_Learning.pdf
├── ANA500_Micro_Project_4_Adult_Income_Deep_Learning_Code.pdf
└── adult income.csv
```

Adjust the filenames in this section if the GitHub repository uses different names.

## Running the Notebook

### 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd <YOUR-REPOSITORY-NAME>
```

### 2. Create an environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

### 3. Add the dataset

Place `adult income.csv` in the repository folder. If the file is unavailable, the notebook attempts to retrieve the original Adult data file from UCI.

### 4. Start JupyterLab

```bash
jupyter lab
```

Open the notebook and run the cells in order. Training time and exact neural-network results can vary slightly across operating systems, TensorFlow versions, and hardware.

## Software

- Python 3
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- TensorFlow/Keras
- JupyterLab

## Limitations

- Adult Income is cross-sectional and cannot support genuine time-series forecasting.
- The LSTM receives an artificial feature order that has no temporal meaning.
- The data were derived from the 1994 U.S. Census and may not represent current populations or labor-market conditions.
- The binary income threshold simplifies a more complex socioeconomic outcome.
- A single train-validation-test split cannot establish performance across other samples or time periods.
- Predictive associations should not be interpreted as causal effects.

## AI Use Disclosure and Academic Integrity

OpenAI ChatGPT assisted with project organization, code drafting and debugging, phrasing refinement, and README preparation. The author reviewed and executed the code and remains responsible for the analysis, interpretations, citations, and submitted work. All numerical results and charts originate from the executed Adult Income analysis. AI assistance does not replace the dataset, model output, or the author's academic responsibility.

## References

Becker, B., & Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20

OpenAI. (2026). *ChatGPT* [Large language model]. https://chatgpt.com/

TensorFlow Developers. (2026). *TensorFlow and Keras documentation*. https://www.tensorflow.org/

## License and Educational Use

This repository was created for educational purposes as part of ANA500 coursework. Review the UCI dataset documentation for its applicable terms and citation requirements.
