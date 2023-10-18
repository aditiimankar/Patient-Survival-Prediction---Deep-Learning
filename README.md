# Patient-Survival-Prediction---Deep-Learning
Utilizing a deep learning modeling strategy for training extensive datasets and forecasting patient survival.

**Description:**
Rapidly assessing a patient's overall health context has become critical, particularly during the COVID-19 pandemic when healthcare facilities worldwide grapple with an influx of critically ill patients. Intensive Care Units (ICUs) frequently lack access to patients' verified medical histories. Patients in distress or those arriving confused and unresponsive may not be able to provide information regarding chronic conditions such as heart disease, injuries, or diabetes. The transfer of medical records may experience delays, especially for patients transitioning from other healthcare providers or systems. Knowledge about chronic conditions plays a vital role in clinical decision-making, ultimately enhancing patient survival outcomes.

**Problem Statement:
The primary objective is to predict the binary variable "hospital_death" based on the 186 features. The evaluation will be done using Accuracy and Area under the ROC curve.**

**Key Challenges:**

-Large dataset with over 90,000 rows and 186 columns.

-Limited evident relationships with the target variable.

**Steps Taken to Solve the Problem:**

*Managed Missing Values using Missing Completely at Random (MCAR) techniques.
Conducted Exploratory Data Analysis (EDA) to understand feature distributions and relationships.
Performed Univerient & multivarient analysis to find correlation among variables. 
Implemented a Baseline Neural Network model, achieving a validation score of 65% AUC.
Fine-tuned the Neural Network, improving the validation score to 67% AUC.**

**Results:**

-Subsampling features and reducing data led to an enhancement in modeling metrics.

-The application of Keras-Tuner improved the validation AUC score from 65% to 67%.


