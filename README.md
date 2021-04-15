# Treatment Recommender

Rapid diagnosis and treatment is crucial for patients
with sepsis, a leading cause of mortality in intensive care
units. Currently, there is no consensus within the medi-
cal community for a sepsis treatment protocol. Previous
studies that have attempted to address this problem using
machine learning techniques. These studies use reinforce-
ment learning, leading to inherently biased models. We pro-
pose a hybrid imitation-learning approach. First, predict-
ing whether treatment is required using boosting and ig-
noring patient history; next predicting prescription dosage
using a long short-term memory regression. Such an ap-
proach will be less biased than the reinforcement learn-
ing approach, and has the potential to be fine-tuned using
apprenticeship learning to achieve state of the art perfor-
mance. See the [paper](https://github.com/Ian-Mint/treatment_recommender/blob/master/paper.pdf)
for more details.

## Usage

See:

`python train.py --help`

for usage.

## Data Processing

### Cleaning

Some simplifications were made to the data to speed up data cleaning.
- Data entered using MetaVision requires far less cleaning because there are fewer free-text fields.
  So, CareVue data will be discarded.
 
 ### Feature Selection
 Our target for the data should have 
 - diagnostic status
 - treatment status
 In `T` hour windows (one of the referenced papers used 4).
 
 From: https://www.nature.com/articles/s41591-018-0213-5.pdf
A set of 48 variables, including:
- demographics
    * Patients Table
        * Gender
        * DOB
- Elixhauser premorbidity status
    * Generated from pypi `icd` package
    * Requires diagnosis information
- vital signs
- laboratory values
- fluids
- vasopressors received
- fluid balance

Patients’ data were coded
as multidimensional discrete time series with 4-h time steps. Data variables with
multiple measurements within a 4-h time step were averaged (for example, heart
rate) or summed (for example, urine output) as appropriate
