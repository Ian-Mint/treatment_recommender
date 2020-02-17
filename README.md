# Treatment Recommender

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
    * Admissions Table (Do any of these matter? Excluded for now.)
        * Language
        * Religion
        * Marital_status
        * Ethnicity
    * Patients Table
        * Gender
        * DOB
- Elixhauser premorbid status
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

## Processed data

### Vasopressor

`data/vasopressin_chunked.pkl`

`dict` with `keys=hadm_id` and value is a `numpy` array representing the amount of Vasopressin administered in 4-hour
chunks. This excludes CareVue data.

TODO: consider changing start time from admission start to sepsis diagnosis start.