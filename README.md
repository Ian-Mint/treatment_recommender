# Treatment Recommender

## Data Processing

### Cleaning

Some simplifications were made to the data to speed up data cleaning.
- Data entered using MetaVision requires far less cleaning because there are fewer free-text fields.
  So, CareVue data will be discarded.
 
 ### Desired Format
 Our target for the data should have 
 - diagnostic status
 - treatment status
 In `T` hour windows (one of the referenced papers used 4).
 
 