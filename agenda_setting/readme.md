# Agenda Setting

## Dependencies

Require python package `contextualized_topic_models` (v2.5.0).

## Data 

We use the raw data `russia_ukraine_vk_war_only.csv` for topic-level analysis and `russia_ukraine_vk.csv` for term-level analysis. Check the data in the original repo. 

## Run the code 

### Topic-level analysis

To reproduce Figure 2 in the original paper, run `ipython ctm.ipynb`. All pipeline including preprocessing, training, and evaluation are contained in the notebook. 
### Term-level analysis

To reproduce Figure 3 in the original paper, run `ipython vis.ipynb`. This notebook includes preprocessing and visualize the difference between the two media types. 

## Results

The results are demonstrated as figures in our report. 
