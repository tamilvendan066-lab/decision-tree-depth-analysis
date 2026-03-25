# Decision Tree Depth Analysis

## Overview
This project demonstrates how a decision tree model works and how its depth affects performance. The main focus is on understanding overfitting by comparing a shallow tree with a deeper tree.

## Objective
- Understand the basic concept of decision tree classification  
- Compare a small tree and a large tree  
- Analyse how tree depth impacts model performance  
- Demonstrate overfitting using visual and numerical results  


## Dataset
A synthetic dataset was generated using the `make_classification` function from Scikit-learn.

- 600 samples  
- 2 features  
- 2 classes  

This dataset was chosen because it allows clear visualisation of how the model separates different classes.

## Implementation
The model was implemented using Python and the Scikit-learn library.

The main steps include:
1. Generating the dataset  
2. Splitting the data into training and testing sets  
3. Training two models:
   - A small tree (max_depth = 3)  
   - A large tree (no depth limit)  
4. Evaluating performance using accuracy and confusion matrix  
5. Visualising the results  

---

## Results
The following outputs were generated:

- Dataset distribution plot  
- Decision boundary for small and large trees  
- Confusion matrix  
- Feature importance  
- Decision tree structure  
- Overfitting analysis graph  

The results show that increasing the depth improves training accuracy, but after a certain point, the test accuracy decreases. This indicates overfitting.


## Files
- `code.py` – Implementation of the model  
- `Decision_Tree_Depth_Analysis.pdf` – Project report  
- Output images:
  - dataset.png  
  - small_tree.png  
  - big_tree.png  
  - confusion_matrix.png  
  - feature_importance.png  
  - tree.png  
  - overfitting.png
  
## Technologies Used
- Python  
- Scikit-learn  
- NumPy  
- Matplotlib

##  How to Run
1. Install required libraries:
2. Open Jupyter Notebook:
3. Run the notebook file

## Results
- Shallow tree gives better generalization
- Deep tree leads to overfitting
- Model complexity increases with depth

##  Key Insight
Increasing tree depth improves training accuracy but reduces testing performance due to overfitting.

##  Author
Tamilvendan Sathyamoorthy

## License
This project is licensed under the MIT License.
