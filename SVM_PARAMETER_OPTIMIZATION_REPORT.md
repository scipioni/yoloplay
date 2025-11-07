# SVM Parameter Optimization Report

## Dataset Analysis
- Dataset: `/lab/yoloplay/data/rooms/termica.csv`
- Number of samples: 4,963
- Number of features: 34 (17 x,y pose keypoints)
- Label distribution: 4,963 samples with label 0 (normal poses)

## Grid Search Results

### Best Parameters Overall (Extended Search)
- **nu**: 0.15
- **kernel**: sigmoid
- **gamma**: 0.1
- **Score**: 186.4627

### Best Parameters (Original Script Grid)
- **nu**: 0.2
- **kernel**: rbf
- **gamma**: 0.01
- **Score**: 28.3984

### Top 5 Parameter Combinations
1. gamma=0.1, kernel=sigmoid, nu=0.15 -> Score: 186.4627
2. gamma=0.1, kernel=sigmoid, nu=0.1 -> Score: 134.1301
3. gamma=0.1, kernel=sigmoid, nu=0.05 -> Score: 71.5196
4. gamma=0.01, kernel=rbf, nu=0.2 -> Score: 28.3984 (original grid)
5. gamma=0.01, kernel=rbf, nu=0.15 -> Score: 23.9708 (original grid)

## Model Performance
- The sigmoid kernel with gamma=0.1 achieved the highest scores
- Within the RBF kernel family, gamma=0.01 and nu=0.2 performed best
- With optimal parameters (nu=0.2), approximately 20% of training samples are marked as anomalies
  - This is expected behavior for One-Class SVM, as nu parameter upper bounds the fraction of outliers

## Recommendations

### For Best Performance (Extended Search)
```bash
python -m yoloplay.svm --csv data/rooms/termica.csv --nu 0.15 --kernel sigmoid --gamma 0.1
```

### For Consistency with Original Codebase (Grid Search)
```bash
python -m yoloplay.svm --csv data/rooms/termica.csv --nu 0.2 --kernel rbf --gamma 0.01 --grid-search
```

### Key Insights
1. The sigmoid kernel with higher gamma values (0.1) performs exceptionally well on this dataset
2. The RBF kernel with gamma=0.01 also provides good performance and is more conventional
3. The nu parameter around 0.15-0.2 works well, marking approximately 15-20% of training data as anomalies
4. Linear and polynomial kernels performed poorly on this dataset