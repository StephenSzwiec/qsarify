## Usage Example

```python 
from qsarify.modeling.regressors import UnifiedModelComparison
from qsarify.modeling.ga import GeneticConfig
from qsarify.preprocessing.preprocessing import PreprocessingConfig

# Configure GA-MLR and preprocessing

genetic_config = GeneticConfig(max_vars=10, top_k_models=3) 
preprocessing_config = PreprocessingConfig(correlation_threshold=0.95) 

# Run comprehensive comparison 
comparison = UnifiedModelComparison(genetic_config=genetic_config)
results = comparison.run_comprehensive_comparison(
    X_raw, y_raw, preprocessing_config=preprocessing_config
)

# Results contain:
# - MLR models with GA feature selection
# - Non-MLR models with gradient-based hyperparameter optimization
# - Comprehensive QSAR statistics for all models 
# - Head-to-head performance rankings
```

