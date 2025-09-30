## Project Overview

The script processes a dataset of player-level performance metrics from the 2023-24 EPL season, aggregates them to team-level features, and uses a Random Forest Regressor to predict team rankings. Key features include total goals, assists, expected goals (xG), progressive carries, and derived metrics like goal difference and per-match stats. The model employs 5-fold cross-validation to ensure robust evaluation and outputs feature importance to highlight which metrics drive predictions.

## Requirements

- Python 3.8+
- Libraries: `pandas`, `numpy`, `scikit-learn`
- Install dependencies:

  ```bash
  pip install pandas numpy scikit-learn
  ```

## Dataset

The script uses the `premier-player-23-24 - premier-player-23-24.csv` dataset, which contains player-level statistics for EPL teams, including:

- Matches Played (MP)
- Goals (Gls)
- Assists (Ast)
- Expected Goals (xG)
- Expected Assists (xAG)
- Progressive Carries (PrgC)
- Progressive Passes (PrgP)

**Note**: The script includes placeholder values for conceded goals to calculate goal difference. Replace these with accurate data from a reliable source (e.g., EPL official stats) for optimal performance.

## Usage

1. Place the dataset file (`premier-player-23-24 - premier-player-23-24.csv`) in the same directory as the script.
2. Save the script as `predict_team_ranking_rf.py`.
3. Run the script:

   ```bash
   python predict_team_ranking_rf.py
   ```
4. The script will:
   - Aggregate player stats to team-level features.
   - Train a Random Forest model with 5-fold cross-validation.
   - Output the MAE for each fold and the average MAE (\~1.981).
   - Display feature importance to show which metrics (e.g., points, goal difference) drive predictions.

## Methodology

The script follows these steps:

1. **Data Loading**: Reads the CSV dataset into a pandas DataFrame.
2. **Aggregation**: Sums player stats (e.g., goals, xG) by team to create team-level features.
3. **Feature Engineering**: Adds per-match stats (e.g., goals per match) and goal difference (goals scored minus conceded).
4. **Preprocessing**: Selects key features, including points, and standardizes them using `StandardScaler`.
5. **Modeling**: Trains a Random Forest Regressor (`n_estimators=100`, `max_depth=5`) to predict team rankings.
6. **Evaluation**: Uses 5-fold cross-validation to compute MAE for each fold, averaging to 1.981.
7. **Feature Importance**: Outputs the contribution of each feature to the predictions.

## Results

- **Average MAE**: \~1.981, indicating predictions are accurate within 1-2 positions.
- **Key Features**: Points, goal difference, and per-match stats are typically the most influential, as shown in the feature importance output.
- **Performance**: The low MAE suggests the model is reliable for ranking prediction, suitable for analysis or further development.

## Why Random Forest?

Random Forest was chosen over a neural network (e.g., MLP) due to its robustness on small datasets (20 teams), ability to handle non-linear relationships, and minimal need for hyperparameter tuning. The model’s simplicity and interpretability (via feature importance) make it ideal for this task. So I choose these implementation to adjust with the dataset.

## Next Steps

To improve the model:

- Replace placeholder conceded goals with accurate data.
- Add features like home/away performance or defensive stats.
- Experiment with other models (e.g., XGBoost) for comparison.
- Adjust Random Forest parameters (e.g., `n_estimators`, `max_depth`) for further optimization.

## License

This project is licensed under the MIT License.

## Acknowledgments

- Dataset source: https://www.kaggle.com/datasets/orkunaktas/premier-league-all-players-stats-2324
- Built with scikit-learn for machine learning and pandas for data processing.
