# Movie Recommendation System

## Dataset

This project uses the **Wiki Movie Plots Dataset** from Kaggle, which contains movie titles, genres, and plot summaries.

### Steps to Load the Dataset

1. Download the dataset from Kaggle: [Wiki Movie Plots Deduped] - [https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots?resource=download](url)
2. Save the file as `wiki_movie_plots_deduped.csv` in the project directory.
3. The script automatically loads and processes the dataset (reduce the sample to 200 for quick processing and save that sample file as sampled_movie_dataset.csv)

## Setup

### Prerequisites

- Python 3.8+
- VS Code (recommended)

### Installation

1. Clone the repository or download the script.
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Running the Code

### VS Code

1. Open the script in **VS Code**.
2. Run the script by clicking on the **Run** button or using the terminal.

## Results

### Example Output

```
Recommended Movies:
For input: 'I like action movies set in space with aliens'
Kaizoku Sentai Gokaiger vs. Space Sheriff Gavan: The Movie (Similarity: 0.1449)
The Roots of Heaven (Similarity: 0.0732)
The Mirror (Similarity: 0.0525)

For input: 'I enjoy movies about time travel and complex narratives'
Hotel Chelsea (Similarity: 0.1976)
One from the Heart (Similarity: 0.0334)
Seeta Rama Jananam (Similarity: 0.0299)
```

## Explanation

- **TF-IDF Vectorization** converts movie plots into numerical representations.
- **Cosine Similarity** measures how similar the input description is to each movie plot.
- The system returns the top N movies with the highest similarity scores.

## Future Improvements

- Expand the dataset for better recommendations.
- Implement user feedback-based recommendations.
- Deploy as a web app for interactive usage.
