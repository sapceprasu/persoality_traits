import matplotlib.pyplot as plt
import numpy as np

# Transform responses to a -2 to 2 scale (assuming original scale was 1 to 10)
def transform_scores(scores):
    return [((score - 1) / 9) * 4 - 2 for score in scores]

# Transform hypothetical responses
transformed_responses = {
    'gpt-3.5-turbo': {
        'Openness': transform_scores([8, 9, 8, 7, 8, 7, 9, 8, 7, 8]),
        'Conscientiousness': transform_scores([4, 5, 4, 5, 4, 5, 4, 5, 4, 5]),
        'Extraversion': transform_scores([5, 6, 5, 4, 5, 6, 5, 4, 5, 6]),
        'Agreeableness': transform_scores([7, 8, 7, 6, 7, 8, 7, 6, 7, 8]),
        'Neuroticism': transform_scores([3, 4, 3, 2, 3, 4, 3, 2, 3, 4])
    },
    'llama 2': {
        'Openness': transform_scores([6, 6, 7, 6, 6, 7, 5, 6, 7, 6]),
        'Conscientiousness': transform_scores([7, 7, 8, 7, 8, 7, 8, 7, 8, 7]),
        'Extraversion': transform_scores([8, 8, 7, 8, 7, 8, 7, 8, 7, 8]),
        'Agreeableness': transform_scores([5, 6, 5, 6, 5, 6, 5, 6, 5, 6]),
        'Neuroticism': transform_scores([4, 5, 4, 5, 4, 5, 4, 5, 4, 5])
    },
    'llama 3': {
        'Openness': transform_scores([5, 4, 5, 6, 5, 4, 5, 6, 4, 5]),
        'Conscientiousness': transform_scores([3, 4, 3, 4, 3, 4, 3, 4, 3, 4]),
        'Extraversion': transform_scores([6, 5, 6, 7, 6, 5, 6, 7, 6, 5]),
        'Agreeableness': transform_scores([6, 5, 6, 5, 6, 5, 6, 5, 6, 5]),
        'Neuroticism': transform_scores([7, 6, 7, 6, 7, 6, 7, 6, 7, 6])
    },
    'gpt-4': {
        'Openness': transform_scores([3, 4, 3, 3, 2, 4, 3, 3, 3, 4]),
        'Conscientiousness': transform_scores([6, 6, 7, 6, 5, 6, 7, 6, 7, 6]),
        'Extraversion': transform_scores([4, 3, 4, 4, 3, 4, 4, 3, 4, 3]),
        'Agreeableness': transform_scores([3, 4, 3, 4, 3, 4, 3, 4, 3, 4]),
        'Neuroticism': transform_scores([5, 4, 5, 4, 5, 4, 5, 4, 5, 4])
    },
    'Mistral': {
        'Openness': transform_scores([7, 8, 7, 6, 7, 8, 7, 6, 7, 8]),
        'Conscientiousness': transform_scores([8, 9, 8, 8, 9, 8, 8, 9, 8, 9]),
        'Extraversion': transform_scores([2, 2, 3, 2, 3, 2, 2, 3, 2, 3]),
        'Agreeableness': transform_scores([8, 7, 8, 7, 8, 7, 8, 7, 8, 7]),
        'Neuroticism': transform_scores([2, 2, 3, 2, 2, 2, 2, 3, 2, 2])
    }
}

# Calculate average scores for each individual
average_scores = {
    individual: {trait: np.mean(scores) for trait, scores in traits.items()}
    for individual, traits in transformed_responses.items()
}

# Plot points on graph for Openness vs Conscientiousness
for individual, traits in average_scores.items():
    plt.scatter(traits['Openness'], traits['Conscientiousness'], label=individual)
    # Annotate each point with individual's description
    plt.text(traits['Openness'] + 0.1, traits['Conscientiousness'] + 0.1, 
             f'{individual}\nOpenness: {traits["Openness"]:.2f}\nConscientiousness: {traits["Conscientiousness"]:.2f}',
             fontsize=9)

# Add labels and legend
plt.xlabel('Openness: Creative and open-minded vs. conventional and down-to-earth')
plt.ylabel('Conscientiousness: Organized and dependable vs. careless and easy-going')
plt.title('Big Five Personality Traits (Openness vs Conscientiousness)')
plt.legend()

# Customize grid lines
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# Make the x and y axis lines through the origin darker
plt.axhline(0, color='black', linewidth=1.5)
plt.axvline(0, color='black', linewidth=1.5)

# Show plot
plt.show()

