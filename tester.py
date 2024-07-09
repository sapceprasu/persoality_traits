# # import matplotlib.pyplot as plt

# # # Example responses for three individuals
# # responses = {
# #     'gpt-3.5-turbo': {'Openness': 4, 'Conscientiousness': 3, 'Extraversion': 2, 'Agreeableness': -5, 'Neuroticism': 3},
# #     'llama 2': {'Openness': 5, 'Conscientiousness': 4, 'Extraversion': 3, 'Agreeableness': -3, 'Neuroticism': 2},
# #     'llama 3': {'Openness': 3, 'Conscientiousness': 2, 'Extraversion': 4, 'Agreeableness': 4, 'Neuroticism': 4},
# #     'gpt-4': {'Openness': -4, 'Conscientiousness': 3, 'Extraversion': 2, 'Agreeableness': -5, 'Neuroticism': 3},
# #     'mistral': {'Openness': -5, 'Conscientiousness': -4, 'Extraversion': 3, 'Agreeableness': 3, 'Neuroticism': 2},
# #     'xyz': {'Openness': -3, 'Conscientiousness': 2, 'Extraversion': -4, 'Agreeableness': 4, 'Neuroticism': 4}
# # } 

# # # # Calculate total scores for each personality trait for each individual
# # # for individual, traits in responses.items():
# # #     total_openness = traits['Openness']
# # #     total_conscientiousness = traits['Conscientiousness']
# # #     total_extraversion = traits['Extraversion']
# # #     total_agreeableness = traits['Agreeableness']
# # #     total_neuroticism = traits['Neuroticism']
    
# # #     # Plot points on graph
# # #     plt.scatter(total_extraversion, total_agreeableness, label=individual)

# # # # Add labels and legend
# # # plt.xlabel('Extraversion')
# # # plt.ylabel('Agreeableness')
# # # plt.title('Personality Traits')
# # # plt.legend()

# # # # Customize grid lines
# # # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# # # # Make the x and y axis lines through the origin darker
# # # plt.axhline(0, color='black',linewidth=1.5)
# # # plt.axvline(0, color='black',linewidth=1.5)

# # # # Show plot
# # # plt.show()



# # import matplotlib.pyplot as plt
# # from sklearn.decomposition import PCA
# # import numpy as np

# # # # Example responses for three individuals
# # # responses = {
# # #     'Individual 1': {'Openness': 4, 'Conscientiousness': 3, 'Extraversion': 2, 'Agreeableness': 5, 'Neuroticism': 3},
# # #     'Individual 2': {'Openness': 5, 'Conscientiousness': 4, 'Extraversion': 3, 'Agreeableness': 3, 'Neuroticism': 2},
# # #     'Individual 3': {'Openness': 3, 'Conscientiousness': 2, 'Extraversion': 4, 'Agreeableness': 4, 'Neuroticism': 4}
# # # }

# # # Convert responses to a numpy array
# # X = np.array([list(individual.values()) for individual in responses.values()])

# # # Perform PCA for dimensionality reduction
# # pca = PCA(n_components=2)
# # X_pca = pca.fit_transform(X)

# # # Plot the reduced-dimensional data
# # plt.scatter(X_pca[:, 0], X_pca[:, 1])

# # # Add labels for each point
# # for i, label in enumerate(responses.keys()):
# #     plt.text(X_pca[i, 0], X_pca[i, 1], label)

# # # Add labels and title
# # plt.xlabel('Principal Component 1')
# # plt.ylabel('Principal Component 2')
# # plt.title('PCA Plot of Personality Traits')

# # plt.legend()

# # # Show plot
# # # Customize grid lines
# # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# # # Make the x and y axis lines through the origin darker
# # plt.axhline(0, color='black', linewidth=1.5)
# # plt.axvline(0, color='black', linewidth=1.5)
# # plt.show()
# import matplotlib.pyplot as plt

# # Sample data: [(stress_level, perceived_control), ...]
# data = [(24, 2)]  # John's data

# # Split data into X and Y
# stress_levels = [point[0] for point in data]
# perceived_control = [point[1] for point in data]

# # Create plot
# plt.figure(figsize=(10, 6))
# plt.scatter(stress_levels, perceived_control, c='red', edgecolors='k')

# # Add labels and title
# plt.axhline(0, color='grey', linewidth=0.5)
# plt.axvline(13, color='grey', linewidth=0.5)  # Assuming 0-13 is low stress
# plt.axvline(26, color='grey', linewidth=0.5)  # Assuming 14-26 is moderate stress
# plt.xlabel('Stress Level (Low to High)')
# plt.ylabel('Perceived Control (Low to High)')
# plt.title('Perceived Stress Scale Results')
# plt.grid(True)

# # Display plot
# plt.show()



# import matplotlib.pyplot as plt

# # Sample data for plotting
# # [(self_esteem_score, perceived_stress_score)]
# data = [(20, 24)]  # Jane Doe's data

# # Split data into X and Y
# self_esteem_scores = [point[0] for point in data]
# perceived_stress_scores = [point[1] for point in data]

# # Create plot
# plt.figure(figsize=(12, 8))
# plt.scatter(self_esteem_scores, perceived_stress_scores, c='blue', edgecolors='k', s=100)

# # Main Axes
# plt.axhline(y=20, color='black', linewidth=1.5)
# plt.axvline(x=20, color='black', linewidth=1.5)

# # Subdividing axes into regions
# plt.axhline(y=13, color='grey', linewidth=0.5, linestyle='--')
# plt.axhline(y=26, color='grey', linewidth=0.5, linestyle='--')
# plt.axvline(x=15, color='grey', linewidth=0.5, linestyle='--')
# plt.axvline(x=24, color='grey', linewidth=0.5, linestyle='--')

# # Add labels and title
# plt.xlabel('Self-Esteem Level (Low to High)', fontsize=14)
# plt.ylabel('Perceived Stress Level (Low to High)', fontsize=14)
# plt.title('Rosenberg Self-Esteem and Perceived Stress Scale Results', fontsize=16)
# plt.grid(True, linestyle='--', alpha=0.7)

# # Add detailed labels for axes
# plt.text(10, 41, 'Low Self-Esteem', horizontalalignment='center', fontsize=12, color='red')
# plt.text(20, 41, 'Moderate Self-Esteem', horizontalalignment='center', fontsize=12, color='orange')
# plt.text(27, 41, 'High Self-Esteem', horizontalalignment='center', fontsize=12, color='green')
# plt.text(-3, 10, 'Low Stress', verticalalignment='center', fontsize=12, color='green')
# plt.text(-3, 20, 'Moderate Stress', verticalalignment='center', fontsize=12, color='orange')
# plt.text(-3, 33, 'High Stress', verticalalignment='center', fontsize=12, color='red')

# # Show plot
# plt.show()





import matplotlib.pyplot as plt

# Sample data for plotting
# [(general_self_efficacy_score, perceived_stress_score)]
data = [(40, 24)]  # Jane Doe's data

# Split data into X and Y
self_efficacy_scores = [point[0] for point in data]
perceived_stress_scores = [point[1] for point in data]

# Create plot
plt.figure(figsize=(12, 8))
plt.scatter(self_efficacy_scores, perceived_stress_scores, c='blue', edgecolors='k', s=100)

# Main Axes
plt.axhline(y=20, color='black', linewidth=1.5)
plt.axvline(x=30, color='black', linewidth=1.5)

# Subdividing axes into regions
plt.axhline(y=13, color='grey', linewidth=0.5, linestyle='--')
plt.axhline(y=26, color='grey', linewidth=0.5, linestyle='--')
plt.axvline(x=20, color='grey', linewidth=0.5, linestyle='--')
plt.axvline(x=35, color='grey', linewidth=0.5, linestyle='--')

# Set limits to ensure all labels are visible
plt.xlim(0, 40)
plt.ylim(0, 40)

# Add labels and title
plt.xlabel('General Self-Efficacy Level (Low to High)', fontsize=14)
plt.ylabel('Perceived Stress Level (Low to High)', fontsize=14)
plt.title('General Self-Efficacy and Perceived Stress Scale Results', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)

# Add detailed labels for axes
plt.text(10, 38, 'Low Self-Efficacy', horizontalalignment='center', fontsize=12, color='red')
plt.text(25, 38, 'Moderate Self-Efficacy', horizontalalignment='center', fontsize=12, color='orange')
plt.text(35, 38, 'High Self-Efficacy', horizontalalignment='center', fontsize=12, color='green')
plt.text(-2, 6, 'Low Stress', verticalalignment='center', fontsize=12, color='green')
plt.text(-2, 20, 'Moderate Stress', verticalalignment='center', fontsize=12, color='orange')
plt.text(-2, 34, 'High Stress', verticalalignment='center', fontsize=12, color='red')

# Show plot
plt.show()
