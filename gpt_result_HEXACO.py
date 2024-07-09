import matplotlib.pyplot as plt
import numpy as np

def plot_spider_chart(mean_list, llms_name, cat_list):
    # Number of variables
    num_vars = len(cat_list)

    # Compute angle for each category
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Make the plot circular
    angles += angles[:1]

    # Initialize the spider plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Plot each LLM result
    for i, values in enumerate(mean_list):
        values += values[:1]  # Complete the loop for circular plot
        ax.fill(angles, values, alpha=0.25, label=llms_name[i])
        ax.plot(angles, values, linewidth=2, linestyle='solid')

    # Add labels
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cat_list, fontsize=12)

    # Add title and legend
    plt.title('Spider Chart for LLM Results HEXACO Personality Traits', size=15, color='black', y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    # Show the plot
    plt.show()

# Example data
#  these experiensts are conducted for one runs only that is one test. 

mean_list = [
    [3.5, 3.2777777777777777, 3.25, 3.4375, 3.15, 3.5],
    [3.5, 3.7222222222222223, 3.6875, 4.1875, 3.45, 4],
    [2.666666666666667, 2.666666666666667, 2.75, 2.5, 2.6500000000000004, 2.75],
    [4.5, 3.6111111111111107, 3.875, 3.9375, 3.65, 4],
    [3.19, 3.50, 2.94, 3.44, 3.92, 3.41]
]
llms_name = ["Gpt-4", "Gpt-3.5-turbo",'Gpt-4-Turbo', "gpt-4o" ,"Crowd_data"]
cat_list = ['Honesty-Humility', 'Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']

# Call the function to plot the spider chart
plot_spider_chart(mean_list, llms_name, cat_list)
