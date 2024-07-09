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
    plt.title('Spider Chart for LLM Results', size=15, color='black', y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    # Show the plot
    plt.show()

# Example data
mean_list = [
    [3.4375, 3.166666666666667, 3.7222222222222223, 2.78125, 3.975],
    [3.53125, 3.5, 3.5277777777777777, 2.96875, 4.1],
    [3, 3.16, 3.25,2.8, 3.22],
    [3.0625, 3.0555555555555554, 3.0833333333333335, 2.71875, 3.325],
    [3.25, 3.64, 3.45, 3.32, 3.92]
]
llms_name = ["Gpt-4", "Gpt-3.5-turbo",'Gpt-4-Turbo', "gpt-4o" ,"Crowd_data"]
cat_list = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']

# Call the function to plot the spider chart
plot_spider_chart(mean_list, llms_name, cat_list)
