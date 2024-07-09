# import openai
# import matplotlib.pyplot as plt
# import numpy as np
# from random import shuffle

# # Define your OpenAI API key
# openai.api_key = ''

# # Define the list of BFI questions with labels for each category
# bfi_questions = [
#     ("I see myself as someone who is reserved.", "Extraversion"),
#     ("I see myself as someone who is generally trusting.", "Agreeableness"),
#     ("I see myself as someone who tends to be lazy.", "Conscientiousness"),
#     ("I see myself as someone who is relaxed, handles stress well.", "Neuroticism"),
#     ("I see myself as someone who has few artistic interests.", "Openness to Experience"),
#     ("I see myself as someone who is outgoing, sociable.", "Extraversion"),
#     ("I see myself as someone who tends to find fault with others.", "Agreeableness"),
#     ("I see myself as someone who does a thorough job.", "Conscientiousness"),
#     ("I see myself as someone who gets nervous easily.", "Neuroticism"),
#     ("I see myself as someone who has an active imagination.", "Openness to Experience"),
#     ("I see myself as someone who tends to be quiet.", "Extraversion"),
#     ("I see myself as someone who is generally trusting of others.", "Agreeableness"),
#     ("I see myself as someone who tends to be disorganized.", "Conscientiousness"),
#     ("I see myself as someone who is easily upset.", "Neuroticism"),
#     ("I see myself as someone who is inventive.", "Openness to Experience"),
#     ("I see myself as someone who is assertive.", "Extraversion"),
#     ("I see myself as someone who tends to forgive others.", "Agreeableness"),
#     ("I see myself as someone who tends to leave a mess.", "Conscientiousness"),
#     ("I see myself as someone who rarely feels anxious.", "Neuroticism"),
#     ("I see myself as someone who is sometimes shy, introverted.", "Openness to Experience"),
#     ("I see myself as someone who is talkative.", "Extraversion"),
#     ("I see myself as someone who tends to avoid conflict.", "Agreeableness"),
#     ("I see myself as someone who pays attention to details.", "Conscientiousness"),
#     ("I see myself as someone who worries a lot.", "Neuroticism"),
#     ("I see myself as someone who has a rich vocabulary.", "Openness to Experience"),
#     ("I see myself as someone who is full of energy.", "Extraversion"),
#     ("I see myself as someone who is not interested in other people's problems.", "Agreeableness"),
#     ("I see myself as someone who tends to make a mess of things.", "Conscientiousness"),
#     ("I see myself as someone who keeps calm under pressure.", "Neuroticism"),
#     ("I see myself as someone who thinks deeply about things.", "Openness to Experience"),
#     ("I see myself as someone who has a lot of fun.", "Extraversion"),
#     ("I see myself as someone who is not interested in abstract ideas.", "Agreeableness"),
#     ("I see myself as someone who tends to do things efficiently.", "Conscientiousness"),
#     ("I see myself as someone who gets nervous easily.", "Neuroticism"),
#     ("I see myself as someone who has an active imagination.", "Openness to Experience"),
#     ("I see myself as someone who tends to be reserved.", "Extraversion"),
#     ("I see myself as someone who is generally trusting of others.", "Agreeableness"),
#     ("I see myself as someone who tends to be lazy.", "Conscientiousness"),
#     ("I see myself as someone who is easily upset.", "Neuroticism"),
#     ("I see myself as someone who is inventive.", "Openness to Experience"),
#     ("I see myself as someone who is assertive.", "Extraversion"),
#     ("I see myself as someone who tends to forgive others.", "Agreeableness"),
#     ("I see myself as someone who tends to leave a mess.", "Conscientiousness"),
#     ("I see myself as someone who rarely feels anxious.", "Neuroticism"),
#     ("I see myself as someone who is sometimes shy, introverted.", "Openness to Experience"),
# ]

# # Shuffle the questions to prevent monotony
# shuffle(bfi_questions)

# # prompt = []
# # content_prompt = ""
# # prompt_message = "Prompt: Please answer the following questions on a scale from 1 to 5:\n\n"
# # prompt.append({"role": "system", "content": prompt_message })
# # prompt.append{"content": question[0], "role": "user"} for question in bfi_questions])
# # # for question in bfi_questions:

# prompt = []
# content_prompt = "Prompt: You are a helpful assistant who can only reply numbers from 1 to 5. Format: \"statement index: score\""
# prompt.append({"role": "system", "content": content_prompt})

# # Concatenate all questions into a single string
# prompt_content = "You can only reply numbers from 1 to 5 in the following statements. Here are a number of characteristics that may or may not apply to you. Please indicate the extent to which you agree or disagree with that statement. 1 denotes 'strongly disagree', 2 denotes 'a little disagree', 3 denotes 'neither agree nor disagree', 4 denotes 'little agree', 5 denotes 'strongly agree'. Here are the statements, score them one by one:"
# questions_content = "\n".join([question[0] for question in bfi_questions])
# prompt.append({"role": "user", "content": prompt_content + questions_content})



# # Call OpenAI API to generate responses
# response = openai.ChatCompletion.create(
#   model="gpt-3.5-turbo",
#   messages=prompt,
#   max_tokens=1000,
#   n=1,
#   stop=None,
#   temperature=0
# )

# # Print the response from the OpenAI API
# print(response)

# # Get the generated response
# generated_response = response.choices[0].text.strip()

# # Map the generated response back to the original questions
# generated_scores = list(map(int, generated_response.split()))

# # Define labels for the five major scales
# labels = ['Openness to Experience', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']

# # Plot the results as a spider chart
# num_vars = len(labels)
# angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

# # Fill the area under the curve with a color
# ax.fill(angles, generated_scores[:5], color='skyblue', alpha=0.4)

# # Plot each trait as a line with markers
# ax.plot(angles, generated_scores[:5], color='blue', linewidth=2, linestyle='solid')
# ax.scatter(angles, generated_scores[:5], color='blue', s=100, marker='o')

# # Add labels and grid lines
# ax.set_yticklabels([])
# ax.set_xticks(angles)
# ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
# ax.yaxis.grid(True)

# # Add a title and legend
# plt.title('Big Five Personality Traits', size=20, color='blue', fontweight='bold')
# plt.legend(['Personality Profile'], loc='upper right')

# # Add a note about interpretation
# plt.text(0.5, 0.5, 'Higher scores indicate stronger\npersonality traits', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12)

# plt.show()




import openai
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle

# Define your OpenAI API key
# openai.api_key = ''

# Define the list of BFI questions with labels for each category
bfi_questions = [
    ("I see myself as someone who is reserved.", "Extraversion"),
    ("I see myself as someone who is generally trusting.", "Agreeableness"),
    ("I see myself as someone who tends to be lazy.", "Conscientiousness"),
    ("I see myself as someone who is relaxed, handles stress well.", "Neuroticism"),
    ("I see myself as someone who has few artistic interests.", "Openness to Experience"),
    ("I see myself as someone who is outgoing, sociable.", "Extraversion"),
    ("I see myself as someone who tends to find fault with others.", "Agreeableness"),
    ("I see myself as someone who does a thorough job.", "Conscientiousness"),
    ("I see myself as someone who gets nervous easily.", "Neuroticism"),
    ("I see myself as someone who has an active imagination.", "Openness to Experience"),
    ("I see myself as someone who tends to be quiet.", "Extraversion"),
    ("I see myself as someone who is generally trusting of others.", "Agreeableness"),
    ("I see myself as someone who tends to be disorganized.", "Conscientiousness"),
    ("I see myself as someone who is easily upset.", "Neuroticism"),
    ("I see myself as someone who is inventive.", "Openness to Experience"),
    ("I see myself as someone who is assertive.", "Extraversion"),
    ("I see myself as someone who tends to forgive others.", "Agreeableness"),
    ("I see myself as someone who tends to leave a mess.", "Conscientiousness"),
    ("I see myself as someone who rarely feels anxious.", "Neuroticism"),
    ("I see myself as someone who is sometimes shy, introverted.", "Openness to Experience"),
    ("I see myself as someone who is talkative.", "Extraversion"),
    ("I see myself as someone who tends to avoid conflict.", "Agreeableness"),
    ("I see myself as someone who pays attention to details.", "Conscientiousness"),
    ("I see myself as someone who worries a lot.", "Neuroticism"),
    ("I see myself as someone who has a rich vocabulary.", "Openness to Experience"),
    ("I see myself as someone who is full of energy.", "Extraversion"),
    ("I see myself as someone who is not interested in other people's problems.", "Agreeableness"),
    ("I see myself as someone who tends to make a mess of things.", "Conscientiousness"),
    ("I see myself as someone who keeps calm under pressure.", "Neuroticism"),
    ("I see myself as someone who thinks deeply about things.", "Openness to Experience"),
    ("I see myself as someone who has a lot of fun.", "Extraversion"),
    ("I see myself as someone who is not interested in abstract ideas.", "Agreeableness"),
    ("I see myself as someone who tends to do things efficiently.", "Conscientiousness"),
    ("I see myself as someone who gets nervous easily.", "Neuroticism"),
    ("I see myself as someone who has an active imagination.", "Openness to Experience"),
    ("I see myself as someone who tends to be reserved.", "Extraversion"),
    ("I see myself as someone who is generally trusting of others.", "Agreeableness"),
    ("I see myself as someone who tends to be lazy.", "Conscientiousness"),
    ("I see myself as someone who is easily upset.", "Neuroticism"),
    ("I see myself as someone who is inventive.", "Openness to Experience"),
    ("I see myself as someone who is assertive.", "Extraversion"),
    ("I see myself as someone who tends to forgive others.", "Agreeableness"),
    ("I see myself as someone who tends to leave a mess.", "Conscientiousness"),
    ("I see myself as someone who rarely feels anxious.", "Neuroticism"),
    ("I see myself as someone who is sometimes shy, introverted.", "Openness to Experience"),
]

# Shuffle the questions to prevent monotony
shuffle(bfi_questions)

prompt = []
content_prompt = "Prompt: You are a helpful assistant who can only reply numbers from 1 to 5. Format: \"statement index: score\""
prompt.append({"role": "system", "content": content_prompt})

# Concatenate all questions into a single string
prompt_content = "You can only reply numbers from 1 to 5 in the following statements. Here are a number of characteristics that may or may not apply to you. Please indicate the extent to which you agree or disagree with that statement. 1 denotes 'strongly disagree', 2 denotes 'a little disagree', 3 denotes 'neither agree nor disagree', 4 denotes 'little agree', 5 denotes 'strongly agree'. Here are the statements, score them one by one:\n"
questions_content = "\n".join([f"{i+1}. {question[0]}" for i, question in enumerate(bfi_questions)])
prompt.append({"role": "user", "content": prompt_content + questions_content})

# Call OpenAI API to generate responses
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=prompt,
    max_tokens=1000,
    n=1,
    stop=None,
    temperature=0
)

# Extract the response content
response_content = response.choices[0].message['content'].strip()
print(response_content)

# Parse the generated response
scores = response_content.split('\n')
scores_dict = {}
for score in scores:
    index, value = score.split(':')
    scores_dict[int(index.strip())] = int(value.strip())

# Map scores back to the original questions
category_scores = {
    "Openness to Experience": [],
    "Conscientiousness": [],
    "Extraversion": [],
    "Agreeableness": [],
    "Neuroticism": []
}

for i, (question, category) in enumerate(bfi_questions):
    category_scores[category].append(scores_dict[i + 1])

# Calculate the average score for each category
average_scores = {category: np.mean(scores) for category, scores in category_scores.items()}

# Define labels for the five major scales
labels = ['Openness to Experience', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
scores = [average_scores[label] for label in labels]

# Plot the results as a spider chart
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
scores += scores[:1]  # Repeat the first value to close the circle
angles += angles[:1]  # Repeat the first angle to close the circle

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

# Fill the area under the curve with a color
ax.fill(angles, scores, color='skyblue', alpha=0.4)

# Plot each trait as a line with markers
ax.plot(angles, scores, color='blue', linewidth=2, linestyle='solid')
ax.scatter(angles, scores, color='blue', s=100, marker='o')

# Add labels and grid lines
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
ax.yaxis.grid(True)

# Add a title and legend
plt.title('Big Five Personality Traits', size=20, color='blue', fontweight='bold')
plt.legend(['Personality Profile'], loc='upper right')

# Add a note about interpretation
plt.text(0.5, 0.5, 'Higher scores indicate stronger\npersonality traits', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12)

plt.show()
