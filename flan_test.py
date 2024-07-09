# import pdb
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# import pdb



# # Define the model and checkpoint mapping
# model_name = "flan-t5-base"  # The model you want to use
# messages = [{'role': 'system', 'content': 'You are a helpful assistant who can only reply numbers from 0 to 1. Format: "statement index: score".'},{'role': 'user', 'content': "You can only reply numbers from 0 to 1 in the following statements. Please respond to each question by using the number '1' to indicate 'YES' and '0' to indicate 'NO'. There are no right or wrong answers, and no trick questions. Work quickly and do not think too long about the exact meaning of the questions. Here are the statements, score them one by one:\n1. Do you have many different hobbies?\n2. Do you stop to think things over before doing anything?\n3. Does your mood often go up and down?\n4. Have you ever taken the praise for something you knew someone else had really done?\n5. Do you take muchby helping yourself to more than your share of anything?\n11. Are you rather lively?\n12. Would it upset you a lot to see a child or an animal suffer?\n13. Do you often worry about things you should not have done or said?\n14. Do you dislike people who don't know how to behave themselves?\n15. If you say you will do something, do you always keep your promise no matter how inconvenient it might be?\n16. Can you usually let yourself go and enjoy yourself at a lively party?\n17. Are you an irritable person?\n18. Should people always respect the law?\n19. Have you ever blamed someone for doing something you knew was really your faulocial occasions?\n25. Would you take drugs which may have strange or dangerous effects?\n26. Do you often feel 'fed-up'?\n27. Have you ever taken anything (even a pin or button) that belonged to someone else?\n28. Do you like going out a lot?\n29. Do you prefer to go your own way rather than act by the rules?\n30. Do you enjoy hurting people you love?"}]  # Example messages

# # Determine the appropriate checkpoint and task
# if model_name == "Llama-2-7b-chat-hf":
#     checkpoint = "meta-llama/Llama-2-7b-chat-hf"
#     task = "text-generation"
# elif model_name == "flan-t5-base":
#     checkpoint = "google/flan-t5-base" 
#     task = "text2text-generation"
# elif model_name == "ctrl":
#     checkpoint = "Salesforce/ctrl"
#     task = "text-generation"
# elif model_name == "deepmount-rag":
#     checkpoint = "DeepMount00/Minerva-3B-base-RAG"
#     task = "text-generation"
# else: 
#     print("No model was identified. Please provide a valid working model")
#     checkpoint = None
#     task = None

# if checkpoint:
#     # Initialize the model and tokenizer
#     if task == "text2text-generation":
#         model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
#     else:
#         model = AutoModelForCausalLM.from_pretrained(checkpoint)

#     tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
#     # Tokenize the input messages
#     # input_text = " ".join(messages)
#     input_ids = tokenizer( f"{messages[0]}{messages[1]}", return_tensors="pt").input_ids
    
#     pdb.set_trace()
#     # Generate response
#     response_ids = model.generate(input_ids)
#     response = tokenizer.decode(response_ids[0], skip_special_tokens=True)

#     print("Response:", response)
# else:
#     print("Checkpoint or task was not set properly.")



import pdb
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Define the model and checkpoint mapping
model_name = "flan-t5-base"  # The model you want to use
messages = [
    {'role': 'system', 'content': 'You are a helpful assistant who can answer my questions".'},
    {'role': 'user', 'content': "Do you know how to give ratings to a certain statement? Can you do them if I give you a few questions ?Answer in detail please"}
]  # Example messages
# messages = [
#     {'role': 'system', 'content': 'You are a helpful assistant who can only reply numbers from 0 to 1. Format: "statement index: score".'},
#     {'role': 'user', 'content': "You can only reply numbers from 0 to 1 in the following statements. Please respond to each question by using the number '1' to indicate 'YES' and '0' to indicate 'NO'. There are no right or wrong answers, and no trick questions. Work quickly and do not think too long about the exact meaning of the questions. Here are the statements, score them one by one:\n1. Do you have many different hobbies?\n2. Do you stop to think things over before doing anything?\n3. Does your mood often go up and down?\n4. Have you ever taken the praise for something you knew someone else had really done?\n5. Do you take much more than your share of anything?\n11. Are you rather lively?\n12. Would it upset you a lot to see a child or an animal suffer?\n13. Do you often worry about things you should not have done or said?\n14. Do you dislike people who don't know how to behave themselves?\n15. If you say you will do something, do you always keep your promise no matter how inconvenient it might be?\n16. Can you usually let yourself go and enjoy yourself at a lively party?\n17. Are you an irritable person?\n18. Should people always respect the law?\n19. Have you ever blamed someone for doing something you knew was really your fault?\n25. Would you take drugs which may have strange or dangerous effects?\n26. Do you often feel 'fed-up'?\n27. Have you ever taken anything (even a pin or button) that belonged to someone else?\n28. Do you like going out a lot?\n29. Do you prefer to go your own way rather than act by the rules?\n30. Do you enjoy hurting people you love?"}
# ]  # Example messages

# Determine the appropriate checkpoint and task
if model_name == "Llama-2-7b-chat-hf":
    checkpoint = "meta-llama/Llama-2-7b-chat-hf"
    task = "text-generation"
elif model_name == "flan-t5-base":
    checkpoint = "google/flan-t5-base"
    task = "text2text-generation"
elif model_name == "ctrl":
    checkpoint = "Salesforce/ctrl"
    task = "text-generation"
elif model_name == "deepmount-rag":
    checkpoint = "DeepMount00/Minerva-3B-base-RAG"
    task = "text-generation"
else:
    print("No model was identified. Please provide a valid working model")
    checkpoint = None
    task = None

if checkpoint:
    # Initialize the model and tokenizer
    if task == "text2text-generation":
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # Format the input messages to include roles
    input_text = ""
    for message in messages:
        role = message['role']
        content = message['content']
        if role == 'system':
            input_text += f"System: {content}\n"
        elif role == 'user':
            input_text += f"User: {content}\n"

    # Tokenize the input messages
    inputs = tokenizer(input_text, return_tensors="pt")

    pdb.set_trace()

    # Generate response
    response_ids = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_new_tokens = 200)
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)

    print("Response:", response)
else:
    print("Checkpoint or task was not set properly.")
