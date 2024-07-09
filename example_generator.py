import openai
import os
import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import time
from tqdm import tqdm
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    BitsAndBytesConfig,
    StoppingCriteriaList,
    pipeline
)
import pdb


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))

def chat(
    model,           # gpt-4, gpt-4-0314, gpt-4-32k, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-0301
    messages,        # [{"role": "system"/"user"/"assistant", "content": "Hello!", "name": "example"}]
    temperature=0,   # [0, 2]: Lower values -> more focused and deterministic; Higher values -> more random.
    n=1,             # Chat completion choices to generate for each input message.
    max_tokens=1024, # The maximum number of tokens to generate in the chat completion.
    delay=1          # Seconds to sleep after each request.
):
    time.sleep(delay)
    # pdb.set_trace()
    print("the prompt is", messages)
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        n=n,
        max_tokens=max_tokens
    )
    
    # print("the actual response:", response)
    if n == 1:
        return response['choices'][0]['message']['content']
    else:
        return [i['message']['content'] for i in response['choices']]


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion(
    model,           # text-davinci-003, text-davinci-002, text-curie-001, text-babbage-001, text-ada-001
    prompt,          # The prompt(s) to generate completions for, encoded as a string, array of strings, array of tokens, or array of token arrays.
    temperature=0,   # [0, 2]: Lower values -> more focused and deterministic; Higher values -> more random.
    n=1,             # Completions to generate for each prompt.
    max_tokens=1024, # The maximum number of tokens to generate in the chat completion.
    delay=1         # Seconds to sleep after each request.
):
    time.sleep(delay)
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        n=n,
        max_tokens=max_tokens
    )
    
    if n == 1:
        return response['choices'][0]['text']
    else:
        response = response['choices']
        response.sort(key=lambda x: x['index'])
        return [i['text'] for i in response['choices']]
    
def convert_text_to_binary(input_list):
    # Extract the single string from the input list
    input_text = input_list[0]
    
    # Split the input text into individual responses
    responses = input_text.split(' ')
    
    # Initialize an empty list to hold the results
    results = []
    
    # Iterate over the responses
    for i in range(0, len(responses), 2):
        # Extract the number and the answer
        # pdb.set_trace()
        number = responses[i].strip('.')
        answer = responses[i+1]
        
        # Convert 'Yes' to '1' and 'No' to '0'
        binary_answer = '1' if answer == 'Yes' else '0'
        
        # Append the formatted result to the results list
        results.append(f"{number}:{binary_answer}\n")
    
    # Join the results list into a single string with spaces separating the values
    formatted_result = ' '.join(results)
    
    return formatted_result




    # prompt response generation from the other hf models.
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def pipeline_generator(
    model,           # text-davinci-003, text-davinci-002, text-curie-001, text-babbage-001, text-ada-001
    messages,          # The prompt(s) to generate completions for, encoded as a string, array of strings, array of tokens, or array of token arrays.
    temperature=0.001,   # [0, 2]: Lower values -> more focused and deterministic; Higher values -> more random.
    n=1,             # Completions to generate for each prompt.
    max_tokens=1024, # The maximum number of tokens to generate in the chat completion.
    delay=1         # Seconds to sleep after each request.
):
    time.sleep(delay)

    if model == "Llama-2-7b-chat-hf":
        checkpoint = "meta-llama/Llama-2-7b-chat-hf"
        task = "text-generation"
    elif model == "flan-t5-base":
        checkpoint = "google/flan-t5-base" 
        task = "text2text-generation"
    elif model == "ctrl":
        checkpoint = "Salesforce/ctrl"
        task = "text-generation"
    elif model == "deepmount-rag":
        checkpoint = "DeepMount00/Minerva-3B-base-RAG"
        task = "text-generation"
    else: 
        print("No model was identified. Please Provide a valid working model")


    token = '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'
    pdb.set_trace()

    # model = AutoModelForSeq2SeqLM.from_pretrained(
    #     pretrained_model_name_or_path = checkpoint,

    # )
    # tokenizer = AutoTokenizer.from_pretrained(
    #     pretrained_model_name_or_path = checkpoint
    # )


    #         # Greedy Search
    # input_ids = tokenizer(
    #     f"{messages[0]}{messages[1]}", return_tensors="pt"
    # )

    # response = model.generate(input_ids)

    # completion = tokenizer.decode(
    #         response.sequences[0], skip_special_tokens=True
    #     ).strip()

    # print("if the tokeizatio was corerect", completion)


#  working pipeline commented to test another code. 

    pipe = pipeline(task = task , model = checkpoint, token = token, max_new_tokens = 200)
    # print("the prompt is: ", prompt)
    response = pipe(
        # f"{messages[0]}{messages[1]}",
        messages,
        temperature=temperature,
        do_sample = True
    )
    
    print("the resposne:", response)
    # if n == 1:
    #     return response['choices'][0]['text']
    # else:
    print("inside the generated response",[response[0]["generated_text"]])

    resulted_output = convert_text_to_binary([response[0]["generated_text"]])
    return resulted_output
    # response = response['choices']
    # response.sort(key=lambda x: x['index'])
    # return [i['text'] for i in response['choices']]




def convert_results(result, column_header):
    result = result.strip()  # Remove leading and trailing whitespace
    try:
        result_list = [int(element.strip()[-1]) for element in result.split('\n') if element.strip()]
    except:
        result_list = ["" for element in result.split('\n')]
        print(f"Unable to capture the responses on {column_header}.")
        
    return result_list


def example_generator(questionnaire, args):
    testing_file = args.testing_file
    model = args.model
    records_file = args.name_exp if args.name_exp is not None else model

    openai.organization = args.openai_organization
    openai.api_key = args.openai_key

    # Read the existing CSV file into a pandas DataFrame
    df = pd.read_csv(testing_file)
    # pdb.set_trace()
    # print("the testing file", testing_file)
    # Find the columns whose headers start with "order"
    order_columns = [col for col in df.columns if col.startswith("order")]
    # result for this is: ['order-0','order-1']

    shuffle_count = 0
    insert_count = 0
    total_iterations = len(order_columns) * args.test_count

    with tqdm(total=total_iterations) as pbar:
        for i, header in enumerate(df.columns):
            if header in order_columns:
                # Find the index of the previous column
                # because the question or the prompt comes before the
                # order-0 and order-1 
                questions_column_index = i - 1
                shuffle_count += 1
                
                # Retrieve the column data as a string
                questions_list = df.iloc[:, questions_column_index].astype(str)
                separated_questions = [questions_list[i:i+30] for i in range(0, len(questions_list), 30)]  
                questions_list = ['\n'.join([f"{i+1}.{q.split('.')[1]}" for i, q in enumerate(questions)]) for j, questions in enumerate(separated_questions)]
                # print("the question list", questions_list)

                for k in range(args.test_count):
                    
                    df = pd.read_csv(testing_file)
                    
                    # Insert the updated column into the DataFrame with a unique identifier in the header
                    column_header = f'shuffle{shuffle_count - 1}-test{k}'
                    
                    while(True):
                        result_string_list = []
                        previous_records = []
                        
                        for questions_string in questions_list:
                            # print("THIS IS THE QUESTIONIARE>>>", questions_string)
                            result = ''
                            if model == 'text-davinci-003':
                                inputs = questionnaire["inner_setting"].replace('Format: \"index: score\"', 'Format: \"index: score\\\n\"') + questionnaire["prompt"] + '\n' + questions_string
                                
                                result = completion(model, inputs)
                            elif model in ['gpt-3.5-turbo', 'gpt-4', 'gpt-3.5-turbo-0301', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-1106', 'gpt-4-turbo', "gpt-4o" ]:
                                inputs = previous_records + [
                                    {"role": "system", "content": questionnaire["inner_setting"]},
                                    {"role": "user", "content": questionnaire["prompt"] + '\n' + questions_string}
                                ]
                                
                                result = chat(model, inputs)
                                print("results from gpt 3.5 turbo", result)
                                previous_records.append({"role": "user", "content": questionnaire["prompt"] + '\n' + questions_string})
                                previous_records.append({"role": "assistant", "content": result})
                            elif model in ['Llama-2-7b-chat-hf', 'flan-t5-base', 'ctrl', 'deepmount-rag']:
                                

                                # pdb.set_trace()
                                # inputs = previous_records + [
                                #     {"role": "user", "content": questionnaire["prompt"] + '\n' + questions_string}
                                # ]
                                # prompt_entry = "Prompt: You can only reply numbers from 0 to 1 in the following statements. Please respond to each question by using the number '1' to indicate 'YES' and '0' to indicate 'NO'. There are no right or wrong answers, and no trick questions. Work quickly and do not think too long about the exact meaning of the questions. Here are the statements, score them one by one:"
                                # # print("inputs>", inputs)

                                # raw_inputs = [sentence.strip() for sentence in inputs[0]['content'].split('\n')[1:-1]]
                                # raw_inputs.insert(0,prompt_entry)
                                # print(raw_inputs)
                                inputs = previous_records + [
                                    {"role": "system", "content": questionnaire["inner_setting"]},
                                    {"role": "user", "content": questionnaire["prompt"] + '\n' + questions_string}
                                ]
                                # print("inpoutssdsdsdf",inputs)
                                
                                # Concatenate 'role' and 'content' for each dictionary into a single string and join them into one string
                                combined_string = ' '.join([f"role: {item['role']}, content: {item['content']}" for item in inputs])
                                
                                # Creating the result array with a single joined string
                                result_array = [combined_string]

                                # print("the result array:", result_array)

                                result = pipeline_generator(model = model, messages = result_array)
                                previous_records.append({"role": "user", "content": questionnaire["prompt"] + '\n' + questions_string})
                                previous_records.append({"role": "assistant", "content": result})


                            else:
                                raise ValueError("The model is not supported or does not exist.")
                        
                            result_string_list.append(result.strip())
                        
                            # Write the prompts and results to the file
                            os.makedirs("prompts", exist_ok=True)
                            os.makedirs("responses", exist_ok=True)

                            with open(f'prompts/{records_file}-{questionnaire["name"]}-shuffle{shuffle_count - 1}.txt', "a") as file:
                                file.write(f'{inputs}\n====\n')
                            with open(f'responses/{records_file}-{questionnaire["name"]}-shuffle{shuffle_count - 1}.txt', "a") as file:
                                file.write(f'{result}\n====\n')
                        # pdb.set_trace()
                        result_string = '\n'.join(result_string_list)

                        # print("this is the result string",)
                        # pdb.set_trace()
                        # this will retun the list of the results only, only the ratings no 
                        # indices and any other explanations.
                        #the colummn header defines the shuffle and test counts that are necessary
                        # like shuffle 0 test 0 
                        result_list = convert_results(result_string, column_header)
                        
                        try:
                            if column_header in df.columns:
                                df[column_header] = result_list
                            else:
                                df.insert(i + insert_count + 1, column_header, result_list)
                                insert_count += 1
                            break
                        except:
                            print(f"Unable to capture the responses on {column_header}.")

                    # Write the updated DataFrame back to the CSV file
                    df.to_csv(testing_file, index=False)
                    
                    pbar.update(1)
