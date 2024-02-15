import os
from dotenv import load_dotenv
import openai
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# Function to load prompts from a separate file
def load_prompts(file_path):
    with open(file_path, 'r') as file:
        prompts = file.read().split('\n\n')
    return prompts

# Function to analyze sentiment
def analyze_sentiment(text, system_prompt):
    global api_call_count

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Analyze the following product review and determine if the sentiment is: positive or negative. 
                                            Return answer in single word as either positive or negative: {text}"""}
        ]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages, 
            max_tokens=1, 
            n=1, 
            stop=None, 
            temperature=0)

        api_call_count += 1

        response_text = response.choices[0].message.content.strip().lower()

        return response_text

    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return "error"

# Function to find the best prompt based on F1 score
def find_best_prompt_based_on_f1(df, actual_sentiment_col, prompt_cols):
    f1_scores = {}

    for col in prompt_cols:
        temp_df = df[(df[actual_sentiment_col].isin(['positive', 'negative'])) & (df[col].isin(['positive', 'negative']))]
        score = f1_score(temp_df[actual_sentiment_col], temp_df[col], pos_label='positive', average='binary')
        f1_scores[col] = score

    max_f1 = max(f1_scores.values())
    best_prompts = [prompt for prompt, score in f1_scores.items() if score == max_f1]

    if len(best_prompts) == 1:
        print(f"The best prompt based on F1 score is {best_prompts[0]} with an F1 score of {max_f1:.4f}.")
    else:
        print(f"The best prompts based on F1 score are {', '.join(best_prompts)} with an F1 score of {max_f1:.4f}.")

# Main execution
if __name__ == "__main__":
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("No OpenAI API key found. Check your .env file.")
    openai.api_key = openai_api_key

    client = openai.OpenAI()
    api_call_count = 0

    df = pd.read_csv("IMDB Dataset.csv")
    df_small = df.iloc[99:120]

    prompt_file_path = 'prompts.txt' # Enter correct path here 
    prompts = load_prompts(prompt_file_path)

    for i, prompt in enumerate(prompts, start=1):
        column_name = f"prompt{i}"
        df_small[column_name] = df_small["review"].apply(lambda x: analyze_sentiment(x, prompt))
    
    print(f"Total API calls made: {api_call_count}")

   
