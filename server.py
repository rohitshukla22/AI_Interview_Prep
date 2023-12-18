import csv
from langchain import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import time
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router import MultiPromptChain
import json
import warnings
import random
import logging
import language_tool_python
import spacy
import math
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
logging.getLogger("language_tool_python").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
import os
import openai
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
#enter your openai api key in place of 'your_api_key'
api_key = 'your_api_key'
"""Prompt for the router chain in the multi-prompt chain."""
MULTI_PROMPT_ROUTER_TEMPLATE = """\
Given a raw text input to a language model select the model prompt best suited for \
the input. You will be given the names of the available prompts and a description of \
what the prompt is best suited for. You may NOT revise the original input even if you \
think that revising it will ultimately lead to a better response from the language \
model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
"destination": string \\ name of the prompt to use or "DEFAULT"
"next_inputs": string \\ the original input from the user
}}}}
```

REMEMBER: "destination" MUST be one of the candidate prompt names specified below OR \
it can be "DEFAULT" if the input is not well suited for any of the candidate prompts.
REMEMBER: "next_inputs" should be the original input\


<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (must include ```json at the start of the response) >>
<< OUTPUT (must end with ```) >>
"""
positive="""You are the judge in a data analyst interview satisfied with the answer of the user on the provided question.
MAKE SURE THE ANSWER ANSWERS THE QUESTION AND IS ELABORATE AND COVERING ALL ESSENTIAL POINTS.
Here's the question and asnwer respectively:
{input}
AS AN OUTPUT PROVIDE POSITIVE FEEDBACK ENCOURAGING THE USER.
OUTPUT THE TEXT AS IF YOU ARE TALKING TO THE CANDIDATE DIRECTLY.
"""
neutral="""You are the judge in a data analyst interview who is somewhat satisfied with the user's answer to the question.
MAKE SURE THE ANSWER ADDRESSES THE ISSUE IN THE QUESTION AND COVERS ALL NECESSARY POINTS THAT NEEDS TO BE COVERED. HIGHLIGHT WHAT IS MISSING.
Make sure the answer provided is directly addressesing the question and provides specific details related to where the user is lacking. If any information is missing or irrelevant, mention it in the feedback for improvement.
Here's the question, asnwer and instructions respectively:
{input}
AS AN OUTPUT PROVIDE both somewhat positive FEEDBACK and negative feedback. Also mention the IMPROVEMENT WAYS and also what can be added to the answer given by user as
feedback:  and improvement: and sample:
OUTPUT THE TEXT AS IF YOU ARE TALKING TO THE CANDIDATE DIRECTLY.
"""
negative="""You are the judge in a data analyst interview unsatisfied with the user's answer.The question fails to afddress the question and lacks information.
Provide ways where the user can improve.
Make sure the answer provided directly addresses the question and provides specific details related to where the user is going wrong. If any information is missing or irrelevant, mention it in the feedback for improvement and emphasize on it.
Here's the question, asnwer and instructions respectively:
{input}
AS AN OUTPUT PROVIDE negative feedback highlighting where improvement is needed and A SAMPLE WAY THEY CAN ANSWER THE QUESTION
as FEEDBACK: and SAMPLE ANSWER:
OUTPUT THE TEXT AS IF YOU ARE TALKING TO THE CANDIDATE DIRECTLY.
"""
prompt_infos=[
{
    "name":"positive",
    "description":"Good at analysing the answers that are long(2-3 paragraphs) and the most relevant to the question.",
    "prompt_name":positive
},
{
    "name":"neutral",
    "description":"Good at analysing the answers that are mid(3-5 sentences) and somewhat relevant to the question.",
    "prompt_name":neutral
},
{
    "name":"negative",
    "description":"good for analysing short, empty and irrelevant answers to the question.",
    "prompt_name":negative
}
]

llm=OpenAI(model_name="text-davinci-003",openai_api_key=api_key)
destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_name"]
    prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain
destination_chains

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
    )

router_chain = LLMRouterChain.from_llm(llm, router_prompt)

chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=destination_chains["neutral"],
    verbose=False,
    )

filepath1="AI_Interview_Prep.csv"
behv_data=pd.read_csv(filepath1)
x=behv_data.iloc[:,0]
filepath2="final_technical_dataset (1).csv"
technical_data=pd.read_csv(filepath2)
y=technical_data.iloc[:,0]

from flask import Flask, render_template, redirect, request, session

app = Flask(__name__)
app.secret_key = '123456'

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/behaviour')
def behaviour():
    return render_template("behaviour.html")

@app.route('/technical')
def technical():
    return render_template("technical.html")

def get_response(question,answer):
    input_data="Question:"+question+"Answer:"+answer
    response = chain.run(input=json.dumps(input_data))
    time.sleep(10)
    return response

@app.route('/feedback', methods=['GET'])
def feedback():
    return render_template('feedback.html')

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    if request.method == 'POST':
        user_feedback = request.form.get('user_feedback')
        
        # Append the feedback to a CSV file
        with open('feedback.csv', 'a', newline='') as csvfile:
            fieldnames = ['user_feedback']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write the header if the file is empty
            if csvfile.tell() == 0:
                writer.writeheader()

            # Write the feedback
            writer.writerow({'user_feedback': user_feedback})

        # Redirect the user to a thank you page or another appropriate page
        return render_template('home.html')

@app.route('/behave_start', methods=['GET', 'POST'])
def behave_start():
    if request.method == 'GET':
        # Initialize session variables to store questions and current question index
        random_index = random.randint(1, 30)
        random_question = behv_data.iloc[random_index, 0]
        session['current_question']=random_question
        # Render the template with the first question
        return render_template('behave_start.html', question=random_question)

    elif request.method == 'POST':
        # Get user answer from the form
        user_answer = request.form.get('user_answer')
        # Get the current question and increment the index

        # Call your get_response function
        response = get_response(session['current_question'], user_answer)
        clarity_score = assess_clarity(user_answer)
        readability_score = assess_readability(user_answer)
        grammar_errors = assess_grammar(user_answer)

        normalized_clarity = normalize_score(clarity_score)
        normalized_readability = normalize_score(readability_score)
        current_question=session['current_question']
        # If all questions are answered, display a summary
        return render_template('summary1.html', question=current_question, answer=user_answer,feedback=response,clarity=normalized_clarity,readability=normalized_readability,grammar=grammar_errors)

positive_responses = ["Great job! You provided a comprehensive answer that directly addressed the question.",
            "Excellent work! Your response was detailed and met all the necessary instructions.",
            "Well done! You showcased a deep understanding of the topic and followed the guidelines perfectly."]


positive_improvements = ["Consider providing more real-world examples to enrich your answer further.",
"Try organizing your points into sections for better clarity and structure.",
"You could enhance your response by diving deeper into the implications of your analysis."]

neutral_responses = ["Your answer touched upon various aspects of the question, but room for improvement exists.",
"Your response was decent, covering some instructions, but there's space for more details.",
"Good attempt, but a bit more depth in your explanation could make it stronger."]

neutral_improvements = ["Try connecting your points more cohesively to create a more fluid narrative.",
"Adding specific data or statistics can bolster the credibility of your response."
"Consider focusing on the core aspects of the question to avoid digressing."]


negative_responses =["Your answer lacked relevance to the question and missed several instructions.",
"It seems the instructions were not followed, leading to an answer that's off-topic.",
"There's a need for improvement as the response didn't directly address the question."]


negative_improvements = ["Ensure you thoroughly understand the question before providing your response.",
"Organize your thoughts better to align with the requirements of the question.",
"Review the instructions carefully and address each one specifically in your response."]

nlp = spacy.load("en_core_web_sm")

def assess_clarity(answer):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(answer)
    sentence_lengths = [len(sent) for sent in doc.sents]
    if len(sentence_lengths) > 0:
        avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)
    else:
        avg_sentence_length = 0 
    word_lengths = [len(token.text) for token in doc if not token.is_punct]
    avg_word_length = sum(word_lengths) / len(word_lengths) if len(word_lengths) > 0 else 0
    complex_word_count = len([token for token in doc if len(token.text) > 3])
    composite_score = abs(math.ceil((avg_sentence_length + avg_word_length - complex_word_count) / 3))
    return composite_score


def assess_readability(answer):
    words = len(answer.split())
    sentences = len(list(filter(lambda x: x != '', answer.split('.'))))
    syllables = 0
    for word in answer.split():
        syllables += sum([1 for vowel in word if vowel in 'aeiou'])

    if words > 0 and sentences > 0 and syllables > 0:
        readability_score = (
            0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59
        )
    else:
        readability_score = 0  # or any default value you prefer

    return readability_score


def assess_grammar(answer):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(answer)
    return len(matches)  # Number of detected grammar errors

def normalize_score(score):
    # Scale the score to be between 0 and 10 using min-max normalization
    min_score = 0  # Minimum possible score
    max_score = 10  # Maximum possible score

    # Ensure the score is within the range
    if score < min_score:
        score = min_score
    elif score > max_score:
        score = max_score

    normalized_score = ((score - min_score) / (max_score - min_score)) * 10
    return normalized_score

def determine_response_category(similarity_score):
    if similarity_score is None:
        return "none"  
    threshold_positive = 0.55
    threshold_negative = 0.25
    if similarity_score >= threshold_positive:
        return "positive"
    elif similarity_score >= threshold_negative:
        return "neutral"
    else:
        return "negative"
    
def calculate_cosine_similarity(user_answer, sample_answer):
    if len(user_answer) < 10:
        return None  # Insufficient data for similarity calculation

# Use TF-IDF vectorization to convert answers into numerical representations
    vectorizer = TfidfVectorizer()
    all_answers = [user_answer, sample_answer]
    vectorizer.fit_transform(all_answers)
    user_answer_vector = vectorizer.transform([user_answer])
    sample_answer_vector = vectorizer.transform([sample_answer])

# Calculate cosine similarity between subsequent answers
    similarity_scores = cosine_similarity(user_answer_vector, sample_answer_vector)
    return similarity_scores

@app.route('/technical_start', methods=['GET', 'POST'])
def technical_start():
    if request.method == 'GET':
        # Initialize session variables to store questions and current question index
        random_index = random.randint(1, 30)
        random_question = technical_data.iloc[random_index, 0]
        random_sample_answer = technical_data.iloc[random_index, 1]
        session['current_question']=random_question
        session['current_sample_answer']=random_sample_answer
        # Render the template with the first question
        return render_template('technical_start.html', question=random_question)

    elif request.method == 'POST':
        # Get user answer from the form
        user_answer = request.form.get('user_answer')
        # Get the current question and increment the index

        # Call your get_response function
        clarity_score = assess_clarity(user_answer)
        readability_score = assess_readability(user_answer)
        grammar_errors = assess_grammar(user_answer)

        normalized_clarity = normalize_score(clarity_score)
        normalized_readability = normalize_score(readability_score)
        similarity_scores = calculate_cosine_similarity(user_answer,session['current_sample_answer'])
        response_category = determine_response_category(similarity_scores)

        if response_category == "positive":
            final_feedback = str(random.choice(positive_responses))+str(random.choice(positive_improvements))
        elif response_category == "neutral":
            final_feedback = str(random.choice(neutral_responses))+str(random.choice(neutral_improvements))
        elif response_category=="negative":                     
            final_feedback = str(random.choice(negative_responses))+str(random.choice(negative_improvements))
        else:
            final_feedback = "Try answering the question in sentences."
        current_question=session['current_question']
        # If all questions are answered, display a summary
        return render_template('summary.html', question=current_question, answer=user_answer,feedback=final_feedback,clarity=normalized_clarity,readability=normalized_readability,grammar=grammar_errors)

if __name__ == '__main__':
    app.run(debug=True)
