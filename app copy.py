from langchain_groq import ChatGroq
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
import streamlit as st
import pandas as pd


# Step 1: Load the optimized LLaMA model on Groq

llama = ChatGroq(
    model="LLaMA3-70B-8192",
    groq_api_key='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',
    temperature=0.0
)


# Step 2: Define the AI Tutor’s functionality
class AITutor:
    def __init__(self):
        self.model = llama
        self.prompt_template = """
        You are an AI tutor specialized in history. Given the following topic: {topic}, 
        you have to provide a detailed explanation of the topic over 250 words with key historical facts, and an interactive timeline to understand the history.
        Also, generate a quiz with 10 multiple-choice questions for student assessment.
        ### NO PREAMBLE And ONLY FOCUS ON CONTENT.
        """


    def generate_fact_and_quiz(self, topic: str):
        prompt = self.prompt_template.format(topic=topic)
        
        # Step 3: Run the inference using Groq
        self.model.invoke(prompt)

        # Step 4: Define a chain to process history facts and generate quizzes
def create_history_facts():
    prompt_facts = PromptTemplate(template="Provide historical facts with a 'DID YOU KNOW' format about {topic}.")
    prompt_explain = PromptTemplate(template="Provide an explanation of 512 words on {topic}.")
    # Create individual chains
    fact_chain = LLMChain(llm=llama, prompt=prompt_facts)
    explain_chain = LLMChain(llm=llama, prompt=prompt_explain)
    # Sequential chain
    history_fakt = SimpleSequentialChain(chains=[explain_chain, fact_chain])
    return history_fakt

def create_history_explanation():
    prompt_facts = PromptTemplate(template="Provide historical facts about {topic}.")
    prompt_explain = PromptTemplate(template="Provide an explanation of about 500 words covering the historical places, events and dates about the given {topic} for better understanding of history.")
    # Create individual chains
    fact_chain = LLMChain(llm=llama, prompt=prompt_facts)
    explain_chain = LLMChain(llm=llama, prompt=prompt_explain)
    # Sequential chain
    history_explaination = SimpleSequentialChain(chains=[fact_chain, explain_chain])
    return history_explaination

def create_history_quizes():
    prompt_facts = PromptTemplate(template="Provide historical and must remembering facts  about {topic}.")
    prompt_quiz = PromptTemplate(template="Create a quiz with 50 questions on {topic}.")
    prompt_explain = PromptTemplate(template="Provide an explanation of 1000 words on {topic}.")
    # Create individual chains
    fact_chain = LLMChain(llm=llama, prompt=prompt_facts)
    quiz_chain = LLMChain(llm=llama, prompt=prompt_quiz)
    explain_chain = LLMChain(llm=llama, prompt=prompt_explain)
    # Sequential chain
    history_quizes = SimpleSequentialChain(chains=[explain_chain, fact_chain, quiz_chain])
    return history_quizes

def create_timeline():
    dates= PromptTemplate(template="Provide dates of events on or about {topic}.")
    places= PromptTemplate(template="Provide places where the events on or about {topic}. Also, mention the dates with the places or events.")
    dates_chain = LLMChain(llm=llama, prompt=dates)
    places_chain = LLMChain(llm=llama, prompt=places)
    timeline = SimpleSequentialChain(chains=[dates_chain, places_chain])
    return timeline

def create_story():
    imagine= PromptTemplate(template="Provide ideas to write and understand historical stories relevent to the {topic}.")
    places= PromptTemplate(template="Provide places where the events on or about {topic}.")
    imagine1= PromptTemplate(template="Based on the given places provided, imagine real world scenarios about the {topic}.")
    imagine2= PromptTemplate(template="Based on the given places and scenarios provided, create real world Characters who were actually in the history of the given {topic}.")
    story= PromptTemplate(template="Make story with the information gained by Places, scenarios and Characters while studying the {topic}.")
    
    imagine_chain = LLMChain(llm=llama, prompt=imagine)
    imagine2_chain = LLMChain(llm=llama, prompt=imagine2)
    magine1_chain = LLMChain(llm=llama, prompt=imagine1)
    places_chain = LLMChain(llm=llama, prompt=places)
    story_chain = LLMChain(llm=llama, prompt=story)

    storyline = SimpleSequentialChain(chains=[places_chain, imagine_chain, magine1_chain, imagine2_chain, story_chain])
    return storyline

# Step 5: Engaging AI tutor workflow
def ai_tutor_workflow(topic: str):
    # Initialize AI Tutor and LangChain history chain
    ai_tutor = AITutor()
    history_fact_chain = create_history_facts()
    history_explaination_chain = create_history_explanation()
    history_quiz_chain = create_history_quizes()
    timelines = create_timeline()
    stories = create_story()

    # Invoking the model for every chain
    history_factss = history_fact_chain.invoke(topic)
    history_timeline = timelines.invoke(topic)
    history_explaination = history_explaination_chain.invoke(topic)
    history_quiz = history_quiz_chain.invoke(topic)
    history_story = stories.invoke(topic)

    # print("=== Explanation ===")
    # print(history_explaination)
    
    # print("=== Historical Facts ===")
    # print(history_factss)


    # print("\n=== Quiz ===")
    # print(history_quiz)

    # print("\n=== Timeline ===")
    # print(history_timeline)

    # print("\n=== Storyline ===")
    # print(history_story)

    return  history_quiz, history_factss, history_explaination, history_timeline, history_story


# Title of the app
st.title("AI History Tutor by PARTH")

# Introduction text
st.write("""
    This AI-powered tutor, powered by LLaMA 3.1 70B model, can generate interactive Explanations, Historical Facts, Quizzes, Stories and Timelines.
    Just enter a historical topic, and it will generate information for you.
""")

# Sidebar for additional information
st.sidebar.title("Happy Learning")
st.sidebar.write("Ask anything from your history book to interact with the AI tutor. \n My AI tutor will give you 50 Quizes for you to get ready for the exam.")
df = pd.DataFrame({ "List Information":['Quiz', 'Facts', 'Explaination', 'Timeline', 'Story']})
st.sidebar.write(df)
# Get user input
topic = st.text_input("Enter a historical topic or event:")

# Load Groq compiled LLaMA model (replace with your actual model path)
@st.cache_resource

# Generate a response from the model
def generate_response(prompt):
    # Tokenize the input prompt
    prompt_template = """
        You are an AI tutor specialized in history. Given the following topic: {topic}, 
        you have to provide a detailed explanation of the topic over 250 words with key historical facts, historical places, historical events, and an interactive timeline with dates to understand the history.
        Also, generate a quiz with 10 multiple-choice questions for student assessment.
        ### NO PREAMBLE And ONLY FOCUS ON CONTENT.
        """
    prompt = prompt_template.format(topic=topic)

    # Decode the output tokens into human-readable text
    response = ai_tutor_workflow(topic)
    
    return response
   

    # Run the model and get the output

# When user provides input
if topic:
    st.write(f"Generating a response for: **{topic}**")
    
    # Generate the response using the model
    with st.spinner('Generating response...'):
        result = generate_response(topic)
        result = list(result)
    # Display the result
    st.write(result)