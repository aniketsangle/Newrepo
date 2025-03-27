
To extract resume information more accurately using regular expressions (RegEx) in Python, you can improve the parsing logic for extracting key details like names, emails, phone numbers, skills, and experience. Below is a structured approach with optimized RegEx patterns.


---

Step 1: Install Required Libraries

pip install pymupdf spacy
python -m spacy download en_core_web_sm


---

Step 2: Extract Text from PDF Resumes

We will use PyMuPDF (fitz) to read text from resumes.

import fitz  # PyMuPDF for PDF text extraction

def extract_text_from_pdf(pdf_path):
    """Extracts raw text from a PDF resume."""
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text")
    return text

# Example Usage
resume_text = extract_text_from_pdf("sample_resume.pdf")
print(resume_text)


---

Step 3: Define Optimized Regular Expressions

We will define patterns to extract key details.

import re

def extract_email(text):
    """Extracts email using a regex pattern."""
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    matches = re.findall(email_pattern, text)
    return matches[0] if matches else None

def extract_phone_number(text):
    """Extracts phone number using regex."""
    phone_pattern = r"(\+?\d{1,3}[-.\s]?)??\d{3}?[-.\s]?\d{3}[-.\s]?\d{4}"
    matches = re.findall(phone_pattern, text)
    return matches[0] if matches else None

def extract_experience(text):
    """Extracts years of experience."""
    experience_pattern = r"(\d+)\s*(?:years|yrs|year|Yrs|yr) of experience"
    matches = re.findall(experience_pattern, text, re.IGNORECASE)
    return matches[0] if matches else None

def extract_skills(text):
    """Extracts skills based on predefined keywords."""
    skill_keywords = ["Python", "Java", "C++", "Machine Learning", "SQL", "Django", "React", "AWS", "Data Science", "TensorFlow"]
    found_skills = [skill for skill in skill_keywords if skill.lower() in text.lower()]
    return found_skills


---

Step 4: Extract Name Using NLP

For extracting names, RegEx is unreliable. Instead, we use spaCy.

import spacy

nlp = spacy.load("en_core_web_sm")

def extract_name(text):
    """Extracts the candidate's name using NLP."""
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None


---

Step 5: Full Resume Parsing Function

def extract_resume_data(pdf_path):
    """Extracts structured information from a resume PDF."""
    text = extract_text_from_pdf(pdf_path)
    return {
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone_number(text),
        "experience": extract_experience(text),
        "skills": extract_skills(text)
    }

# Example Usage
resume_data = extract_resume_data("sample_resume.pdf")
print(resume_data)


---

Example Output

For a resume with this text:

John Doe
Software Engineer
Email: johndoe@example.com
Phone: +1 234-567-8901
Skills: Python, Machine Learning, SQL
Experience: 5 years of experience in software development

The output will be:

{
    "name": "John Doe",
    "email": "johndoe@example.com",
    "phone": "+1 234-567-8901",
    "experience": "5",
    "skills": ["Python", "Machine Learning", "SQL"]
}


---

Future Enhancements

Use Named Entity Recognition (NER) for better name extraction.

Use a pre-trained model like BERT to extract job-related skills more accurately.

Store extracted data in MySQL or a Data Warehouse for HR chatbot integration.


This method ensures better accuracy using regular expressions and NLP. Let me know if you need improvements!


-----

Creating a chatbot for your HR database using Large Language Models (LLMs) involves multiple steps, including setting up the LLM, connecting it to your database, and ensuring it can handle HR-related queries effectively. Below is a step-by-step explanation and code.


---

Step 1: Understanding the Workflow

1. User Query: The HR personnel asks a question about applicants.


2. Chatbot Processing: The chatbot interprets the query and determines the necessary data.


3. Database Querying: The chatbot fetches relevant data from the HR database.


4. Response Generation: The chatbot formats and presents the response.




---

Step 2: Setting Up the Environment

You'll need:

Python 3.8+

LLM Model (OpenAI GPT, Llama, or a self-hosted model)

FastAPI (for chatbot API)

MySQL Database (with applicant data)


Install dependencies:

pip install openai fastapi uvicorn mysql-connector-python sqlalchemy langchain


---

Step 3: Connecting to the HR Database

We use SQLAlchemy for better query handling.

from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base

# Define Database Connection
DATABASE_URL = "mysql+mysqlconnector://username:password@localhost/hr_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# Define Applicant Model
class Applicant(Base):
    __tablename__ = "applicants"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255))
    skills = Column(String(500))
    experience = Column(Integer)
    position_applied = Column(String(255))

Base.metadata.create_all(engine)

# Function to fetch applicant data
def get_applicant(name):
    session = SessionLocal()
    applicant = session.query(Applicant).filter(Applicant.name.ilike(f"%{name}%")).first()
    session.close()
    return applicant


---

Step 4: Setting Up the Chatbot with LLM

Using OpenAI GPT as the default LLM.

import openai

# OpenAI API Key (Replace with your key)
OPENAI_API_KEY = "your-api-key"

def generate_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are an HR assistant."},
                  {"role": "user", "content": prompt}],
        api_key=OPENAI_API_KEY
    )
    return response["choices"][0]["message"]["content"]


---

Step 5: Creating the Chatbot API with FastAPI

from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/chat")
def chat(query: str):
    # Extract applicant name from the query (basic implementation)
    words = query.split()
    for word in words:
        applicant = get_applicant(word)
        if applicant:
            response = f"Applicant {applicant.name} applied for {applicant.position_applied} with {applicant.experience} years of experience and skills: {applicant.skills}."
            return {"response": response}

    # If no specific applicant is found, use LLM
    llm_response = generate_response(query)
    return {"response": llm_response}

# Run server using:
# uvicorn chatbot:app --reload


---

Step 6: Testing the Chatbot

You can test the API using:

curl "http://127.0.0.1:8000/chat?query=Tell me about John Doe"

OR open http://127.0.0.1:8000/chat in the browser and pass the query.


---

Future Enhancements

Better NLP Query Parsing (Use Named Entity Recognition for extracting applicants)

Vector Database for Resume Search (Use FAISS for similarity matching)

Fine-tuning LLM for HR Queries (Train on HR-specific prompts)


This chatbot can provide applicant details, find matching resumes, and answer HR-related questions automatically. Let me know if you need help with improvements!


-------
To build your HR chatbot from scratch, we’ll follow these steps:

1. Scrape Data from PDF Resumes


2. Store Extracted Data in a Data Warehouse


3. Create a Chatbot to Query the Data




---

Step 1: Extract Data from PDF Resumes

We will use PyMuPDF (fitz) and spaCy for text extraction and NLP.

Install Required Libraries

pip install pymupdf spacy mysql-connector-python sqlalchemy fastapi uvicorn openai
python -m spacy download en_core_web_sm

Resume Parsing Script

import fitz  # PyMuPDF for PDF reading
import spacy
import os

nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text

def extract_information(text):
    """Extract candidate information using NLP."""
    doc = nlp(text)
    extracted_data = {
        "name": None,
        "email": None,
        "phone": None,
        "skills": [],
        "experience": None
    }

    # Extract email
    for token in doc:
        if "@" in token.text:
            extracted_data["email"] = token.text
            break

    # Extract phone number (basic pattern matching)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            extracted_data["name"] = ent.text
        elif ent.label_ == "DATE":
            extracted_data["experience"] = ent.text  # This needs better parsing
        elif ent.label_ == "ORG":
            extracted_data["skills"].append(ent.text)  # Assuming organizations may be skills

    return extracted_data

def process_resume(pdf_path):
    """Extract structured data from a resume PDF."""
    text = extract_text_from_pdf(pdf_path)
    return extract_information(text)

# Test the function
if __name__ == "__main__":
    resume_data = process_resume("sample_resume.pdf")
    print(resume_data)


---

Step 2: Store Extracted Data in a MySQL Data Warehouse

We’ll use SQLAlchemy to insert extracted data into a MySQL database.

Define Database and Table

from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "mysql+mysqlconnector://username:password@localhost/hr_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Applicant(Base):
    __tablename__ = "applicants"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255))
    email = Column(String(255), unique=True)
    phone = Column(String(20))
    skills = Column(Text)
    experience = Column(String(255))

Base.metadata.create_all(engine)

def store_applicant(data):
    """Store parsed resume data in the database."""
    session = SessionLocal()
    applicant = Applicant(
        name=data["name"],
        email=data["email"],
        phone=data["phone"],
        skills=",".join(data["skills"]),
        experience=data["experience"]
    )
    session.add(applicant)
    session.commit()
    session.close()

Process and Store Data

def process_and_store_resume(pdf_path):
    """Extracts data from a resume and stores it in MySQL."""
    extracted_data = process_resume(pdf_path)
    store_applicant(extracted_data)

# Example usage
process_and_store_resume("sample_resume.pdf")


---

Step 3: Create the HR Chatbot

Now that the resumes are stored, we build a chatbot to query the data.

Set Up OpenAI API

import openai

OPENAI_API_KEY = "your-api-key"

def generate_response(prompt):
    """Use GPT to generate HR-related responses."""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are an HR assistant."},
                  {"role": "user", "content": prompt}],
        api_key=OPENAI_API_KEY
    )
    return response["choices"][0]["message"]["content"]


---

Step 4: Build a Chatbot API with FastAPI

from fastapi import FastAPI, Query
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from models import Applicant  # Import from Step 2

app = FastAPI()

@app.get("/chat")
def chat(query: str):
    """Processes HR queries and fetches applicant data."""
    session = SessionLocal()
    
    # Check if query mentions a candidate
    words = query.split()
    for word in words:
        applicant = session.query(Applicant).filter(Applicant.name.ilike(f"%{word}%")).first()
        if applicant:
            session.close()
            return {
                "response": f"{applicant.name} applied for a position. Email: {applicant.email}, Skills: {applicant.skills}, Experience: {applicant.experience} years."
            }

    session.close()

    # If no candidate found, use LLM
    llm_response = generate_response(query)
    return {"response": llm_response}

# Run with:
# uvicorn chatbot:app --reload


---

Step 5: Test the Chatbot

curl "http://127.0.0.1:8000/chat?query=Tell me about John Doe"

or open http://127.0.0.1:8000/chat?query=Tell+me+about+John+Doe in a browser.


---

Future Enhancements

1. Improve Resume Parsing

Use spaCy NER to extract better entities (Skills, Experience).

Integrate a pre-trained model for better skill extraction.



2. Better Query Handling

Use vector search (FAISS) for resume similarity matching.



3. Advanced LLM Integration

Fine-tune GPT for HR-specific queries.




This chatbot will extract applicant data from resumes, store it in a database, and provide HR responses. Let me know if you need modifications!


