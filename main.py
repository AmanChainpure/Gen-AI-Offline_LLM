from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import fitz

# Define Flan T5 model and tokenizer
# To save model
model_name = "sjrhuschlee/flan-t5-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

model.save_pretrained("models")
tokenizer.save_pretrained("models")

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf_document:
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
    return text

pdf_path = "data/Delivery assistance.pdf"
text = extract_text_from_pdf(pdf_path)


# Define question-answering pipeline
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

# Ask a question
question = input("Enter your question: ")

# Perform question answering
qa_input = {'question': question, 'context': text}
result = nlp(qa_input)

# Print the answer
print("Answer:", result['answer'])
