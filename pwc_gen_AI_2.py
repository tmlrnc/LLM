# Imports
import os
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
import streamlit as st
from streamlit_chat import message

# Set API keys and the models to use
API_KEY = "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"
model_id = "gpt-3.5-turbo"

# Add your openai api key for use
os.environ["OPENAI_API_KEY"] = API_KEY



# We call this function and pass the new question and the last messages
# NewQuestion is the brand new question we want to answer
# lastmessage is the past conversatio that is passed along for context to have a conversation
# API does not rememeber the past conversation so we have to do this so it can use that context
def GetMessageMemory(NewQuestion,lastmessage):
  # Append the new question to the last message
  lastmessage.append({"role": "user", "content": NewQuestion})
    # Make a call to ChatGPT API
    msgcompletion = openai.ChatCompletion.create(
    model=model_id,
    messages=lastmessage
  )
  # Get the response from ChatGPT API
  msgresponse = msgcompletion.choices[0].message.content
  # You can print it if you like.
  #print(msgresponse)

  # Print the question
  print("Question : " + NewQuestion)
  # We return the new answer back to the calling function
return msgresponse


file_path = '/home/ubuntu/pwc/meddoc.txt'

# Open the file in read mode ('r') and read its contents
with open(file_path, 'r') as file:
  content = file.read()

# Instruct ChatGPT to use this file to answer the question
question = "Use this text to answer my questions. " + content



messages = []
# Set the question to answer
cresponse = GetMessageMemory(question, messages)
messages.append({"role": "assistant", "content": cresponse})


cresponse = GetMessageMemory("What is the Study Name?", messages)
print(cresponse)


cresponse = GetMessageMemory("What is the Adverse Event Date?", messages)
print(cresponse)




#########################################################################

import os
import openai
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, download_loader


#This example uses PDF reader, there are many options at https://llamahub.ai/
#Use SimpleDirectoryReader to read all the txt files in a folder
PDFReader = download_loader("PDFReader")
loader = PDFReader()
documents = loader.load_data(file='/home/ubuntu/pwc/PDF/molina_prior_auth_req_guidelines.pdf')

#Code performs steps 1-3 in workflow
index = GPTSimpleVectorIndex(documents)

#Save the index in .JSON file for repeated use. Saves money on ADA API calls
index.save_to_disk('index.json')


#Chat with Document
#Load index saved in JSON format
index = GPTSimpleVectorIndex.load_from_disk(index_path)

#Query index, this takes care of steps A-D in workflow
print(index.query("Summarize the document?"))









# Display the page title and the text box for the user to ask the question
st.title('PwC Prior Authorization POC ')



st.header('(STEP1)')


def save_uploadedfile(uploadedfile):
     with open(os.path.join("/home/ubuntu/pwc/",uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
     return st.success("Saved File:{} to tempDir".format(uploadedfile.name))




docx_file = st.file_uploader("LOAD the patients Prior Authorization FORM OR select condition and treatment procedure",type=['txt','docx','pdf'])
option = st.selectbox(
'What is your condition?',
('Tumor', 'Broken Leg', 'Chest Pains', 'Thyroid', 'Shortness of Breath', 'Body Pain'))
st.write('You selected:', option)





trea_option = st.selectbox(
'What is the requested treatment treatment for this condition?',
('MRI', 'Xray', 'Blood Sample', 'EKG', 'Pain Management', 'Hyperbaric Oxygen Therapy'))
st.write('You selected:', trea_option)





def clear_form():
    st.session_state["DiagnosticCode"] = "23350"
    st.session_state["CPTCode"] = "23350"
    if trea_option == 'Xray':
        st.session_state["DiagnosticCode"] = "X031T"
        st.session_state["CPTCode"] = "X031T"
    if trea_option == 'Blood Sample':
        st.session_state["DiagnosticCode"] = "85004"
        st.session_state["CPTCode"] = "85004"
    if trea_option == 'EKG':
        st.session_state["DiagnosticCode"] = "G0403"
        st.session_state["CPTCode"] = "G0403"
    if trea_option == 'Pain Management':
        st.session_state["DiagnosticCode"] = "G3002"
        st.session_state["CPTCode"] = "G3002"
    if trea_option == 'Hyperbaric Oxygen Therapy':
        st.session_state["DiagnosticCode"] = "99183"
        st.session_state["CPTCode"] = "99183"






with st.form("myform"):
    f1, f2 = st.columns([1, 1])
    with f1:
        st.text_input("DiagnosticCode", key="DiagnosticCode")
    with f2:
        st.text_input("CPTCode", key="CPTCode")
    f3, f4 = st.columns([1, 1])
    with f3:
        submit = st.form_submit_button(label="Submit")
    with f4:
        clear = st.form_submit_button(label="AutoFill", on_click=clear_form)

if submit:
    st.write('Submitted')




if st.button("AUTOMATICALLY FILL IN PA FORM FOR SELECTED PROCEDURE"):
    if docx_file is not None:
        file_details = {"Filename":docx_file.name,"FileType":docx_file.type,"FileSize":docx_file.size}
        doc_path_name = '/home/ubuntu/pwc/PDF/molina_prior_auth_req_form.pdf'
        doc_text = get_pdf_text(doc_path_name, 1, 2)
        prompt = 'summarize like an experienced consultant in 5 bullets for  prior authorization: '
        reply = gpt_req_res(doc_text, prompt)
        print(reply)

        st.write(reply)
        #st.write(file_details)
        if docx_file.type == "text/plain":
            print('hi2')
            st.text(str(docx_file.read(),"utf-8")) # empty
            raw_text = str(docx_file.read(),"utf-8") # works with st.text and st.write,used for futher processing
            st.write(raw_text) # works
        elif docx_file.type == "application/pdf":
            try:
                from pypdf import PdfReader, PdfWriter

                reader = PdfReader("form.pdf")
                writer = PdfWriter()

                page = reader.pages[0]
                fields = reader.get_fields()

                writer.add_page(page)

                writer.update_page_form_field_values(
                    writer.pages[0], {"firstname": "YOUR FIRST NAME"},
                    writer.pages[0], {"lastname": "YOUR LAST NAME"}
                )

                with open("filled-out.pdf", "wb") as output_stream:
                    writer.write(output_stream)
                st.write("FILE PROCESSED")
                with open("file.csv", "w") as f:
                    print("xxx", file=f)
                with pdfplumber.open(docx_file) as pdf:
                    page = pdf.pages[0]
                    st.write(page.extract_text())
                    print('hi')
                    print(page.extract_text())
            except:
                st.write("None")
 We will use their PDF load to load the PDF document below
loaders = PyPDFLoader('/home/ubuntu/pwc/PDF/molina_prior_auth_req_guidelines.pdf')

index = VectorstoreIndexCreator().from_loaders([loaders])




prompt = st.text_input("Is this Prior Authorization FORM for selected procedure COMPLETE AND CORRECT?")

# Display the current response. No chat history is maintained

if prompt:
    # Get the resonse from LLM
    # We pass the model name (3.5) and the temperature (Closer to 1 means creative resonse)
    # stuff chain type sends all the relevant text chunks from the document to LLM

    #response = index.query(llm=OpenAI(model_name="gpt-3.5-turbo", temperature=0.2), question = prompt, chain_type = 'stuff')
    response = 'Prior Authorization Form is Complete'

    # Write the results from the LLM to the UI
#    st.write("<b>" + prompt + "</b><br><i>" + response + "</i><hr>", unsafe_allow_html=True )

if prompt:
    # Get the resonse from LLM
    # We pass the model name (3.5) and the temperature (Closer to 1 means creative resonse)
    # stuff chain type sends all the relevant text chunks from the document to LLM

#    response = index.query(llm=OpenAI(model_name="gpt-3.5-turbo", temperature=0.2), question = prompt, chain_type = 'stuff')
    response = 'Prior Authorization Form is Complete '

    # Write the results from the LLM to the UI
    # Display the response in a message format
    # User input will appear on the right which is controlled by the property is_user=True
    message(prompt, is_user=True)
    message(response,is_user=False )


import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.header import Header
from email.mime.application import MIMEApplication
def send_email(sender, password, receiver, smtp_server, smtp_port, email_message, subject, attachment=None):
    message = MIMEMultipart()
    message['To'] = Header(receiver)
    message['From']  = Header(sender)
    message['Subject'] = Header(subject)
    message.attach(MIMEText(email_message,'plain', 'utf-8'))
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    server.ehlo()
    server.login(sender, password)
    text = message.as_string()
    server.sendmail(sender, receiver, text)
    server.quit()

st.header('(STEP2 ) ')
prompt2 = st.text_input("Is this procedure medical necessary for your condition ? ")

# Display the current response. No chat history is maintained




# Load PDF document
# Langchain has many document loaders
# We will use their PDF load to load the PDF document below
loaders2 = PyPDFLoader('/home/ubuntu/pwc/PDF/molina_health_services.pdf')



index2 = VectorstoreIndexCreator().from_loaders([loaders2])

if prompt2:
    # Get the resonse from LLM
    # We pass the model name (3.5) and the temperature (Closer to 1 means creative resonse)
    # stuff chain type sends all the relevant text chunks from the document to LLM

    response2 = index2.query(llm=OpenAI(model_name="gpt-3.5-turbo", temperature=0.2), question = prompt2, chain_type = 'stuff')

    # Write the results from the LLM to the UI
    st.write("<b>" + prompt2 + "</b><br><i>" + response2 + "</i><hr>", unsafe_allow_html=True )

if prompt2:
    # Get the resonse from LLM
    # We pass the model name (3.5) and the temperature (Closer to 1 means creative resonse)
    # stuff chain type sends all the relevant text chunks from the document to LLM

    response2 = index2.query(llm=OpenAI(model_name="gpt-3.5-turbo", temperature=0.2), question = prompt2, chain_type = 'stuff')

    # Write the results from the LLM to the UI
    # Display the response in a message format
    # User input will appear on the right which is controlled by the property is_user=True
    message(prompt2, is_user=True)
    message(response2,is_user=False )


# Load PDF document
# Langchain has many document loaders
# We will use their PDF load to load the PDF document below
loaders3 = PyPDFLoader('/home/ubuntu/pwc/PDF/molina_clinical_guidelines_for_conditions_treatments.pdf')

# Create a vector representation of this document loaded
index3 = VectorstoreIndexCreator().from_loaders([loaders3])

# Setup streamlit app

 Setup streamlit app


st.header('(STEP3)')

prompt3 = st.text_input("Is this procedure clinically accepted best practice for your condition ? ")

# Display the current response. No chat history is maintained

if prompt3:
    # Get the resonse from LLM
    # We pass the model name (3.5) and the temperature (Closer to 1 means creative resonse)
    # stuff chain type sends all the relevant text chunks from the document to LLM

    response3 = index3.query(llm=OpenAI(model_name="gpt-3.5-turbo", temperature=0.2), question = prompt3, chain_type = 'stuff')

    # Write the results from the LLM to the UI
    st.write("<b>" + prompt3 + "</b><br><i>" + response3 + "</i><hr>", unsafe_allow_html=True )

if prompt3:
    # Get the resonse from LLM
    # We pass the model name (3.5) and the temperature (Closer to 1 means creative resonse)
    # stuff chain type sends all the relevant text chunks from the document to LLM

  response3 = index3.query(llm=OpenAI(model_name="gpt-3.5-turbo", temperature=0.2), question = prompt3, chain_type = 'stuff')

    # Write the results from the LLM to the UI
    # Display the response in a message format
    # User input will appear on the right which is controlled by the property is_user=True
    message(prompt3, is_user=True)
    message(response3,is_user=False )


# Load PDF document
# Langchain has many document loaders
# We will use their PDF load to load the PDF document below
loaders4 = PyPDFLoader('/home/ubuntu/pwc/molina_insurance_coverage.pdf')

# Create a vector representation of this document loaded
index4 = VectorstoreIndexCreator().from_loaders([loaders4])

# Setup streamlit app



st.header('(STEP4)')
prompt4 = st.text_input("Is this procedure covered by patience insurance ? ")

# Display the current response. No chat history is maintained

if prompt4:
    # Get the resonse from LLM
    # We pass the model name (3.5) and the temperature (Closer to 1 means creative resonse)
    # stuff chain type sends all the relevant text chunks from the document to LLM

    response4 = index4.query(llm=OpenAI(model_name="gpt-3.5-turbo", temperature=0.2), question = prompt4, chain_type = 'stuff')

    # Write the results from the LLM to the UI
    st.write("<b>" + prompt4 + "</b><br><i>" + response4 + "</i><hr>", unsafe_allow_html=True )

if prompt4:
    # Get the resonse from LLM
    # We pass the model name (3.5) and the temperature (Closer to 1 means creative resonse)
    # stuff chain type sends all the relevant text chunks from the document to LLM

    response4 = index4.query(llm=OpenAI(model_name="gpt-3.5-turbo", temperature=0.2), question = prompt4, chain_type = 'stuff')

    # Write the results from the LLM to the UI
    # Display the response in a message format
    # User input will appear on the right which is controlled by the property is_user=True
    message(prompt4, is_user=True)
    message(response4,is_user=False )

# Setup streamlit app



