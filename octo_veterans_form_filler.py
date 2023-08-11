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






# Display the page title and the text box for the user to ask the question
st.title('OCTO VETERANS ELIGIBILITY and ENROLLMENT APPLICATION FOR HEALTH BENEFITS - AUTOMATIC FILL')



st.header('(STEP1)')


def save_uploadedfile(uploadedfile):
     with open(os.path.join("/home/ubuntu/pwc/",uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
     return st.success("Saved File:{} to tempDir".format(uploadedfile.name))


docx_file = st.file_uploader("LOAD the VETERANS ENROLLMENT APPLICATION FOR HEALTH BENEFITS",type=['txt','docx','pdf'])
option = st.selectbox(
'Select Start Year of your Service ',
('Tumor', 'Broken Leg', 'Chest Pains', 'Thyroid'))
st.write('You selected:', option)




trea_option = st.selectbox(
'What is the treatment for this condition?',
('MRI', 'Xray', 'Blood Sample', 'EKG'))
st.write('You selected:', trea_option)


# Different ways to use the API

if st.button("AUTOMATICALLY FILL IN VETERANS ENROLLMENT APPLICATION FOR HEALTH BENEFITS FORM"):
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




from PyPDF2 import PdfReader
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
st.header('DETERMINE candidates VETERANS healtchare ELIGIBILITY then automatically fill in VETERANS ENROLLMENT APPLICATION FOR HEALTH BENEFITS FORM ')

# Display the current response. No chat history is maintained
# Load PDF document
# Langchain has many document loaders
# We will use their PDF load to load the PDF document below
loaders = PyPDFLoader('/home/ubuntu/pwc/PDF/EligibilityForVAHealthCare.pdf')

d = st.date_input("Enter your service start date")
d = st.date_input("Enter your service end date")

prompt = st.text_input("What would like to know? ")
index = VectorstoreIndexCreator().from_loaders([loaders])

if prompt:
    # Get the resonse from LLM
    # We pass the model name (3.5) and the temperature (Closer to 1 means creative resonse)
    # stuff chain type sends all the relevant text chunks from the document to LLM

    response = index.query(llm=OpenAI(model_name="gpt-3.5-turbo", temperature=0.2), question = prompt, chain_type = 'stuff')

    # Write the results from the LLM to the UI
    st.write("<b>" + prompt + "</b><br><i>" + response + "</i><hr>", unsafe_allow_html=True )

if prompt:
    # Get the resonse from LLM
    # We pass the model name (3.5) and the temperature (Closer to 1 means creative resonse)
    # stuff chain type sends all the relevant text chunks from the document to LLM

    response = index.query(llm=OpenAI(model_name="gpt-3.5-turbo", temperature=0.2), question = prompt, chain_type = 'stuff')

    # Write the results from the LLM to the UI
    # Display the response in a message format
    # User input will appear on the right which is controlled by the property is_user=True
    message(prompt, is_user=True)
    message(response,is_user=False )









st.header('DETERMINE candidates VETERANS healtchare ELIGIBILITY then automatically fill inVETERANS ENROLLMENT APPLICATION FOR HEALTH BENEFITS FORM ')
prompt = st.text_input("Is this person  ELIGIBILle or VETERANS healtchare  ? ")

# Display the current response. No chat history is maintained
# Load PDF document
# Langchain has many document loaders
# We will use their PDF load to load the PDF document below
loaders = PyPDFLoader('/home/ubuntu/pwc/PDF/EligibilityForVAHealthCare.pdf')


index = VectorstoreIndexCreator().from_loaders([loaders2])

if prompt:
    # Get the resonse from LLM
    # We pass the model name (3.5) and the temperature (Closer to 1 means creative resonse)
    # stuff chain type sends all the relevant text chunks from the document to LLM

    response = index2.query(llm=OpenAI(model_name="gpt-3.5-turbo", temperature=0.2), question = prompt, chain_type = 'stuff')

    # Write the results from the LLM to the UI
    st.write("<b>" + prompt + "</b><br><i>" + response + "</i><hr>", unsafe_allow_html=True )

if prompt:
    # Get the resonse from LLM
    # We pass the model name (3.5) and the temperature (Closer to 1 means creative resonse)
    # stuff chain type sends all the relevant text chunks from the document to LLM

    response = index.query(llm=OpenAI(model_name="gpt-3.5-turbo", temperature=0.2), question = prompt, chain_type = 'stuff')

    # Write the results from the LLM to the UI
    # Display the response in a message format
    # User input will appear on the right which is controlled by the property is_user=True
    message(prompt, is_user=True)
    message(response,is_user=False )


# Load PDF document
# Langchain has many document loaders
# We will use their PDF load to load the PDF document below
loaders3 = PyPDFLoader('/home/ubuntu/pwc/PDF/molina_clinical_guidelines_for_conditions_treatments.pdf')

# Create a vector representation of this document loaded
index3 = VectorstoreIndexCreator().from_loaders([loaders3])

# Setup streamlit app


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


