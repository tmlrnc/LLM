import streamlit as st
import pandas as pd
import numpy as np
# Imports
import os 
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from streamlit_chat import message
import openai

from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator

filled_in = 0 
openai.api_key = "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"

API_KEY = openai.api_key
model_id = "gpt-3.5-turbo"

# Add your openai api key for use
os.environ["OPENAI_API_KEY"] = API_KEY

if 'form_filled' not in st.session_state:
    st.session_state['form_filled'] = 0
from PyPDF2 import PdfReader

import os
import openai

import os

openai.api_key = "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"
def gpt_req_res(subject_text='prior authorization',
                prompt_base='answer like an experienced consultant for prior authorization ',
                model='text-davinci-003',
                max_tokens=1200,
                temperature=0.8):

    response = openai.Completion.create(
        model=model,
        prompt=prompt_base + ': ' + subject_text,
        temperature=temperature,
        max_tokens=1200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response.choices[0].text


# This function is reading PDF from the start page to final page
# given as input (if less pages exist, then it reads till this last page)
def get_pdf_text(document_path, start_page=1, final_page=999):
    reader = PdfReader(document_path)
    number_of_pages = len(reader.pages)
    page = ""
    for page_num in range(start_page - 1, min(number_of_pages, final_page)):
        page += reader.pages[page_num].extract_text()
    return page




# Load PDF document
# Langchain has many document loaders 
# We will use their PDF load to load the PDF document below

#cfg = get_cfg()
#cfg.MODEL.DEVICE = 'cpu'

# insert your OPENAI API key under here

pdf_folder_name = '/home/ubuntu/pwc/PDF'

def create_pdf_loaders(pdf_folder_name):
    loaders = []
    for i in os.listdir(pdf_folder_name):
        file_path = os.path.join(pdf_folder_name, i)
        loader = UnstructuredPDFLoader(file_path)
        loaders.append(loader)
    return loaders

#loaders = create_pdf_loaders(pdf_folder_name)
#print(loaders)

#doc_path_name = '/home/ubuntu/pwc/PDF/pa_form.pdf'
doc_path_name = '/home/ubuntu/pwc/PDF/pa_mri.pdf'

loaders = PyPDFLoader(doc_path_name)

# Create a vector representation of this document loaded
index = VectorstoreIndexCreator().from_loaders([loaders])


#index = VectorstoreIndexCreator().from_loaders(loaders)

import pandas as pd
import numpy as np
from tqdm import tqdm



# Define the maximum number of words to tokenize (DistilBERT can tokenize up to 512)
MAX_LENGTH = 128


# Define function to encode text data in batches
def batch_encode(tokenizer, texts, batch_size=256, max_length=MAX_LENGTH):
    """""""""
    A function that encodes a batch of texts and returns the texts'
    corresponding encodings and attention masks that are ready to be fed
    into a pre-trained transformer model.

    Input:
        - tokenizer:   Tokenizer object from the PreTrainedTokenizer Class
        - texts:       List of strings where each string represents a text
        - batch_size:  Integer controlling number of texts in a batch
        - max_length:  Integer controlling max number of words to tokenize in a given text
    Output:
        - input_ids:       sequence of texts encoded as a tf.Tensor object
        - attention_mask:  the texts' attention mask encoded as a tf.Tensor object
    """""""""

    input_ids = []
    attention_mask = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer.batch_encode_plus(batch,
                                             max_length=max_length,
                                             padding='longest', #implements dynamic padding
                                             truncation=True,
                                             return_attention_mask=True,
                                             return_token_type_ids=False
                                             )
        input_ids.extend(inputs['input_ids'])
        attention_mask.extend(inputs['attention_mask'])


    return tf.convert_to_tensor(input_ids), tf.convert_to_tensor(attention_mask)


X_train_ids, X_train_attention = batch_encode(tokenizer, X_train.tolist())

# Encode X_valid
X_valid_ids, X_valid_attention = batch_encode(tokenizer, X_valid.tolist())

# Encode X_test
X_test_ids, X_test_attention = batch_encode(tokenizer, X_test.tolist())


def augment_sentence(sentence, aug, num_threads):
    """""""""
    Constructs a new sentence via text augmentation.

    Input:
        - sentence:     A string of text
        - aug:          An augmentation object defined by the nlpaug library
        - num_threads:  Integer controlling the number of threads to use if
                        augmenting text via CPU
    Output:
        - A string of text that been augmented
    """""""""
    return aug.augment(sentence, num_thread=num_threads)

def augment_text(df, aug, num_threads, num_times):
    """""""""
    Takes a pandas DataFrame and augments its text data.

    Input:
        - df:            A pandas DataFrame containing the columns:
                                - 'comment_text' containing strings of text to augment.
                                - 'isToxic' binary target variable containing 0's and 1's.
        - aug:           Augmentation object defined by the nlpaug library.
        - num_threads:   Integer controlling number of threads to use if augmenting
                         text via CPU
        - num_times:     Integer representing the number of times to augment text.
    Output:
        - df:            Copy of the same pandas DataFrame with augmented data
                         appended to it and with rows randomly shuffled.
    """""""""

    # Get rows of data to augment
    to_augment = df[df['isToxic']==1]
    to_augmentX = to_augment['comment_text']
    to_augmentY = np.ones(len(to_augmentX.index) * num_times, dtype=np.int8)

    # Build up dictionary containing augmented data
    aug_dict = {'comment_text':[], 'isToxic':to_augmentY}
    for i in tqdm(range(num_times)):
        augX = [augment_sentence(x, aug, num_threads) for x in to_augmentX]
        aug_dict['comment_text'].extend(augX)

    # Build DataFrame containing augmented data
    aug_df = pd.DataFrame.from_dict(aug_dict)

    return df.append(aug_df, ignore_index=True).sample(frac=1, random_state=42)


import openai
import streamlit as st
from streamlit_chat import message
import os
openai.api_key = "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"
def generate_response(prompt):
    completion=openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.6,
    )
    message=completion.choices[0].text
    return message



def save_uploadedfile(uploadedfile):
     with open(os.path.join("/home/ubuntu/pwc/",uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
     return st.success("Saved File:{} to tempDir".format(uploadedfile.name))


st.header('Submit Your Prior Authorization Request')
docx_file = st.file_uploader("Upload File",type=['txt','docx','pdf'])
if st.button("READ"):
    if docx_file is not None:
        file_details = {"Filename":docx_file.name,"FileType":docx_file.type,"FileSize":docx_file.size}
        doc_path_name = '/home/ubuntu/pwc/PDF/pa_form.pdf'
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

# Different ways to use the API

if st.button("READ AND FILL IN"):
    if docx_file is not None:
        file_details = {"Filename":docx_file.name,"FileType":docx_file.type,"FileSize":docx_file.size}
        doc_path_name = '/home/ubuntu/pwc/PDF/pa_form.pdf'
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


def generate_response_email(user_input_email):
    import smtplib


    mailserver = smtplib.SMTP('smtp.gmail.com',587)
    mailserver.ehlo()
    mailserver.starttls()
    mailserver.ehlo()
    mailserver.login('anunahealth@gmail.com', 'Tfltfl646!')
    mailserver.sendmail('tmlrnc@anuna.com','tmlrnc@gmail.com','fuck')
    mailserver.quit()
    return



with open('/home/ubuntu/pwc/file_session_storage.csv') as f:
   st.download_button('Download Answer', f)  # Defaults to 'text/plain'






# Setup streamlit app

# Display the page title and the text box for the user to ask the question
#st.title('Ask A Questoin About your Prior Authorization document and next steps - NO BERT WORDVEC SUMMARY FEEDBACK')
#prompt = st.text_input("Ask A Questoin About your Prior Authorization document and next steps - NO BERT WORDVEC SUMMARY FEEDBACK")


# Display the current response. No chat history is maintained

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


#------------------------------------------------------------------
# Save history

# This is used to save chat history and display on the screen
if 'answer' not in st.session_state:
    st.session_state['answer'] = []

if 'question' not in st.session_state:
    st.session_state['question'] = []


#------------------------------------------------------------------
# Display the current response. Chat history is displayed below

if prompt:
 # Get the resonse from LLM
 # We pass the model name (3.5) and the temperature (Closer to 1 means creative resonse)
 # stuff chain type sends all the relevant text chunks from the document to LLM
response = index.query(llm=OpenAI(model_name = model_id, temperature=0.2), question = prompt, chain_type = 'stuff')

 # Add the question and the answer to display chat history in a list
 # Latest answer appears at the top
# st.session_state.question.insert(0,prompt )
# st.session_state.answer.insert(0,response )
 
 # Display the chat history
# for i in range(len( st.session_state.question)) :
#    message(st.session_state['question'][i], is_user=True)
#    message(st.session_state['answer'][i], is_user=False)


################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
st.title('Ask A Question About your Prior Authorization')
#storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
user_input=st.text_input("You:",key='input')
if user_input:
    output=generate_response("is my prior authorization form complete " +user_input)
    #store the output
    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(output)
    wt = str(output)
    myts = []
    res = wt.split()
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    nltk.download('stopwords')
    nltk.download('punkt')
    for i in res:
        myr = i + " "
        myts.append(myr)
    stop_words = set(stopwords.words('english'))
    stopwords = nltk.corpus.stopwords.words('english')
    newStopWords = ['and','the','are','is', 'that','can']


    stopwords.extend(newStopWords)
    #tokens = word_tokenize(myts)
    myt = [i for i in myts if not i in stopwords]
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(use_idf=True, max_df=0.5,min_df=1, ngram_range=(1,3))
    vectors = vectorizer.fit_transform(myt)

    dict_of_tokens={i[1]:i[0] for i in vectorizer.vocabulary_.items()}
    tfidf_vectors = []  # all deoc vectors by tfidf
    for row in vectors:
        tfidf_vectors.append({dict_of_tokens[column]:value for (column,value) in zip(row.indices,row.data)})


    doc_sorted_tfidfs =[]  # list of doc features each with tfidf weight
    for dn in tfidf_vectors:
        newD = sorted(dn.items(), key=lambda x: x[1], reverse=True)
        newD = dict(newD)
        doc_sorted_tfidfs.append(newD)
    print(doc_sorted_tfidfs)
    top_words_topics = str(doc_sorted_tfidfs[0]) + str(doc_sorted_tfidfs[1]) + str(doc_sorted_tfidfs[2]) + str(doc_sorted_tfidfs[3])
    print('top_words_topics')
    print(top_words_topics)



    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=100)
    topic_model = BERTopic(umap_model=umap_model, language="english", calculate_probabilities=True)
    topics, probabilities = topic_model.fit_transform(myt)
    print(topic_model.get_topic(0))


    from transformers import DistilBertTokenizerFast
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    from transformers import TFDistilBertModel, DistilBertConfig

    DISTILBERT_DROPOUT = 0.2
    DISTILBERT_ATT_DROPOUT = 0.2

    config = DistilBertConfig(dropout=DISTILBERT_DROPOUT, attention_dropout=DISTILBERT_ATT_DROPOUT, output_hidden_states=True)

    distilBERT = TFDistilBertModel.from_pretrained('distilbert-base-uncased', config=config)

    with open("file_session_storage.csv", "a") as f:
        print(wt, file=f)

    with open("file_session_storage.csv", "a") as f:
        print(top_words_topics, file=f)



if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')




with open("/home/ubuntu/pwc/file_session_storage.txt") as f:
    state_of_the_union = f.read()
    

from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain

llm = OpenAI(temperature=0)
summary_chain = load_summarize_chain(llm, chain_type="map_reduce")

from langchain.chains import AnalyzeDocumentChain


summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain)

summarize_document_chain.run(state_of_the_union)

from langchain.chains.question_answering import load_qa_chain

qa_chain = load_qa_chain(llm, chain_type="map_reduce")


qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)


qa_document_chain.run(input_document=state_of_the_union, question="is my prior authorization ready for approval?")

