import subprocess
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.app import App
from kivy.uix.image import Image as KivyImage
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image as KivyImage
from kivy.uix.gridlayout import GridLayout


import os
os.environ['OPENAI_API_KEY'] = "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"
from langchain.llms import OpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import csv
from sentence_transformers import SentenceTransformer
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from kivy.uix.scrollview import ScrollView
import argostranslate.package
import argostranslate.translate
from bs4 import BeautifulSoup
import urllib.request
from PIL import Image
def get_disease(file_path, row_number):
    try:
        with open(file_path, 'r') as csv_file:
            reader = csv.reader(csv_file)
            for i, row in enumerate(reader):
                if i == row_number:
                    if len(row) >= 3:
                        return row[2]
                    else:
                        return "Row does not have a second column."
            return "Row not found in the CSV file."
    except FileNotFoundError:
        return "File not found."
    except Exception as e:
        return f"An error occurred: {e}"
def retrieve_image (disease,images):
    """0 if no wikipedia page found
    1 if no image on wikipedia page
    """
    # disease = "Heart Palpitations"
    disease = '_'.join(disease.split())
    try: 
        fp = urllib.request.urlopen(f"https://en.wikipedia.org/wiki/{disease}")
    except urllib.error.HTTPError:
        if os.path.exists("cache.png"):
            os.remove("cache.png")
        return 0,images # couldn't find Wikipedia page
    
    mybytes = fp.read()
    mystr = mybytes.decode("utf8")
    fp.close()

    soup = BeautifulSoup(mystr)

    text = soup.find('table', class_="infobox")
    if text:
        text = text.find('img')['src']
    else: 
        if os.path.exists("cache.png"):
            os.remove("cache.png")
        return 1,images # no images on Wikipedia page
    
    urllib.request.urlretrieve("https://" + text[2:], "cache.png")

    img = Image.open("cache.png") 
    images+=[img]
    # img.show()
    return img,images

def ask_llama(prompt):
    cmd = [
        "llm", 
        "-m", "Llama-2-7b-chat",
        "-o", "temperature", "0.9",
        "-o", "top_p", "0.9",
        prompt
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout


def MASHQA_process(promptlol):
    # pip install argostranslate
    st_callback = StreamlitCallbackHandler(st.container())

    

    def translate_hindi_to_english (text): 
        from_code = "hi"
        to_code = "en"

        # Download and install Argos Translate package
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        package_to_install = next(
            filter(
                lambda x: x.from_code == from_code and x.to_code == to_code, available_packages
            )
        )
        argostranslate.package.install_from_path(package_to_install.download())
        
        # Translate
        translatedText = argostranslate.translate.translate(text, from_code, to_code)

        return translatedText

    def translate_english_to_hindi (text): 
        from_code = "en"
        to_code = "hi"
        
        # Download and install Argos Translate package
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        package_to_install = next(
            filter(
                lambda x: x.from_code == from_code and x.to_code == to_code, available_packages
            )
        )
        argostranslate.package.install_from_path(package_to_install.download())
        
        # Translate
        translatedText = argostranslate.translate.translate(text, from_code, to_code)

        return translatedText

    # Final wrapper function
    def translate_from (text, original_language):
        if original_language == "hi":
            return translate_hindi_to_english (text)
        elif original_language == "en":
            return translate_english_to_hindi (text)
    def get_value_from_csv(file_path, row_number):
        try:
            with open(file_path, 'r') as csv_file:
                reader = csv.reader(csv_file)
                for i, row in enumerate(reader):
                    if i == row_number:
                        if len(row) >= 2:
                            return row[1]
                        else:
                            return "Row does not have a second column."
                return "Row not found in the CSV file."
        except FileNotFoundError:
            return "File not found."
        except Exception as e:
            return f"An error occurred: {e}"
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # embeddings = SentenceTransformer('all-MiniLM-L6-v2')
    doc='q_file.csv'
    loader = CSVLoader(file_path=doc)

    docarray = loader.load()

    persist_directory = 'db'

    # print(docarray)
    text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    text=text_splitter.split_documents(docarray)
    # print(text)
    # vectorstore = DocArrayInMemorySearch.from_documents(text,embeddings)
    # vectorstore = Chroma.from_documents(documents=text, embedding=embeddings, persist_directory=persist_directory)
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(' Leber hereditary')
    # vectorstore.persist()
    # vectordb = None
    # print(get_value_from_csv('csv_file.csv',docs[0].metadata['row']))
    context="\n"
    for i in range(len(docs)):
        context += f"Datapoint {i} :"
        context += "\n"
        context+= f"\n{docs[i].page_content}"
        context+= get_value_from_csv('csv_file.csv',docs[i].metadata['row'])
        context += "\n"
    # print(context)
    # Languages = ['HINDI', "ENLGISH"]
    # choice = int(input("Enter a number to choose a language . 1.HINDI . 2. ENGLISH"))
    # options = {
    #     "HINDI": "HINDI",
    #     "ENGLISH": "ENGLISH",
    # }
    # selected_option = st.selectbox("Select an option", list(options.keys()))
    # user_input = st.text_input("कृपया अपनी क्वेरी दर्ज करें", "डिफ़ॉल्ट पाठ")
    # if choice ==1:
    #     # user_input = input("अपनी क्वेरी दर्ज करें")
    #     user_input = input("Enter your query")
    #     # query = translate_hindi_to_english(user_input)
    #     query = user_input
    # else:
    #     user_input = input("Enter your query")
    #     query  = user_input
    # query = 'What are the symptoms of Deep Vein Thrombosis'
    query = promptlol
    prompt=f"""
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to
    make up an answer.The below mentioned information is accurate , latest and relieble

    {context}

    Role: You are an expert AI Doctor who needs to help his patient  in a remote village . He has the below question , help him to the best of your ability. The above mentioned information is accurate , latest and relieble
    Question: {query}

    Refer the below for a sample on how to answer
    " Question : What are tips for managing my bipolar disorder?
    Answer: Along with seeing your doctor and therapist and taking your medicines, simple daily habits can make a difference. Start with these strategies.
    Pay attention to your sleep. This is especially important for people with bipolar disorder.
    Eat well. Theres no specific diet.
    Focus on the basics: Favor fruits, vegetables, lean protein, and whole grains. And cut down on fat, salt, and sugar. 
    Tame stress. (81 words truncated) You can also listen to music or spend time with positive people who are good company. (73 words truncated)
    Limit caffeine. It can keep you up at night and possibly affect your mood. (47 words truncated) 
    Avoid alcohol and drugs. They can affect how your medications work."
    Think Slowly.
    """
    # images =[]
    # for i in range(len(docs)):
    #     print(get_disease('qad_file.csv',docs[i].metadata['row']))
        # if out != 0 and out != 1:
        #     print("OK")
    # qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=db.as_retriever())
    # qa_chain = RetrievalQA.from_chain_type(
    #     llm=OpenAI(),
    #     chain_type="stuff",
    #     retriever=retriever,
    #     return_source_documents=True,
    #     verbose = True
    # )

    # we can now execute queries against our Q&A chain
    # result = qa_chain({'query': 'How many unfilled jobs in wellington'})
    # print(result['result'])
    # llm = OpenAI()
    # result = llm(prompt)
    # with open("input.txt", "w") as file:
    #     # Write the string to the file
    #     file.write(prompt)
    # print(prompt)
    # print(result)
    # sentences = ["This is an example sentence", "Each sentence is converted"]

    # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # embeddings = model.encode(sentences)
    # print(embeddings)
    return prompt


# def image_generation(iprompt):


class MyApp(App):
    
    def build(self):
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Input
        self.text_input = TextInput(hint_text="Enter your prompt here...")
        layout.add_widget(self.text_input)
        
        # Button
        btn = Button(text="Ask Llama")
        btn.bind(on_press=self.ask_button_pressed)
        layout.add_widget(btn)
        
        # Label for feedback
        self.info_label = Label()
        layout.add_widget(self.info_label)
        self.image_grid = GridLayout(cols=2)  # Assuming 2 images per row, adjust as needed
        layout.add_widget(self.image_grid)
        return layout

    def ask_button_pressed(self, instance):
        # Read from the input box
        prompt = self.text_input.text
        
         # Get the response
        #tejas code
        mashqaout=MASHQA_process(prompt)
        # for img in images:
        #     img_path = f"image_{images.index(img)}.png"
        #     img.save(img_path)
        #     kivy_img = KivyImage(source=img_path, keep_ratio=True, allow_stretch=True, size_hint=(1, None), size=(Window.width/2, Window.width/2))
        #     self.image_grid.add_widget(kivy_img)

        
        response = ask_llama(mashqaout)
        # Write to output.txt
        with open("output.txt", "w") as outfile:
            outfile.write(response)

        # max_length = 10000
        # if len(response) > max_length:
        #     response = response[:max_length - 3] + '...'
        


        self.info_label.text = response
        self.info_label.halign = 'center'
        self.info_label.valign = 'middle'
        self.info_label.text_size = (self.info_label.width, None)  # Set the width and let height be determined by the text content
        # Provide feedback to the user
        self.info_label.text = response
    # def ask_button_pressed(self, instance):
    # # Read from the input box
    #     prompt = self.text_input.text

    # # Get the response
    # # tejas code
    #     mashqaout = MASHQA_process(prompt)
    
    # # You can bring back the image saving and displaying code here if needed

    #     response = ask_llama(mashqaout)
    # # Write to output.txt
    #     with open("output.txt", "w") as outfile:
    #         outfile.write(response)

    # # Create a ScrollView to allow scrolling
    #     scroll_view = ScrollView(do_scroll_x=False, do_scroll_y=True)

    # # Adjust the label settings for scrolling
    #     self.info_label = Label(
    #       text=response,
    #       size_hint_y=None,  # Allow the label to grow in height
    #       height=50  # This will be recalculated below
    #  )
        
    #     self.info_label.halign = 'center'
    #     self.info_label.valign = 'middle'
    #     self.info_label.text_size = (None, None)  # Set the width and let height be determined by the text content
    # # Recalculate the label height
    #     self.info_label.bind(texture_size=self.info_label.setter('size'))

    # # Add the label to the ScrollView
    #     scroll_view.add_widget(self.info_label)

    # # Add the ScrollView to the main layout
    #     instance.parent.add_widget(scroll_view)
# def ask_button_pressed(self, instance):
#     # Read from the input box
#     prompt = self.text_input.text

#     # Get the response
#     # tejas code
#     mashqaout = MASHQA_process(prompt)
    
#     # You can bring back the image saving and displaying code here if needed

#     response = ask_llama(mashqaout)
#     # Write to output.txt
#     with open("output.txt", "w") as outfile:
#         outfile.write(response)

#     # Create a ScrollView to allow scrolling
#     scroll_view = ScrollView(do_scroll_x=False, do_scroll_y=True)

#     # Adjust the label settings for scrolling
#     self.info_label = Label(
#         text=response,
#         size_hint_y=None,  # Allow the label to grow in height
#         height=50  # This will be recalculated below
#     )
#     # Recalculate the label height
#     self.info_label.bind(texture_size=self.info_label.setter('size'))

#     # Add the label to the ScrollView
#     scroll_view.add_widget(self.info_label)

#     # Add the ScrollView to the main layout
#     instance.parent.add_widget(scroll_view)




if __name__ == "__main__":
    MyApp().run()
