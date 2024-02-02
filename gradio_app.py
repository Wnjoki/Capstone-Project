#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install gradio  
#!pip install  transformers
#!pip install nltk



# In[2]:


import gradio as gr
import re
import warnings
import pickle


#from gradio.mix import Parallel
from nltk.tokenize import sent_tokenize
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSeq2SeqLM,
    Wav2Vec2ForCTC,
    Wav2Vec2Tokenizer,
    pipeline,
)


# Ignore specific warnings
warnings.filterwarnings("ignore")



# **Load Data and Define Summarizing functions**

# In[37]:


with open('C:/Users/njoki/OneDrive/Desktop/Capstone Project/cleaned_data.pkl', 'rb') as file:
     dataset = pickle.load(file)


# In[38]:


len(dataset)


# In[5]:


dataset[1]


# **a. Summarization via Hugging Face Pipeline**
# 
# 
# Hugging Face's pipeline allows you to load up several summarization models, from FB's Bart to Google's T5. 

# In[6]:


#create a pipeline

pipeline_summ = pipeline(
    "summarization",
    model="facebook/bart-large-cnn", # one can also use"t5-small" etc 
    tokenizer="facebook/bart-large-cnn",
    framework="pt",
)

# First of 2 summarization function
def fb_summarizer(dataset):
    #input_text = dataset
    results = pipeline_summ(dataset)
    return [result["summary_text"] for result in results]



# In[7]:


# First of 2 Gradio apps to use in "parallel"
summary1 = gr.Interface(
    fn=fb_summarizer,
    inputs=gr.Textbox(),
    outputs=gr.Textbox(label="Summary by FB/Bart-large"),
)

summary1.launch(share=True)


# **b. Summarization using Hugging  Face's Model Hub**

# In[8]:


model_name = "google/pegasus-cnn_dailymail" 

# Second of 2 summarization function
def google_summarizer(dataset):
    input_texts = dataset
    
    tokenizer_pegasus = AutoTokenizer.from_pretrained(model_name)

    model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    batch = tokenizer_pegasus.prepare_seq2seq_batch(
        input_texts, truncation=True, padding="longest", return_tensors="pt"
    )
    translated = model_pegasus.generate(**batch)

    pegasus_summary = tokenizer_pegasus.batch_decode(
        translated, skip_special_tokens=True
    )

    return pegasus_summary[0]




# In[9]:


summary2 = google_summarizer(dataset[1])
print(summary2)


# In[10]:


# Second of 2 Gradio apps to use in "parallel"
summary2 = gr.Interface(
    fn=google_summarizer,
    inputs=gr.Textbox(),
    outputs=gr.Textbox(label="Summary by Google/Pegasus-CNN-Dailymail"),
)

summary2.launch(share=True)


# **2. Launch the Models in Parallel**

# In[11]:


get_ipython().system('pip install sentencepiece')


# In[39]:


# Create a Gradio Interface with parallel inputs
def parallel_summaries(text):
    # Call individual summarization functions
    fb_summary = fb_summarizer(text)
    google_summary = google_summarizer(text)
    return fb_summary, google_summary

iface = gr.Interface(
    fn=parallel_summaries,  # List of functions for parallel processing
    inputs=[gr.Textbox(lines=20, label="Paste some text here"),
            gr.File(label="Or Upload a Text File"),
            gr.Dropdown(choices=dataset)],
    outputs=[gr.Textbox(interactive=False, label="Summary 1"), 
             gr.Textbox(interactive=False, label="Summary 2")],
    title="Compare 2 AI Summarizers"
)

# Launch the interface
iface.launch(share=True)


# In[40]:




# In[ ]:




