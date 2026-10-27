import gradio as gr
import torch

from transformers import pipeline

model_path = ("../Models/models--deepset--roberta-base-squad2/snapshots"
              "/adc3b06f79f797d1c575d5479d6f5efe54a9e3b4")

# question_ans = pipeline("question-answering",
#                         model="deepset/roberta-base-squad2")

question_ans = pipeline("question-answering",
                        model=model_path)

def read_file_content(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            context = file.read()
            return context

    except FileNotFoundError:
        return "Error: File not found."

    except Exception as e:
        return f"Error reading file: {e}"


# Example usage
# content = read_file("sample.txt")
# print(content)

# context = ("Mukesh Dhirubhai Ambani (born 19 April 1957) is an Indian businessman. He is the chairman and managing"
#            "director of Reliance Industries, the largest public company in India by market capitalisation in 2025."
#            "As of June 2026, he is the richest person in Asia and the 22nd richest in the world, with a net worth of US$91.8 billion."
#            "He has attracted fame due to his growth and wealth, and criticism for being a plutocrat, and reports of market manipulation, "
#            "political corruption, cronyism, and exploitation.Born in 1957 to Dhirubhai Ambani,"
#            "the founder of Reliance Industries, and Kokilaben in Aden, Mukesh Ambani completed his studies at St. Xavier"
#            'College, Mumbai and Institute of Chemical Technology. He dropped out of Stanford University in 1980 to join Reliance Industries.'
#            ' He expanded the energy ventures of the company, and directed the set up of its largest petroleum refinery at Jamnagar. '
#             'He took an increasing role in the company after his father developed health issues in the late 1980s. After the death of his father in 2002, '
#             'the ownership of the companies in the group was divided between him and his younger brother Anil Ambani.[12] Later, he expanded the portfolio '
#             'of the group and launched or acquired several ventures in energy, finance, healthcare, media, telecommunications, and sports.')
#
# question ="who is mukesh?"

def get_answer(file, question):
    context =read_file_content(file)
    answer = question_ans(question=question, context=context)
    return answer["answer"]

demo = gr.Interface(fn=get_answer,
    inputs=[gr.File(label="Upload your file"), gr.Textbox(label="Input your question", lines=10)],
    outputs=[gr.Textbox(label="Answer the Given Text",lines=1)],
    title="GENAI PROJECT4: Document Q & A",
    description="THIS APPLICATION WILL BE USED TO ANSWER THE QUESTION BASED ON GIVEN CONTEXT ")

demo.launch()