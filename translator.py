from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from nltk.tokenize import word_tokenize
import nltk
import config
import os

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

nltk.download("punkt_tab")

class Translator:

    def __init__(self, input_file: str, output_file: str) -> None:
        self.__input_file = input_file
        self.__output_file = output_file
        self.__model = AutoModelForSeq2SeqLM.from_pretrained(config.MODEL)
        self.__tokenizer = AutoTokenizer.from_pretrained(config.MODEL)
        self.__input_raw_text = ""
        self.__chunks = []
        self.__chunk_tokens = {} # {chunk : tokens}
        self.__translated_chunks = []
    
    def __readTextFromFile(self, fileName: str) -> None:
        with open(fileName, 'r', encoding='utf-8') as file:
            self.__input_raw_text = file.read()

    def __split_raw_text_into_chunks(self) -> None:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = config.CHUNK_SIZE, 
            separators = ["\n\n", "\n", ". ", "! ", "? ", " ", ""], 
            is_separator_regex=False, 
            length_function = len, 
        )
        self.__chunks = text_splitter.split_text(self.__input_raw_text)

    def __tokenize_chunks(self) -> None:
        for chunk in self.__chunks:
            self.__chunk_tokens[chunk] = self.__tokenizer(chunk, return_tensors="pt")

    def __translate(self) -> None:
        for chunk, chunk_token in self.__chunk_tokens.items():
            outputs = self.__model.generate(**chunk_token)
            self.__translated_chunks.append(self.__tokenizer.decode(outputs[0], skip_special_tokens=True))

    def start(self) -> None:
        self.__readTextFromFile("input.txt")
        self.__split_raw_text_into_chunks()
        self.__tokenize_chunks()
        self.__translate()
        # print(self.__tokenizer.model_max_length)
        with open(self.__output_file, 'w', encoding='utf-8') as file:
            file.writelines(self.__translated_chunks)