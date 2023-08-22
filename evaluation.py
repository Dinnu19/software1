import pandas as pd
import nltk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer,pipeline
from nltk.translate.bleu_score import corpus_bleu

nltk.download('punkt')

class Evaluation:
    def __init__(self,model_name,src_language,target_language,dataset_path):
        self.model_name = model_name
        self.src_language = src_language
        self.target_language = target_language
        self.dataset_path = dataset_path
        
    def __call__(self):
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name) 
        self.translator = pipeline('translation',model=model,tokenizer=tokenizer,src_lang=self.src_language,tgt_lang=self.target_language,max_length=400)
        bleu_score,word_error_rate = self.evaluate()
        return bleu_score , word_error_rate
        
   
         
    def evaluate(self):
        references = []
        translations = []
        wer_scores = []
        df = pd.read_csv(self.dataset_path)
        source_sentences = df['Text in Punjab']
        target_sentences = df['Text in English']
        for source,target in zip(source_sentences,target_sentences):
            translation = self.translator(source)
            translation_text = translation[0]['translation_text']
            # print(target+"  "+translation_text)
            target_words = nltk.word_tokenize(target.lower())
            predicted_words = nltk.word_tokenize(translation_text.lower())
            references.append([target.lower().split()])
            translations.append(translation_text.lower().split())
            edit_distance = nltk.edit_distance(target_words,predicted_words)
            wer = edit_distance / len(target_words)
            wer_scores.append(wer)
        bleu_score = corpus_bleu(references,translations)
        average_wer = sum(wer_scores) / len(wer_scores)
        return bleu_score , average_wer
    
    
if __name__=='__main__':
    # a=Evaluation('facebook/nllb-200-distilled-600M','hin_deva','eng_Latn','hindi_english_evalaution.csv')
    # a=Evaluation('facebook/nllb-200-distilled-600M','pan_Guru','eng_Latn','punjabitoenglish.csv')
    a = Evaluation('facebook/m2m100_418M','pa','en','punjabitoenglish.csv')
    # a=Evaluation('facebook/m2m100_418M','hi','en','hindi_english_evalaution.csv')
    bleu_score , word_error_rate= a()
    print("bleu_score", bleu_score)
    print("word_error_rate",word_error_rate)
    