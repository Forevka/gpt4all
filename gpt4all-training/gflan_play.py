from random import randrange        
from transformers import pipeline, AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
from datasets import load_dataset

# Replace with your custom model of choice
#model = AutoModel.from_pretrained('/path/to/your/model')
model_id = "google/flan-t5-base"
dataset_id = "samsum"

dataset = load_dataset(dataset_id) 

model_checkpoint = "checkpoint-18415"

local_model_path = f"/home/forevka/deep/flan-t5-base-samsum"
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

#tokenizer.save_pretrained("flan-t5-base-pretrained")
#model = AutoModelForSeq2SeqLM.from_pretrained(model_id).cuda()
# load model and tokenizer from huggingface hub with pipeline
summarizer = pipeline("summarization", model=local_model_path + "/" + model_checkpoint, tokenizer=tokenizer,)

# select a random test sample
sample = """Abby: Have you talked to Miro?
Dylan: No, not really, I've never had an opportunity
Brandon: me neither, but he seems a nice guy
Brenda: you met him yesterday at the party?
Abby: yes, he's so interesting
Abby: told me the story of his father coming from Albania to the US in the early 1990s
Dylan: really, I had no idea he is Albanian
Abby: he is, he speaks only Albanian with his parents
Dylan: fascinating, where does he come from in Albania?
Abby: from the seacoast
Abby: Duress I believe, he told me they are not from Tirana
Dylan: what else did he tell you?
Abby: That they left kind of illegally
Abby: it was a big mess and extreme poverty everywhere
Abby: then suddenly the border was open and they just left 
Abby: people were boarding available ships, whatever, just to get out of there
Abby: he showed me some pictures, like <file_photo>
Dylan: insane
Abby: yes, and his father was among the people
Dylan: scary but interesting
Abby: very!"""
print(f"dialogue: \n{sample}\n---------------")

# summarize dialogue
res = summarizer(sample)

print(f"flan-t5-base-sumsam summary:\n{res[0]['summary_text']}")
