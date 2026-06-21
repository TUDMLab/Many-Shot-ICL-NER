# pip3 install transformers
# python3 deepseek_tokenizer.py
import transformers

chat_tokenizer_dir = "./"

tokenizer = transformers.AutoTokenizer.from_pretrained( 
        chat_tokenizer_dir, trust_remote_code=True
        )


# read a txt file
with open("./prompt_batch-0.txt", "r", encoding="utf-8") as f:
    txt = f.read()

result = tokenizer.encode(txt)
print(result)
print(len(result))
