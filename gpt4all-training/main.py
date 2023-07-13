'''
import torch

cuda_available = torch.cuda.is_available()
if cuda_available:
    print("CUDA is available")
else:
    print("CUDA is not available")
'''

from gpt4all import GPT4All

model = GPT4All(model_name='orca-mini-3b.ggmlv3.q4_0.bin')
#

with model.chat_session():
    response = model.generate(prompt='how i am named in malina telegram', top_k=1)
    for i in model.current_chat_session:
        print(i)

