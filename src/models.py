import os
import torch
import config as cf
from openai import OpenAI
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Gpt4oMini:
    def __init__(self):
        self.model = cf.GPT_MODEL
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def prompt_reader(self, file_name):
        try:
            with open(cf.prompts_dir + file_name, 'r') as file:
                content = file.read()
                return content
        except FileNotFoundError:
            print("The file at given path was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def gpt_api_call(self, prompt, function_name):
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": f"{prompt}"}])
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in {function_name}: {e}")
            return "error"

    def generate_gpt40mini_response(self, source, instruction, sample_output):
        prompt = self.prompt_reader('prompt.txt')
        prompt = prompt.format(source=source,
                               instruction=instruction,
                               sample_output=str(sample_output))
        return self.gpt_api_call(prompt, 'generate_gpt40mini_response')


class LocalModel:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained(cf.LOCAL_MODEL_PATH)
        self.model = GPT2LMHeadModel.from_pretrained(cf.LOCAL_MODEL_PATH)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate_text(self, prompt, max_length=100):
        self.model.eval()
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        attention_mask = torch.ones(inputs.shape, device=self.device)
        with torch.no_grad():
            outputs = self.model.generate(inputs, attention_mask=attention_mask, max_length=max_length,
                                          num_return_sequences=1, pad_token_id=self.tokenizer.eos_token_id)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def generate_local_model_response(self, prompt, instruction):
        _prompt = 'Source: ' + prompt + '\n' + 'Instruction: ' + instruction
        generated_text = self.generate_text(_prompt, max_length=512)
        predicted_output = generated_text.split('Output:', 1)[1]
        return predicted_output
