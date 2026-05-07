from models import Gpt4oMini, LocalModel
from sentence_transformers import util
from config import *

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

lm = LocalModel()
gpt = Gpt4oMini()


class ModelCaller:

    def calculate_similarity(self, similarity_model, text1, text2):
        """Calculate the cosine similarity between two texts using a fast Sentence Transformers model."""
        embeddings1 = similarity_model.encode(text1, convert_to_tensor=True)
        embeddings2 = similarity_model.encode(text2, convert_to_tensor=True)
        return float(util.pytorch_cos_sim(embeddings1, embeddings2).item())

    def generate_and_compare_responses(self, prompt):
        try:
            logger.info("Generating responses for prompt: %s", prompt)
            local_response = lm.generate_local_model_response(prompt, INSTRUCTION)
            print()
            gpt40mini_response = gpt.generate_gpt40mini_response(prompt, INSTRUCTION, SAMPLE_OUTPUT)
            logger.info("Responses generated successfully.")
            return local_response, gpt40mini_response
        except Exception as e:
            logger.error("Error generating responses: %s", e)
            return None, None  # Return None instead of error messages

    def regenerate_gpt4_response(self, prompt):
        """Regenerate GPT-4 response for the given prompt."""
        logger.info("Regenerating GPT-4 response for prompt: %s", prompt)
        _, new_gpt40mini_response = self.generate_and_compare_responses(prompt)
        return new_gpt40mini_response
