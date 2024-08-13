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
# check

# if __name__ == "__main__":
#     mc = ModelCaller()
#     prompt = """Source:  modules.
# (redacted) and management  Web- based software manages performance & generates precise reports for employee's shift.
# Prepared test scenario and test cases.
# Performed Smoke, Functional, Integration testing.
# Detected bug and tracked the defects manually.
# Smart Dustbin based on real time monitoring:  An application-based smart Dustbin that uses different physical sensors(HC-SR04,MQ2 Gas sensor) to detect the different parameters (overflow, hazardous gases and moisture level) and send these data to waste management authorities.
# Involved in fitting Piezoelectric and Ultrasonic sensor   SKILLS   Manual Testing Core - Java Selenium SQL   Tableau MS-Word MS-Excel MS-PowerPoint   BOOK_METADATA/TRAINING  Software Testing Professional Training at (redacted)  (12/2021 - 06/2022)   Learned all types of Manual Testing and Core-Java(Basic Libraries, OOPS, Object and add Methods to a class and different identifiers and operators) and worked on Automation tool   • (redacted), (redacted), (redacted).
# (05/2018 - 06/2018)   I worked in electronic department on the project of PCB fabrication in which I learned about the fabrication of PCB and Surface Mount Technology.
# MANUAL TESTING SKILLS  Good Knowledge in SDLC and ST
# Instruction: Extract book_title, book_work_description, book_genre, book_author, book_year_published"""
#     INSTRUCTION = ("Instruction: Extract book_title, book_work_description,"
#                    " book_genre, book_author, book_year_published")
#     SAMPLE_OUTPUT = ("'book_title: (synthetic placeholder) | "
#                      "book_summary: Seeking practical experience in web development,"
#                      " interface design, and responsive coding. Eager to apply and expand HTML, "
#                      "CSS, and JavaScript skills within a dynamic team environment. | "
#                      "book_genre: NA | book_author: Frontend Book entry | "
#                      "book_year_published: NA'")
#
#     local_response, gpt40mini_response = mc.generate_and_compare_responses(prompt)
#     print('LOCAL : ', local_response)
#     print(' ')
#     print('GPT RES: ', gpt40mini_response)
