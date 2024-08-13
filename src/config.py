# model parameters
GPT_MODEL = "gpt-4o-mini"

prompts_dir = '../data/'
LOCAL_MODEL_PATH = '../data/models/book_metadata_model'

current_json_file_path = '../data/temp/current_response.json'

# Constants for instructions - BOOK_METADATA
INSTRUCTION = ("Instruction: Extract book_title, book_work_description,"
               " book_genre, book_author, book_year_published")

SAMPLE_OUTPUT = ("'book_title: (synthetic placeholder) | "
                 "book_summary: Seeking practical experience in web development,"
                 " interface design, and responsive coding. Eager to apply and expand HTML, "
                 "CSS, and JavaScript skills within a dynamic team environment. | "
                 "book_genre: NA | book_author: Frontend Book entry | "
                 "book_year_published: NA'")