# model parameters
GPT_MODEL = "gpt-4o-mini"

prompts_dir = '../data/'
LOCAL_MODEL_PATH = '../data/models/book_metadata_model'

current_json_file_path = '../data/temp/current_response.json'

# Constants for instructions — book metadata extraction
INSTRUCTION = ("Instruction: Extract book_title, book_author,"
               " book_year_published, book_genre, book_summary")

SAMPLE_OUTPUT = ("'book_title: The Great Gatsby | "
                 "book_author: F. Scott Fitzgerald | "
                 "book_year_published: 1925 | "
                 "book_genre: Literary fiction | "
                 "book_summary: A young Long Island bond trader narrates the rise and fall of Jay Gatsby, "
                 "a self-made millionaire whose obsession with Daisy Buchanan exposes the hollowness of the Jazz Age.'")
