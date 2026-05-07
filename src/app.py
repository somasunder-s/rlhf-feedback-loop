import os
import io
import json
import logging
import pandas as pd
from sentence_transformers import SentenceTransformer
from taipy.gui import Gui, notify
import taipy.gui.builder as tgb

from config import *
from helpers import *
from model_caller import ModelCaller

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize model caller and similarity model
mc = ModelCaller()
similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Initialize global state variables
csv_path = None
data = None
sliced_df = None

initial_row, final_row = 0, 7
cur_actual_row_num = initial_row
current_row = 0

file_name, prompt, val = None, None, None

response_data = {
    "actual_row_num": None,
    "file_name": None,
    "source_text": None,
    "local_model_response": None,
    "gpt4_response": None,
    "local_model_feedback": {},
    "gpt4_feedback": {},
    "text_feedback": {},
    "similarity_score": {}
}

df = pd.DataFrame(columns=response_data.keys())
print(df)

# Initialize current values
current_local_val = {
    "book_title": "Loading..",
    "book_summary": "Loading..",
    "book_genre": "Loading..",
    "book_author": "Loading..",
    "book_year_published": "Loading.."
}

current_gpt_val = {
    "book_title": "Loading..",
    "book_summary": "Loading..",
    "book_genre": "Loading..",
    "book_author": "Loading..",
    "book_year_published": "Loading.."
}

feedback_vals_gpt = {key: None for key in current_gpt_val.keys()}
feedback_vals_local = {key: None for key in current_local_val.keys()}
feedback_as_text_input = {key: None for key in current_local_val.keys()}
current_similarity_score = {}

# Initialize variables for book entry
title, summary, genre, author, year_published = [None] * 5
li_title, li_summary, li_genre, li_author, li_year_published = [None] * 5
gi_title, gi_summary, gi_genre, gi_author, gi_year_published = [None] * 5

buffer, content = None, None
csv_file_name_to_be_saved = None

local_model_correct_predictions, lmp = 0, 0
gpt4_correct_predictions, gmp = 0, 0
total_predictions = 0

button_name = 'Save & Next'


def load_csv_file(state):
    """Load the CSV file and initialize data."""
    try:
        state.data = pd.read_csv(state.csv_path)
        state.data = state.data[['file_name', 'split_on_tokens']]
        state.initial_row, state.final_row = 0, min(len(state.data), 7)
        logger.info("CSV file loaded successfully.")
        notify(state, 'info', 'CSV file loaded successfully.')
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        notify(state, 'error', f'Error loading CSV file: {e}')


def slicing_df(state):
    """Slice the DataFrame based on initial and final rows."""
    try:
        logger.info("Data length: %d", len(state.data))
        state.sliced_df = state.data[int(state.initial_row):int(state.final_row)].reset_index()
        logger.info('Sliced DataFrame: %s', state.sliced_df)
        state.current_row = -1  # Reset current row as new slice is selected
        feedback_taker(state)
        notify(state, 'info', 'Go to Provide_Feedback section.')
    except Exception as e:
        logger.error(f"DataFrame could not be sliced: {e}")
        notify(state, 'error', 'Upload the csv data, '
                               'which contains columns file_name and split_on_tokens')


def process_model_data(state):
    """Process model data for the current row."""
    if state.current_row < 0 or state.current_row >= len(state.sliced_df):
        logger.warning("Current row index is out of range.")
        notify(state, 'warning', 'Current row index is out of range.')
        return

    state.cur_actual_row_num = str(state.sliced_df['index'][state.current_row])
    state.file_name = str(state.sliced_df['file_name'][state.current_row])
    state.prompt = f"Source: {state.sliced_df['split_on_tokens'][state.current_row]}\n{INSTRUCTION}"

    state.response_data['actual_row_num'] = state.cur_actual_row_num
    state.response_data['file_name'] = state.file_name
    state.response_data['source_text'] = state.sliced_df['split_on_tokens'][state.current_row]

    if not pd.isna(state.prompt):
        local_response, gpt40mini_response = mc.generate_and_compare_responses(state.prompt)
        state.response_data['local_model_response'] = local_response
        state.response_data['gpt4_response'] = gpt40mini_response
        logger.info("Model data processed successfully.")
        notify(state, 'info', 'Model has processed the responses!')
    else:
        logger.warning("Empty prompt found at index %d", state.current_row)
        notify(state, 'warning', 'Empty prompt found at index %d' % state.current_row)


def process_next(state):
    """Process the next row."""
    if state.current_row < len(state.sliced_df) - 1:
        state.current_row += 1
        process_model_data(state)
        # Reset values for new row
        state.title, state.summary, state.genre, state.author, state.year_published = [None] * 5
        state.li_title, state.li_summary, state.li_genre, state.li_author, state.li_year_published = (
            "book_title", "book_summary", "book_genre", "book_author", "book_year_published")
        state.gi_title, state.gi_summary, state.gi_genre, state.gi_author, state.gi_year_published = (
            "book_title", "book_summary", "book_genre", "book_author", "book_year_published")
        logger.info("Moved to the next row.")
        # notify(state, 'info', 'Moved to the next row.')
    else:
        state.button_name = 'Thank you, Feedback Completed! Move to download section!'
        logger.info("You have completed the feedback for selected rows.")
        notify(state, 'info', "Feedback Completed, Download the Feedback!")


def feedback_taker(state):
    """Take feedback from the model responses."""
    process_next(state)

    local_response_split = state.response_data['local_model_response'].split('|')
    gpt40mini_response_split = state.response_data['gpt4_response'].split('|||')[0].split('|')

    attributes = ["book_title", "book_summary", "book_genre",
                  "book_author", "book_year_published"]

    for attr in attributes:
        local_value = next((item.split(':', 1)[1].strip() for item in local_response_split if
                            item.split(':', 1)[0].strip() == attr), 'NA')
        gpt_value = next((item.split(':', 1)[1].strip().strip("'") for item in gpt40mini_response_split if
                          item.split(':', 1)[0].strip().strip("'") == attr), 'NA')

        similarity_score = mc.calculate_similarity(similarity_model, local_value, gpt_value)

        state.current_gpt_val[attr] = gpt_value
        state.current_local_val[attr] = local_value
        state.current_similarity_score[attr] = round(similarity_score, 2)

    state.response_data['similarity_score'] = str(dict(state.current_similarity_score))
    auto_checks(state)


def auto_checks(state):
    sim_dict = dict(state.current_similarity_score)

    for attr, sim_scr in sim_dict.items():
        if sim_scr >= 0.67:
            state.feedback_vals_local[attr] = state.current_local_val[attr]
            state.feedback_vals_gpt[attr] = state.current_gpt_val[attr]
        else:
            state.feedback_vals_local[attr] = None
            state.feedback_vals_gpt[attr] = None

    # for local values
    state.li_title = state.feedback_vals_local[state.li_title]
    state.li_summary = state.feedback_vals_local[state.li_summary]
    state.li_genre = state.feedback_vals_local[state.li_genre]
    state.li_author = state.feedback_vals_local[state.li_author]
    state.li_year_published = state.feedback_vals_local[state.li_year_published]

    # for gpt values
    state.gi_title = state.feedback_vals_gpt[state.gi_title]
    state.gi_summary = state.feedback_vals_gpt[state.gi_summary]
    state.gi_genre = state.feedback_vals_gpt[state.gi_genre]
    state.gi_author = state.feedback_vals_gpt[state.gi_author]
    state.gi_year_published = state.feedback_vals_gpt[state.gi_year_published]


def submit_feedback(state):
    """Submit feedback and save it."""
    add_text_feedback(state)

    state.response_data['local_model_feedback'] = str(dict(state.feedback_vals_local))
    state.response_data['gpt4_feedback'] = str(dict(state.feedback_vals_gpt))
    state.response_data['text_feedback'] = str(dict(state.feedback_as_text_input))

    calculate_model_scores(state)

    logger.info('GPT Feedback Output: %s', dict(state.feedback_vals_gpt))
    logger.info('Local Feedback Output: %s', dict(state.feedback_vals_local))
    logger.info('Text Feedback: %s', dict(state.feedback_as_text_input))

    # Convert the new data into a DataFrame
    json2df = pd.DataFrame([dict(state.response_data)])
    state.df = pd.concat([state.df, json2df], ignore_index=True)
    logger.info('Data Frame: \n%s', state.df)

    save_json(dict(state.response_data), current_json_file_path)

    # Buffering to download as CSV
    state.buffer = io.BytesIO()
    state.df.to_csv(state.buffer, index=False)
    state.buffer.seek(0)
    state.content = state.buffer.getvalue()

    # CSV file name
    state.csv_file_name_to_be_saved = (f"intern_feedback_row_{state.initial_row}_to_{state.cur_actual_row_num}_"
                                       f"loc_{state.lmp}_gpt{state.gmp}.csv")
    notify(state, 'info', 'Feedback submitted.')

    # After saving, Proceeding with Next Feedback
    feedback_taker(state)


def update_local_feedback(state, var: str, val: str):
    """Update local feedback values."""
    attr_name = 'book' + str(var)[2:]
    state.feedback_vals_local[attr_name] = val[0] if val else None
    logger.info('Local Current Value: %s', state.current_local_val[attr_name])
    logger.info('Local Feedback Value: %s', state.feedback_vals_local[attr_name])

def update_gpt_feedback(state, var: str, val: str):
    """Update GPT feedback values."""
    attr_name = 'book' + str(var)[2:]
    state.feedback_vals_gpt[attr_name] = val[0] if val else None
    logger.info('GPT Current Value: %s', state.current_gpt_val[attr_name])
    logger.info('GPT Feedback Value: %s', state.feedback_vals_gpt[attr_name])

def add_text_feedback(state):
    """Add text feedback for the current book."""
    state.feedback_as_text_input['book_title'] = state.title
    state.feedback_as_text_input['book_summary'] = state.summary
    state.feedback_as_text_input['book_genre'] = state.genre
    state.feedback_as_text_input['book_author'] = state.author
    state.feedback_as_text_input['book_year_published'] = state.year_published
    logger.info("Text feedback updated: %s", dict(state.feedback_as_text_input))

def calculate_model_scores(state):
    state.local_model_correct_predictions = (state.local_model_correct_predictions +
                                             len([value for value in dict(state.feedback_vals_local).values()
                                                  if value is not None]))
    print('LV: ', dict(state.feedback_vals_local).values())
    state.gpt4_correct_predictions = (state.gpt4_correct_predictions +
                                      len([value for value in dict(state.feedback_vals_gpt).values()
                                          if value is not None]))
    print('GV: ', dict(state.feedback_vals_gpt).values())
    state.total_predictions = state.total_predictions + 5
    state.lmp = round((state.local_model_correct_predictions/state.total_predictions)*100)
    print(state.local_model_correct_predictions)
    state.gmp = round((state.gpt4_correct_predictions / state.total_predictions) * 100)
    print(state.gpt4_correct_predictions)


# UI Definition
with tgb.Page() as root_page:
    tgb.navbar()

with tgb.Page() as upload_data:
    tgb.text("## Upload the file", mode="md")
    with tgb.layout('1 1 1 1'):
        with tgb.part():
            tgb.file_selector("{csv_path}", label="Upload file", on_action=load_csv_file, extensions=".csv")
        with tgb.part():
            tgb.input("{initial_row}", label="Initial row")
        with tgb.part():
            tgb.input("{final_row}", label="Final row")
        with tgb.part():
            tgb.button("Read Data", on_action=slicing_df)
    tgb.table("{sliced_df}", allow_all_rows=True, rebuild=True, on_action=load_csv_file)

with tgb.Page() as view_model_response:
    tgb.text("## Model Responses {current_row + 1}", mode="md")
    tgb.text("### Source", mode="md")
    tgb.text("{response_data['source_text']}", mode='md')
    tgb.text("### Local Model Response", mode="md")
    tgb.text("{response_data['local_model_response']}")
    tgb.text("### Gpt4 Response", mode="md")
    tgb.text("{response_data['gpt4_response']}")
    tgb.status("{current_row}")

with tgb.Page() as provide_feedback:
    with tgb.layout('1 1 1 1'):
        with tgb.part():
            tgb.text("#### Attributes", mode="md")
            tgb.text('##### title', mode="md")
            tgb.text('##### author', mode="md")
            tgb.text('##### genre', mode="md")
            tgb.text('##### year_published', mode="md")
            tgb.text('##### summary', mode="md")
        with tgb.part():
            tgb.text("#### Local Model Response : {lmp}%", mode="md")
            tgb.selector(value="{li_title}", lov="{[current_local_val['book_title']]}",
                         mode="check", on_change=update_local_feedback)
            tgb.selector(value="{li_author}", lov="{[current_local_val['book_author']]}",
                         mode="check", on_change=update_local_feedback)
            tgb.selector(value="{li_genre}", lov="{[current_local_val['book_genre']]}",
                         mode="check", on_change=update_local_feedback)
            tgb.selector(value="{li_year_published}", lov="{[current_local_val['book_year_published']]}",
                         mode="check", on_change=update_local_feedback)
            tgb.selector(value="{li_summary}", lov="{[current_local_val['book_summary']]}",
                         mode="check", on_change=update_local_feedback)
        with tgb.part():
            tgb.text("#### Gpt4oMini Response : {gmp}%", mode="md")
            tgb.selector(value="{gi_title}", lov="{[current_gpt_val['book_title']]}",
                         mode="check", on_change=update_gpt_feedback)
            tgb.selector(value="{gi_author}", lov="{[current_gpt_val['book_author']]}",
                         mode="check", on_change=update_gpt_feedback)
            tgb.selector(value="{gi_genre}", lov="{[current_gpt_val['book_genre']]}",
                         mode="check", on_change=update_gpt_feedback)
            tgb.selector(value="{gi_year_published}", lov="{[current_gpt_val['book_year_published']]}",
                         mode="check", on_change=update_gpt_feedback)
            tgb.selector(value="{gi_summary}", lov="{[current_gpt_val['book_summary']]}",
                         mode="check", on_change=update_gpt_feedback)
        with tgb.part():
            tgb.text("#### Manually Provide Response", mode="md")
            tgb.input(value="{title}", on_change=add_text_feedback)
            tgb.input(value="{author}", on_change=add_text_feedback)
            tgb.input(value="{genre}", on_change=add_text_feedback)
            tgb.input(value="{year_published}", on_change=add_text_feedback)
            tgb.input(value="{summary}", on_change=add_text_feedback)

    with tgb.layout('1 1 1 1 1'):
        with tgb.part():
            tgb.text("", mode="md")
        with tgb.part():
            tgb.text("", mode="md")
        with tgb.part():
            tgb.text("", mode="md")
        with tgb.part():
            tgb.text("", mode="md")
        with tgb.part():
            tgb.button("{button_name}", on_action=submit_feedback)

    tgb.text("### Source", mode="md")
    tgb.text("{response_data['source_text']}")

with tgb.Page() as download_data:
    tgb.text("## Download Feedback", mode="md")
    tgb.file_download(content="{content}", name="{csv_file_name_to_be_saved}",
                      label='Download CSV', hover_text='For whatever you have completed so far')

# Create a dictionary to map page names to their content
pages = {
    "/": root_page,
    "Upload_Data": upload_data,
    "Provide_Feedback": provide_feedback,
    "Model_Response": view_model_response,
    "Download_Feedback": download_data,
}

# Initialize and run the Taipy GUI
Gui(pages=pages).run(title='RLHF Application', use_reloader=True, debug=True)
