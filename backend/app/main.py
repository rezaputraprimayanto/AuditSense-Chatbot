from app.api import app
import os
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
