import os
import sys
sys.path.insert(0, os.path.abspath('C:\Users\24441\Downloads\nmt-master\nmt-master\nmt'))

import ushahidi_sphinx_rtd_theme
html_theme = "ushahidi_sphinx_rtd_theme"
html_theme_path = [ushahidi_sphinx_rtd_theme.get_html_theme_path()]

# Embedding
embedding_encoder = variable_scope.get_variable(
    "embedding_encoder", [src_vocab_size, embedding_size], ...)
# Look up embedding:
#   encoder_inputs: [max_time, batch_size]
#   encoder_emb_inp: [max_time, batch_size, embedding_size]
encoder_emb_inp = embedding_ops.embedding_lookup(embedding_encoder, encoder_inputs)