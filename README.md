# music-generation

Generating music with LSTM neural network.

## Instructions:
Put songs in SONGS_DIR directory ('mozart_songs' in this example).

First parse the songs:
python music.py --mode parse

To begin to learn a new model:
python music.py --mode new --model name --epochs number_of_epochs --attr attr
Where attr is one of: "offset", "length", "melody"

To continue learning (weights are automatically saved in data directory):
python music.py --mode train --model name --epochs number_of_epochs --attr attr

To generate a song:
python music.py --mode generate --model name --length lenth

Enjoy!