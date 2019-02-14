import torch.nn as nn
import torch.nn.functional as F
import glob
import pickle
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from music21 import converter, instrument, note, stream, meter, tempo, pitch, interval
import argparse
import yaml

# Offset 400 epochs
# Melody 300 until now, need more
# Length ??? 

SEQ_LEN = 50
BATCH_SIZE = 20
HIDDEN_FEATURES = 256
LIN = 128
LAYERS = 2
FEATURES = 3
SONGS_DIR = 'mozart_songs'

cuda = torch.device('cuda')

np.random.seed(0)

class LSTM(nn.Module):
    # n - number of different notes
    def __init__(self, n):
        super(LSTM, self).__init__()
        # Two layer LSTM
        self.lstm = nn.LSTM(FEATURES, HIDDEN_FEATURES, LAYERS, dropout=0.5)
        self.fc1 = nn.Linear(HIDDEN_FEATURES, LIN)
        self.fc2 = nn.Linear(LIN, n)

    def init_hidden(self, batch):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(LAYERS, batch, HIDDEN_FEATURES).to(cuda),
                torch.zeros(LAYERS, batch, HIDDEN_FEATURES).to(cuda))

    def forward(self, x):
        x, self.hidden = self.lstm(x.view(SEQ_LEN, x.shape[0], FEATURES), self.hidden)
        x = F.dropout(self.fc1(x[-1]))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

def parse():
    """ Parse midi files from ./songs directory and save notes to ./data/notes """
    songs = []

    with open(SONGS_DIR + "/desc.yaml", 'r') as stream:
        try:
            files = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    for file in files:
        notes = []
        midi = converter.parse(SONGS_DIR + '/' + file['name'])
        transp = interval.Interval(file['interval'])
        print("Parsing %s transposition: %s" % (file['name'], str(transp)))
        
        s = instrument.partitionByInstrument(midi)
        notes_to_parse = s.parts[0].recurse()
        
        quarters = 4
        last_offset = 0

        def key(x):
            if isinstance(x, note.Note):
                return (x.offset, x.pitch)
            else:
                return (x.offset, pitch.Pitch('A0'))
        # Sort notes that are played simultaneously
        notes_to_parse = sorted(notes_to_parse, key=key)

        for element in notes_to_parse:
            n = {}
            if last_offset + quarters <= element.offset:
                last_offset += quarters
            if isinstance(element, note.Note):
                # pitch, offset, length
                n = str(element.pitch.transpose(transp)), round(element.offset-last_offset, 2), element.quarterLength
                notes.append(n)
            elif isinstance(element, meter.TimeSignature):
                quarters = element.numerator/(element.denominator/4)
                n = 'TS'+str(quarters), element.offset-last_offset, 0
                notes.append(n)
        songs.append(notes)
        print("Parsed %s with length: %d" % (file['name'], len(notes)))

    print('Classes:', len(set(pitch for pitch, _, _ in songs[0])))

    with open('data/songs', 'wb') as filepath:
        pickle.dump(songs, filepath)

def load_notes():
    """ Load notes from songs """
    notes = []
    for song in load_songs():
        notes += song
    return notes

def load_songs():
    """ Load songs from ./data/songs """
    with open('data/songs', 'rb') as file:
        songs = pickle.load(file)
    return songs

def different_classes(attr='offset'):
    """ Returns number of different classes """
    if attr=='offset':
        elements = get_offsets(load_notes())
    elif attr=='length':
        elements = get_lengths(load_notes())
    else:
        elements = get_pitches(load_notes())
        
    return len(set(elements))

def load_model(model, name):
    model.load_state_dict(torch.load('data/%s.pt' % name))
    model.eval()

def save_model(model, name):
    torch.save(model.state_dict(), 'data/%s.pt' % name)

def get_pitches(notes):
    return [pitch for pitch, _, _ in notes]
def get_offsets(notes):
    return [offset for _, offset, _ in notes]
def get_lengths(notes):
    return [length for _, _, length in notes]
def get_pitch_to_int(notes):
    return dict((note, number) for number, note in enumerate(sorted(set(get_pitches(notes)))))
def get_offset_to_int(notes):
    return dict((note, number) for number, note in enumerate(sorted(set(get_offsets(notes)))))
def get_length_to_int(notes):
    return dict((note, number) for number, note in enumerate(sorted(set(get_lengths(notes)))))

def get_trainloader(attr='offset'):
    """ Get trainloader """
    songs = load_songs()
    notes = load_notes()
    print(len(songs), " songs in trainloader")
    print(len(notes), " notes in trainloader")
    print(different_classes(attr=attr), " different classes")
    pitch_to_int = get_pitch_to_int(notes)
    offset_to_int = get_offset_to_int(notes)
    length_to_int = get_length_to_int(notes)
    
    pitches = get_pitches(notes)
    offsets = get_offsets(notes)
    lengths = get_lengths(notes)

    data = []
    labels = []
    for song in songs:
        for i in range(0, len(song) - SEQ_LEN):
            seq_pitches = [pitch_to_int[pitch] for pitch in pitches[i:i + SEQ_LEN]]
            seq_offsets = [offset_to_int[offset] for offset in offsets[i:i + SEQ_LEN]]
            seq_lengths = [length_to_int[length] for length in lengths[i:i + SEQ_LEN]]
            if attr=='offset':
                successor = offsets[i + SEQ_LEN]
                labels.append(offset_to_int[successor])
            elif attr=='length':
                labels.append(seq_lengths[-1])
                seq_lengths[-1] = -1
            else: # Melody
                labels.append(seq_pitches[-1])
                seq_lengths[-1] = -1
                seq_pitches[-1] = -1
            data.append([seq_offsets, seq_lengths, seq_pitches])

    data = np.array(data)
    trainset = list(zip(data.astype(np.float32), labels))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=False)

    return trainloader

def train(model, trainloader, epochs=100, name='net'):
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(cuda)
            labels = labels.to(cuda)
            model.zero_grad()
            model.hidden = model.init_hidden(inputs.shape[0])

            # forward + backward + optimize
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if (i * BATCH_SIZE) % 400 == 0 and i != 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f running: %.3f' %
                    (epoch + 1, i*BATCH_SIZE + 1, loss.item(), running_loss))
                running_loss = 0.0
        if epoch % 100 == 99:
            save_model(model, name+'_'+str(epoch+1))

def random_arg(probabilities):
    rand = np.random.random()
    p = 0
    for i, prob in enumerate(probabilities):
        p += prob
        if p >= rand:
            return i


def generate_song(model_offset, model_length, model_melody, length):
    """ Generate song """
    notes = load_notes()
    int_to_pitch = dict((number, note) for number, note in enumerate(sorted(set(get_pitches(notes)))))
    int_to_offset = dict((number, note) for number, note in enumerate(sorted(set(get_offsets(notes)))))
    int_to_length = dict((number, note) for number, note in enumerate(sorted(set(get_lengths(notes)))))

    # Starting pattern TODO change it later
    trainloader = get_trainloader(attr='offset')
    pattern = list(trainloader)[:1][0][0][0].view(1, FEATURES, SEQ_LEN).to(cuda)

    # Begin song with time signature
    song = [(int_to_pitch[int(pattern[0, 2, 0])],0,0)]
    print((int_to_pitch[int(pattern[0, 2, 0])],0,0))

    model_offset.hidden = model_offset.init_hidden(1)
    model_length.hidden = model_length.init_hidden(1)
    model_melody.hidden = model_melody.init_hidden(1)
    
    # generate 500 notes
    for note_index in range(length):
        prediction_input = pattern
        
        # Offset
        
        prediction = model_offset(prediction_input)
        probabilities = prediction.view(prediction.numel()).exp().cpu().detach().numpy()
        offset_class = random_arg(probabilities)
        offset = int_to_offset[offset_class]
        
        new_pattern = np.roll(pattern.view(FEATURES, SEQ_LEN).cpu().numpy(), -1, axis=1)
        new_pattern[:, SEQ_LEN-1] = [offset_class, -1, -1]
        prediction_input = torch.from_numpy(new_pattern).view(1, FEATURES, SEQ_LEN).to(cuda)
        
        # Length
        prediction = model_length(prediction_input)
        probabilities = prediction.view(prediction.numel()).exp().cpu().detach().numpy()
        length_class = random_arg(probabilities)
        length = int_to_length[length_class]
        
        new_pattern[:, SEQ_LEN-1] = [offset_class, length_class, -1]
        prediction_input = torch.from_numpy(new_pattern).view(1, FEATURES, SEQ_LEN).to(cuda)
        
        # Melody
        prediction = model_melody(prediction_input)
        probabilities = prediction.view(prediction.numel()).exp().cpu().detach().numpy()
        pitch_class = random_arg(probabilities)
        pitch = int_to_pitch[pitch_class]
        
        new_pattern[:, SEQ_LEN-1] = [offset_class, length_class, pitch_class]
        pattern = torch.from_numpy(new_pattern).view(1, FEATURES, SEQ_LEN).to(cuda)
        print((pitch, offset, length))
        song.append((pitch, offset, length))

    return song

def save_song(song, name):
    """ Convert list of ints to notes and save as a midi file """
    offset = 0
    prev_offset = 0
    output_notes = [tempo.MetronomeMark('allegro')]
    quarters = 3 # Quarters should be first element in the song, so it shouldn't have any effect
    # create note and chord objects based on the values generated by the model
    for e_pitch, e_offset, e_length in song:
        if e_offset < prev_offset:
            offset += quarters
        if e_pitch[:2] == 'TS':
            quarters = float(e_pitch[2:])
            new_note = meter.TimeSignature(str(int(float(e_pitch[2:]))) + '/4')
        elif float(e_length) != 0.0:
            new_note = note.Note(e_pitch)
            new_note.offset = offset + e_offset
            new_note.quarterLength = e_length
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        prev_offset = e_offset

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp='%s.mid' % name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--mode', type=str,
                        help='parse, train, generate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='parse, train, generate')
    parser.add_argument('--model', type=str, default='net',
                        help='Model name to save')
    parser.add_argument('--song', type=str, default='song',
                        help='Song name to save (default: song)')
    parser.add_argument('--length', type=int, default=100,
                        help='Song length to save (default: 100)')
    parser.add_argument('--attr', type=str, default='offset',
                        help='melody/offset/length')
    args = parser.parse_args()

    if args.mode not in ['parse', 'train', 'generate', 'new']:
        print("Choose a mode!")
        exit()
    if args.attr=='melody':
        FEATURES = 3
    
    if args.mode == 'parse':
        print('Parsing mode')
        parse()
        save_song(load_songs()[0], 'parsed')
    elif args.mode == 'new':
        print('New model mode')
        trainloader = get_trainloader(attr=args.attr)
        model = LSTM(different_classes(attr=args.attr)).to(cuda)
        train(model, trainloader, args.epochs, args.attr + '_' + args.model)
        save_model(model, args.attr + '_' + args.model)
    elif args.mode == 'train':
        print('Train model mode')
        trainloader = get_trainloader(attr=args.attr)
        model = LSTM(different_classes(attr=args.attr)).to(cuda)
        load_model(model, args.attr + '_' + args.model)
        model.train()
        train(model, trainloader, args.epochs, args.attr + '_' + args.model)
        save_model(model, args.attr + '_' + args.model)
    elif args.mode == 'generate':
        print('Generating mode')
        SEQ_LEN = 1
        model_offset = LSTM(different_classes(attr='offset')).to(cuda)
        model_length = LSTM(different_classes(attr='length')).to(cuda)
        model_melody = LSTM(different_classes(attr='melody')).to(cuda)
        load_model(model_offset, 'offset_' + args.model)
        load_model(model_length, 'length_' + args.model)
        load_model(model_melody, 'melody_' + args.model)
        song = generate_song(model_offset, model_length, model_melody, args.length)
        save_song(song, args.model)

