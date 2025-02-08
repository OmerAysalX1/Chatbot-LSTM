import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


df_train = pd.read_csv('train.txt',  sep=';', names=['sentence', 'emotion'], encoding='utf-8')
df_test = pd.read_csv('test.txt',sep=';', names= ['sentence', 'emotion'], encoding='utf-8')
df_val = pd.read_csv('va.txt',sep=';', names=['sentence', 'emotion'], encoding='utf-8')
tokenizer = get_tokenizer("basic_english")

def split_tokens(data_iter):
    tokens = []
    for text in data_iter:
        tokens.extend(tokenizer(text))
    return tokens

#tokenler tek bir listede toplandı
all_tokens = split_tokens(df_train['sentence'])
#metindeki kelime sözlükte yoksa onun yerine unknown atandi
vocab = build_vocab_from_iterator([all_tokens], specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

def text_to_vocab(x):
    return vocab(tokenizer(x))

def sequence_padding(batch):
    tensor_batch = []
    for item in batch:
        tensor_batch.append(torch.tensor(text_to_vocab(item), dtype=torch.long))
    return pad_sequence(tensor_batch, padding_value=vocab["<unk>"], batch_first=True)

X_train = []
for text in df_train['sentence']:
    X_train.append(text_to_vocab(text))

X_val = []
for text in df_val['sentence']:
    X_val.append(text_to_vocab(text))

X_test = []
for text in df_test['sentence']:
    X_test.append(text_to_vocab(text))


X_train_pad = sequence_padding(df_train['sentence'])
X_val_pad = sequence_padding(df_val['sentence'])
X_test_pad = sequence_padding(df_test['sentence'])

#her bir duygu için bir rakam belirlendi(kelimeler sayısal bir değer alacağı için)
label_dict = {'joy': 0, 'anger': 1, 'love': 2, 'sadness': 3, 'fear': 4, 'surprise': 5}
Y_train = df_train['emotion'].replace(label_dict).values
Y_val = df_val['emotion'].replace(label_dict).values
Y_test = df_test['emotion'].replace(label_dict).values

Y_train_f = torch.eye(6)[Y_train]
Y_val_f = torch.eye(6)[Y_val]
Y_test_f = torch.eye(6)[Y_test]
class EmotionDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

train_dataset = EmotionDataset(X_train_pad, Y_train_f)
val_dataset = EmotionDataset(X_val_pad, Y_val_f)
test_dataset = EmotionDataset(X_test_pad, Y_test_f)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class EmotionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout,bidirectional):
        nn.Module.__init__(self)
        
        # kelimeler vektörleştirildi
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)

            
        self.lstm = nn.LSTM(embedding_dim,hidden_dim,num_layers=n_layers,bidirectional= True,dropout=dropout,batch_first=True)
     
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text):
        embedded = self.embedding(text)
        
        embedded_dropout = self.dropout(embedded)
        
        # (hidden, cell) son hidden'ı ve son cell'i alır
        lstm_out, (hidden, cell) = self.lstm(embedded_dropout)
    
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        hidden_dropout = self.dropout(hidden)
        
        output = self.fc(hidden_dropout)
        
        return output

model = EmotionModel(vocab_size=len(vocab),embedding_dim=64,hidden_dim=80,output_dim=6,n_layers=2,dropout=0.6,bidirectional=True)

cross_entropy = nn.CrossEntropyLoss()
optimization = optim.Adam(model.parameters(), lr=0.001)
#sistem çok yavaş çalıştığı için hızkandırmak adına böyle bir teknik bulduk
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
cross_entropy = cross_entropy.to(device)



def train(model, iterator, optimization, cross_entropy):
    model.train()
    epoch_loss = 0
    for batch in iterator:
        optimization.zero_grad()
        text, labels = batch
        text = text.to(device)
        labels = labels.to(device)
        predictions = model(text)
        loss = cross_entropy(predictions, torch.max(labels, 1)[1])
        loss.backward()
        optimization.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, cross_entropy):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in iterator:
            text, labels = batch
            text = text.to(device)
            labels = labels.to(device)
            predictions = model(text)
            loss = cross_entropy(predictions, torch.max(labels, 1)[1])
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

n_epochs = 20
for epoch in range(n_epochs):
    train_loss = train(model, train_loader, optimization, cross_entropy)
    val_loss = evaluate(model, val_loader, cross_entropy)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}')
    
test_loss = evaluate(model, test_loader, cross_entropy)
print(f'Test Loss: {test_loss:.3f}')


def get_key(value):
    dictionary = {0: 'joy', 1: 'anger', 2: 'love', 3: 'sadness', 4: 'fear', 5: 'surprise'}
    return dictionary[value]

def predict(sentence, tokenizer, model, maxlen=80):
    model.eval()
    sentence_lst = [sentence]
    
    sentence_seq = []
    for sentence in sentence_lst:
        sentence_seq.append(text_to_vocab(sentence))

    sentence_padded = sequence_padding(sentence_lst)
    sentence_tensor = torch.tensor(sentence_padded, dtype=torch.long).to(device)
    with torch.no_grad():
        prediction = model(sentence_tensor)
        predicted_class = prediction.argmax(dim=1).item()
    return get_key(predicted_class)

print(predict("my exams passed very well but ı heard that my grandma died",tokenizer, model))

def chatbot():
    greeting_responses = ["Hello!", "Hi there!", "Hi!"]
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'q':
            print("Chatbot: Goodbye!")
            break
        elif user_input.lower() in ['hello', 'hi', 'hi there']:
            print("Chatbot: Hello! How are you feeling today?")
            continue
        else:
            predicted_emotion = predict(user_input, tokenizer, model)
            print(f"Chatbot: I think you're feeling {predicted_emotion}.")


chatbot()