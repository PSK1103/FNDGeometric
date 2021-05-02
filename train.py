from model import create_graph_classification_model
from get_data import createGraph
from config import EPOCHS, BATCH_SIZE
from stellargraph.mapper import FullBatchNodeGenerator
from sklearn import model_selection,preprocessing
import stellargraph as sg
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping
import pickle

def get_generators(generator,train_index, test_index, graph_labels, batch_size):
    train_gen = generator.flow(
        train_index, targets=graph_labels.iloc[train_index].values, batch_size=batch_size
    )
    test_gen = generator.flow(
        test_index, targets=graph_labels.iloc[test_index].values, batch_size=batch_size
    )

    return train_gen, test_gen

    
    
if __name__ == '__main__':
    graph,labels = createGraph()
    train_samples, test_samples = model_selection.train_test_split(
        labels, train_size=0.7, test_size=None, stratify=labels
    )
    val_samples,test_samples = model_selection.train_test_split(
        test_samples, train_size=0.33, test_size=None, stratify=test_samples
    )
    # val_samples, test_samples = model_selection.train_test_split(
    #     test_samples, train_size=0.5, test_size=None, stratify=test_samples
    # )

    generator = FullBatchNodeGenerator(graph, method="gcn")

    target_encoding = preprocessing.LabelBinarizer()

    train_targets = target_encoding.fit_transform(train_samples)
    val_targets = target_encoding.transform(val_samples)
    test_targets = target_encoding.transform(test_samples)

    train_gen = generator.flow(train_samples.index, train_targets)

    val_gen = generator.flow(val_samples.index, val_targets)

    model = create_graph_classification_model(generator)

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        verbose=2,
        shuffle=False,
        batch_size=BATCH_SIZE,
        validation_batch_size=BATCH_SIZE
    )

    with open('history.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()