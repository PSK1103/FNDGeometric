from model import create_graph_classification_model
from get_data import createUserDB,createGraph
from config import EPOCHS, BATCH_SIZE
from stellargraph.mapper import FullBatchNodeGenerator
from sklearn import model_selection,preprocessing
import stellargraph as sg

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
    val_samples, test_samples = model_selection.train_test_split(
        test_samples, train_size=0.33, test_size=None, stratify=test_samples
    )

    generator = FullBatchNodeGenerator(graph, method="gcn")

    target_encoding = preprocessing.LabelBinarizer()

    train_targets = target_encoding.fit_transform(train_samples)
    val_targets = target_encoding.transform(val_samples)
    test_targets = target_encoding.transform(test_samples)

    train_gen = generator.flow(train_samples.index, train_targets)

    val_gen = generator.flow(val_samples.index, val_targets)

    es_callback = EarlyStopping(monitor="val_acc", patience=50, restore_best_weights=True)
    model = create_graph_classification_model(generator)

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        verbose=2,
        shuffle=True,
        callbacks=[es_callback],
        batch_size=BATCH_SIZE
    )

    with open('history.pkl','w') as fp:
        pickle.dump(history.history,fp)

    # train_graphs, test_graphs = model_selection.train_test_split(
    # graphs, train_size=0.8, test_size=None, stratify=graph_labels)

    # train_gen = generator.flow(
    #     list(train_graphs.index - 1),
    #     targets=train_graphs.values,
    #     batch_size=50,
    #     symmetric_normalization=False,
    # )

    # test_gen = generator.flow(
    #     list(test_graphs.index - 1),
    #     targets=test_graphs.values,
    #     batch_size=1,
    #     symmetric_normalization=False,
    # )

    # print('here')

    # history = model.fit(train_gen,epochs=EPOCHS,verbose=1,validation_data=test_gen, batch_size=BATCH_SIZE)

    sg.utils.plot_history(history)