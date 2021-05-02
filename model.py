from stellargraph.layer import GCN
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import binary_crossentropy
from config import HIDDEN,LR

def create_graph_classification_model(generator):
    gc_model = GCN(
        layer_sizes=[HIDDEN, HIDDEN],
        activations=["relu", "relu"],
        generator=generator,
        dropout=0.5
    )
    x_inp, x_out = gc_model.in_out_tensors()
    predictions = Dense(units=32, activation="relu")(x_out)
    predictions = Dense(units=2, activation="sigmoid")(predictions)
    

    # Let's create the Keras model and prepare it for training
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(optimizer=Adam(LR), loss=binary_crossentropy, metrics=["acc"])

    return model