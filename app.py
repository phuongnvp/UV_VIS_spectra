# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

import torch
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
from PIL import Image


# -----------------------------
# Configuration
# -----------------------------
MODEL_DIR = "./"
N_TASKS = 481


@st.cache_resource
def load_model():
    model = dc.models.AttentiveFPModel(
        n_tasks=N_TASKS,
        mode="regression",
        num_layers=3,
        num_timesteps=2,
        graph_feat_size=512,
        dense_layer_size=2048,
        dropout=0.1,
        batch_size=32,
        learning_rate=1e-3,
    )
    model.restore(model_dir=MODEL_DIR)
    return model


@st.cache_resource
def load_featurizer():
    return dc.feat.MolGraphConvFeaturizer(use_edges=True)

def predict_spectrum(model, featurizer, smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")

    features = featurizer.featurize([smiles])
    dataset = dc.data.NumpyDataset(X=features)
    pred = model.predict(dataset)[0].astype(float).flatten()

    x = np.linspace(220, 700, len(pred))

    y_smooth = savgol_filter(pred, window_length=20, polyorder=2)
    return x, y_smooth, mol

def make_plot(x, y, mol, smiles):
    mol_img = Draw.MolToImage(mol, size=(300, 300))

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x, y, linewidth=2)
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Intensity")
    ax.set_title(f"Predicted UV-Vis Spectrum")

    img_arr = np.array(mol_img)
    ax_inset = fig.add_axes([0.68, 0.58, 0.2, 0.2])
    ax_inset.imshow(img_arr)
    ax_inset.axis("off")

    plt.tight_layout()
    return fig


def extract_atom_weights(dc_model, smiles, self_loop=True):
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    graph_data = featurizer.featurize([smiles])[0]

    if graph_data is None:
        raise ValueError(f"Could not featurize SMILES: {smiles}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")

    dgl_graph = graph_data.to_dgl_graph(self_loop=self_loop)
    device = dc_model.device
    dgl_graph = dgl_graph.to(device)

    core_model = dc_model.model
    dgl_predictor = core_model.model

    node_feats = dgl_graph.ndata[core_model.nfeat_name]
    edge_feats = dgl_graph.edata[core_model.efeat_name]

    core_model.eval()
    with torch.no_grad():
        _, node_weights = dgl_predictor(
            dgl_graph,
            node_feats,
            edge_feats,
            get_node_weight=True
        )

    atom_weights_per_timestep = [
        w.squeeze(-1).detach().cpu().numpy() for w in node_weights
    ]
    atom_weights_mean = np.mean(atom_weights_per_timestep, axis=0)

    atom_weights_mean = np.asarray(atom_weights_mean, dtype=float)
    if np.allclose(atom_weights_mean.max(), atom_weights_mean.min()):
        atom_weights_norm = np.zeros_like(atom_weights_mean)
    else:
        atom_weights_norm = (
            (atom_weights_mean - atom_weights_mean.min()) /
            (atom_weights_mean.max() - atom_weights_mean.min())
        )

    return mol, atom_weights_norm


def make_similarity_map_image(mol, atom_weights, size=(700, 500)):
    drawer = Draw.MolDraw2DCairo(size[0], size[1])
    SimilarityMaps.GetSimilarityMapFromWeights(
        mol,
        list(atom_weights),
        draw2d=drawer,
    )
    drawer.FinishDrawing()
    png_bytes = drawer.GetDrawingText()
    return Image.open(io.BytesIO(png_bytes))

def pil_image_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def make_csv(x, y, smiles):
    df = pd.DataFrame({
        "Wavelength": x,
        "Intensity": y,
        #"Smiles": [smiles] * len(x)
    })
    return df.to_csv(index=False).encode("utf-8")



# Streamlit UI
def main():
    st.set_page_config(page_title="UV-Vis Spectra Predictor", layout="wide")
    st.title("UV-Vis Spectra Predictor")

    model = load_model()
    featurizer = load_featurizer()

    smiles = st.text_input("Enter SMILES", value="")

    col1, col2, col3 = st.columns(3)
    plot_clicked = col1.button("Plot Graph")
    csv_clicked = col2.button("Export CSV")
    simmap_clicked = col3.button("Export Attention Weight")

    if smiles.strip():
        try:
            x, y, mol = predict_spectrum(model, featurizer, smiles)

            if plot_clicked:
                fig = make_plot(x, y, mol, smiles)
                st.pyplot(fig)

            if csv_clicked:
                csv_bytes = make_csv(x, y, smiles)
                st.download_button(
                    label="Download CSV",
                    data=csv_bytes,
                    file_name="predicted_spectrum.csv",
                    mime="text/csv",
                )

            if simmap_clicked:
                sim_mol, atom_weights = extract_atom_weights(model, smiles)
                sim_img = make_similarity_map_image(sim_mol, atom_weights)

                st.image(sim_img, caption="Attention Weight", use_container_width=True)

                img_bytes = pil_image_to_bytes(sim_img)
                st.download_button(
                    label="Download Attention Weight",
                    data=img_bytes,
                    file_name="attention_way.png",
                    mime="image/png",
                )

        except Exception as e:
            st.error(str(e))
    else:
        st.info("Please enter a SMILES string.")


if __name__ == "__main__":
    main()