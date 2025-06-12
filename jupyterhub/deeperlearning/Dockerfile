ARG OWNER=brineylab
ARG BASE_IMG=$OWNER/jupyterhub-deeplearning
FROM $BASE_IMG
LABEL maintainer='Briney Lab @ Scripps Research <briney@scripps.edu>'

# In alignment with the licensing of the orignal image created
# by the Jupyter Development Team, this image is distributed 
# under the terms of the Modified BSD License.
# https://en.wikipedia.org/wiki/BSD_licenses#3-clause_license

#------------------------
#    Models Directory
#------------------------
ENV MODELS_DIR="/models"
RUN mkdir "${MODELS_DIR}" \
    && chown "${NB_USER}" "${MODELS_DIR}" \
    && fix-permissions "${MODELS_DIR}"


#------------------------
#       ColabDesign
#------------------------
# aria2c
RUN apt-get update \
    && apt-get install --yes aria2 \
    && apt-get clean \
    && sudo rm -rf /var/lib/apt/lists/*

USER ${NB_UID}

# RFdiffusion
RUN cd "${NGS_TOOLS_DIR}" \
    && git clone "https://github.com/RosettaCommons/RFdiffusion.git" \
    && cd RFdiffusion \
    # download model weights
    && mkdir models \
    && cd models \
    && echo "downloading RFdiffusion model weights:" \
    && echo "  - Base_ckpt.pt" \
    && aria2c --max-connection-per-server=8 http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt \
    && echo "  - Complex_base_ckpt.pt" \
    && aria2c --max-connection-per-server=8 http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt \
    && echo "  - Complex_Fold_base_ckpt.pt" \
    && aria2c --max-connection-per-server=8 http://files.ipd.uw.edu/pub/RFdiffusion/60f09a193fb5e5ccdc4980417708dbab/Complex_Fold_base_ckpt.pt \
    && echo "  - InpaintSeq_ckpt.pt" \
    && aria2c --max-connection-per-server=8 http://files.ipd.uw.edu/pub/RFdiffusion/74f51cfb8b440f50d70878e05361d8f0/InpaintSeq_ckpt.pt \
    && echo "  - InpaintSeq_Fold_ckpt.pt" \
    && aria2c --max-connection-per-server=8 http://files.ipd.uw.edu/pub/RFdiffusion/76d00716416567174cdb7ca96e208296/InpaintSeq_Fold_ckpt.pt \
    && echo "  - ActiveSite_ckpt.pt" \
    && aria2c --max-connection-per-server=8 http://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt \
    && echo "  - Base_epoch8_ckpt.pt" \
    && aria2c --max-connection-per-server=8 http://files.ipd.uw.edu/pub/RFdiffusion/12fc204edeae5b57713c5ad7dcb97d39/Base_epoch8_ckpt.pt \
    && echo "  - Complex_beta_ckpt.pt" \
    && aria2c --max-connection-per-server=8 http://files.ipd.uw.edu/pub/RFdiffusion/f572d396fae9206628714fb2ce00f72e/Complex_beta_ckpt.pt \
    && echo "  - RF_structure_prediction_weights.pt" \
    && aria2c --max-connection-per-server=8 http://files.ipd.uw.edu/pub/RFdiffusion/1befcb9b28e2f778f53d47f18b7597fa/RF_structure_prediction_weights.pt \
    # dependencies
    && python3 -m pip install jedi omegaconf hydra-core icecream pyrsistent pynvml decorator \
    && python3 -m pip install "git+https://github.com/NVIDIA/dllogger#egg=dllogger" \
    && python3 -m pip install --no-dependencies dgl==2.0.0 -f "https://data.dgl.ai/wheels/cu121/repo.html" \
    && python3 -m pip install --no-dependencies e3nn==0.3.3 opt_einsum_fx \
    # install SE(3)-Transformers
    && cd ../env/SE3Transformer \ 
    && python3 -m pip install --no-cache-dir -r requirements.txt \
    && python3 -m pip install -e . \
    # install RFdiffusion
    && cd ../.. \
    && python3 -m pip install -e . \
    && fix-permissions "${NGS_TOOLS_DIR}/RFdiffusion"
ENV DGLBACKEND="pytorch"

# AnAnaS -> https://team.inria.fr/nano-d/software/ananas/
RUN cd "${NGS_TOOLS_DIR}" \
    && wget -qnc "https://files.ipd.uw.edu/krypton/ananas" \
    && sudo chmod +x ananas \
    && sudo cp ananas /usr/local/bin/

# AlphaFold weights (for ColabDesign)
ENV ALPHAFOLD2_PARAMS_DIR="/models/alphafold2/params"
RUN mkdir -p "${ALPHAFOLD2_PARAMS_DIR}" \
    && chown "${NB_USER}" "${ALPHAFOLD2_PARAMS_DIR}" \
    && fix-permissions "${ALPHAFOLD2_PARAMS_DIR}"
RUN cd "${ALPHAFOLD2_PARAMS_DIR}" \
    && aria2c -q -x 16 "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar" \
    && tar -xvf "alphafold_params_2022-12-06.tar" \
    && rm "alphafold_params_2022-12-06.tar" \
    && fix-permissions "${ALPHAFOLD2_PARAMS_DIR}"

# install ColabDesign
RUN cd "${NGS_TOOLS_DIR}" \
    && git clone "https://github.com/sokrypton/ColabDesign" \
    && cd ColabDesign \
    && python3 -m pip install . \
    && fix-permissions "${NGS_TOOLS_DIR}/ColabDesign"

# # Rosetta (and PyRosetta)
# ENV CXX=/usr/bin/g++
# ENV CC=/usr/bin/gcc
# RUN cd "${NGS_TOOLS_DIR}" \
#     && git clone "https://github.com/RosettaCommons/Rosetta.git" \
#     && cd Rosetta/source/ \
#     && ./scons.py -j12 mode=release bin \
#     && cd src/python/PyRosetta \
#     && python3 build.py -j12 \
#     && fix-permissions "${NGS_TOOLS_DIR}/Rosetta"


#-------------------------
#    Ab/protein models
#-------------------------
RUN python3 -m pip install \
        # protein models
        boltz \
        fair-esm \
        # antibody models
        ablang2 \
        ImmuneBuilder \
        "git+https://github.com/OliverT1/p-IgGen.git" \
    && python3 -m pip cache purge

# RUN cd "${NGS_TOOLS_DIR}" \
#     && git clone "https://github.com/oxpig/AntiFold" \
#     && cd AntiFold \
#     && python3 -m pip install . \
#     && fix-permissions "${NGS_TOOLS_DIR}/AntiFold"
