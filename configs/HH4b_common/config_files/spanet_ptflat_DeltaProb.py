from configs.HH4b_common.dnn_input_variables import (
    bkg_morphing_dnn_DeltaProb_input_variables,
    sig_bkg_dnn_DeltaProb_input_variables,
)

onnx_model_dict = {
    "spanet": "",
    "vbf_ggf_dnn": "",
    "bkg_morphing_dnn": "",
    "sig_bkg_dnn": "",
    "bkg_morphing_spread_dnn": "",
}


onnx_model_dict  |= {
    "spanet": "/work/tharte/datasets/mass_sculpting_data/hh4b_5jets_e300_s100_ptvary_wide_loose_btag.onnx", # spanet pt vary 0.3, 1.7
    "bkg_morphing_dnn": "/work/mmalucch/out_ML_pytorch/DNN_AN_1e-3_e20drop75_minDelta1em5_SPANet_postEE_DeltaProb/best_models/average_model_from_onnx.onnx",  # DeltaProb, only 2022_postEE, 20 k-folds, early stopping, 1e-5 minDelta, spanet pt vary 
    "sig_bkg_dnn": "/work/mmalucch/out_ML_pytorch/DNN_ptFlatSPANet_class_weights_e5drop75_postEE_allklambda_DeltaProbabilityMorphing/state_dict/model_best_epoch_13.onnx", # DeltaProb
}

config_options_dict = {
    "higgs_parton_matching": False,
    "vbf_parton_matching": False,
    "tight_cuts": False,
    "classification": False,
    "save_chunk": False,
    "vbf_presel": False,
    "semi_tight_vbf": True,
    "dnn_variables": True,
    "run2": False,
    "vr1": False,
    "random_pt": False,
    "rand_type": 0.3,
    "blind": True if onnx_model_dict["sig_bkg_dnn"] else False,
    "sig_bkg_dnn_input_variables": sig_bkg_dnn_DeltaProb_input_variables,
    "bkg_morphing_dnn_input_variables": bkg_morphing_dnn_DeltaProb_input_variables,
    "parton_jet_min_dR": 0.4,
    "max_num_jets": 5,
    "which_bquark": "last",
    "fifth_jet": "pt",
    "donotscale_sumgenweights": True,
    "pad_value": -999.0,
    "arctanh_delta_prob_bin_edge": 2.44,
    "arctanh_delta_prob_pad_limit": 2.,
    "add_jet_spanet": False,
} | onnx_model_dict
