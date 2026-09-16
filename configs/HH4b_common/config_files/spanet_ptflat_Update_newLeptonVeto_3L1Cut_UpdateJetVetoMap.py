import configs.HH4b_common.dnn_input_variables as dnn_vars
import configs.HH4b_common.output_variables as out_vars


from configs.HH4b_common.config_files.default_config import default_onnx_model_dict as onnx_model_dict

from configs.HH4b_common.config_files.default_config import default_config_options_dict as config_options_dict


onnx_model_dict |= {
    # "spanet": "/work/tharte/datasets/onnx_spanet_models_for_pairing_and_mass_sculpting_studies/spanet_1_14_5_h4b_5jets_ptvary_loose_300_btag_wp_newLeptonVeto_3L1Cut_UpdateJetVetoMap.onnx",
    "spanet": "/work/bevila_t/PostDoc/HH4b/Output/SPANet/out_hh4b_pairing_higgs_vbf_processor_ptFlatten_4b_region_all_Klambda_normalized_to_sample_filtered_abs_neg_w_ggF_ZH_ZZ_2023/version_0/out_hh4b_pairing_higgs_vbf_processor_ptFlatten_4b_region_all_Klambda_normalized_to_sample_filtered_abs_neg_w_ggF_ZH_ZZ_2023.onnx",
    "bkg_morphing_dnn": "/work/bevila_t/PostDoc/HH4b/Output/ML_trainings/bkg_reweigting_local/SPANet_5WP_2023_postBPix_ZZ_ZH_HH_260811/best_models/average_model_from_onnx.onnx",
    # "sig_bkg_dnn": "/pnfs/psi.ch/cms/trivcat/store/user/tharte/datasets/ML_pytorch/out/sig_bkg_classifier/DNN_AN_1e-3_e20drop75_minDelta1em5_SPANet_newUpdates_newLeptonVeto_3L1Cut_UpdateJetVetoMap_postEE/run100/state_dict/model_best_epoch_31.onnx",
}


config_options_dict |= {
    "dnn_variables": True,
    "run2": False,
    "sig_bkg_dnn_input_variables": dnn_vars.sig_bkg_dnn_input_variables,
    "bkg_morphing_dnn_input_variables": dnn_vars.bkg_morphing_dnn_input_variables,
    "output_sig_bkg_dnn_input_variables": out_vars.output_sig_bkg_dnn_input_variables,
    "output_bkg_morphing_dnn_input_variables": out_vars.output_bkg_morphing_dnn_input_variables,
    "save_spanet_input_variables": True,
    "fifth_jet": "pt",
    "pad_value": -999.0,
    "add_jet_spanet": True,
    "spanet_input_name": dnn_vars.pairing_spanet_btagWP5,
    "max_num_jets_spanet_class": 5,
    "max_num_jets_good": 4,
    "which_bquark": "last",
    "qt_postEE": None,
    "random_pt": False,
    "rand_type": 0.3,
    # VBF
    "vbf_parton_matching": True,
    "vbf_presel": False,
    "vbf_analysis": True,
    "which_vbf_quark":"with_mothers_children",
    "max_num_jets_add_vbf": 3,
    "jets_add_vbf_order": "pt",
    # "mixeddata": True,
    # "save_chunk": "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/spanet_ptflat_Update_newLeptonVeto_3L1Cut_UpdateJetVetoMap",
    # "qt_postEE": "/work/tharte/datasets/quantile_transformer/1_19_1_3L1_trigger_cuts/qt_events_sig_bkg_dnn_score_kl_1.00.pkl",
    # "qt_postEE": "/pnfs/psi.ch/cms/trivcat/store/user/tharte/datasets/sig_bkg_classifier/1_19_4_spanet_ptflat_Update_newLeptonVeto_3L1Cut_UpdateJetVetoMap_forPhDseminar_with_systs_qt_transform_fixed/quantile_transformer_directfromscore/SRSpanet_qt/qt_events_sig_bkg_dnn_score_kl_1.00.pkl",
}| onnx_model_dict
