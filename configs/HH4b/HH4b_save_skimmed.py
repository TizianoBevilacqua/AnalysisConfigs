import os

from pocket_coffea.lib.cut_functions import (
    get_HLTsel,
)

# from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters import defaults
from pocket_coffea.utils.configurator import Configurator
from workflow_dummy import HH4bbQuarkMatchingProcessorDummy

localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters

CLASSIFICATION = False
TIGHT_CUTS = False
RANDOM_PT = False
SAVE_CHUNK = False

print("CLASSIFICATION ", CLASSIFICATION)
print("TIGHT_CUTS ", TIGHT_CUTS)
print("RANDOM_PT ", RANDOM_PT)

default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir + "/params")

# adding object preselection
year = ["2022_postEE", "2022_preEE", "2023_preBPix", "2023_postBPix"]
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/params/object_preselection.yaml",
    f"{localdir}/params/triggers.yaml",
    f"{localdir}/params/jets_calibration_withVariations.yaml",
    # f"{localdir}/params/plotting_style.yaml",
    update=True,
)

cfg = Configurator(
    # save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/ntuples/DATA_JetMET_JMENano_skimmed",
#    save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/ntuples/DATA_JetMET_JMENano_F_skimmed",
    # save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/ntuples/DATA_JetMET_ParkingHH_2023_D_skimmed",
    save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/testing/DATA_JetMET_JMENano_C_skimmed",
    parameters=parameters,
    datasets={
        "jsons": [
            f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b.json",
            f"{localdir}/../HH4b_common/datasets/signal_ggF_HH4b_spanet_redirector.json",
            f"{localdir}/../HH4b_common/datasets/DATA_JetMET_skimmed.json",
            f"{localdir}/../HH4b_common/datasets/QCD.json",
            f"{localdir}/../HH4b_common/datasets/DATA_ParkingHH.json",
            f"{localdir}/../HH4b_common/datasets/DATA_JetMET_redirector.json",
        ],
        "filter": {
            "samples": (
                [
                    "DATA_JetMET_JMENano_C",
                    # "DATA_JetMET_JMENano_D",
                    # "DATA_JetMET_JMENano_E",
                    # "DATA_JetMET_JMENano_F",
                    # "DATA_JetMET_JMENano_G",
                    # "DATA_JetMET_JMENano_2023_Cv1",
                    # "DATA_JetMET_JMENano_2023_Cv2",
                    # "DATA_ParkingHH_2023_Cv3",
                    # "DATA_ParkingHH_2023_Cv4",
                    # "DATA_ParkingHH_2023_Dv1",
                    # "DATA_ParkingHH_2023_Dv2",
                    # "GluGlutoHHto4B_spanet"
                ]
            ),
            "samples_exclude": [],
            "year": year,
        },
        "subsamples": {},
    },
    workflow=HH4bbQuarkMatchingProcessorDummy,
    workflow_options={
    },
    skim=[
        get_HLTsel(primaryDatasets=["JetMET"]),
        # get_HLTsel(primaryDatasets=["ParkingHH"]),
    ],
    preselections=[
        #
    ],
    categories={
        #
    },
    weights={
        "common": {
            "inclusive": [
                "genWeight",
                "lumi",
                "XS",
            ],
            "bycategory": {},
        },
        "bysample": {},
    },
    variations={
        "weights": {
            "common": {
                "inclusive": [],
                "bycategory": {},
            },
            "bysample": {},
        }
    },
    variables={
        #
    },
    columns={
        #
    },
)
