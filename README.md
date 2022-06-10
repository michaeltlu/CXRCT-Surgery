# CXR-CTSurgery: Deep learning to predict mortality after cardiothoracic surgery using preoperative chest radiographs

![CXR-Age Grad-CAM](/images/CTSurg_Github.png)

[Raghu VK, Moonsamy P, Sundt TM, Ong CS, Singh S, Cheng A, Hou M, Denning L, Gleason TG, Aguirre AD, Lu MT. Deep learning to predict mortality after cardiothoracic surgery using preoperative chest radiographs. Ann Thorac Surg. 2022 May 21:S0003-4975(22)00722-6. doi: 10.1016/j.athoracsur.2022.04.056. Epub ahead of print. PMID: 35609650.](<https://pubmed.ncbi.nlm.nih.gov/35609650/>)

***THIS MODEL IS NOT CURRENTLY INTENDED FOR CLINICAL USE***
This repo contains data intended to promote reproducible research. It is not for clinical care or commercial use. 

## Overview
Decisions to proceed with cardiothoracic surgery require an accurate estimate of risk. Currently, surgery departments rely on the Society of Thoracic Surgeons Predicted Risk of Mortality (STS-PROM) score - a regression model that outputs the probability that a patient will survive a surgical procedure. The STS-PROM score is only applicable to common procedures (e.g., coronary artery bypass, valve surgery) and is cumbersome to use, requiring manual entry of over 60 inputs.

Chest x-rays (radiographs or CXRs) are routinely done prior to cardiothoracic surgery. We and others have shown that deep learning models can extract useful information from CXRs to predict mortality and chronic disease risk. Here, we hypothesized that a deep learning convolutional neural network (CNN) could extract information from these x-rays to estimate a patient's risk of postoperative mortality. We call this model CXR-CTSurgery.

CXR-CTSurgery takes a single chest x-ray image as input and outputs a probability between 0 and 1 of postoperative mortality risk. CXR-CTSurgery was developed in two steps using transfer learning from our previous CXR-Risk model (https://github.com/michaeltlu/cxr-risk). First, it was fine tuned to estimate any postoperative adverse event using images and follow-up data from 9,283 patients from Mass General Hospital. Then, the model was fine-tuned to estimate mortality risk alone. 

CXR-CTSurgery was tested in an independent cohort of 3,615 patients undergoing surgery at Mass General, and externally tested in 2,840 patients undergoing surgery at Brigham and Women's Hospital. CXR-CTSurgery predicted post-operative mortality with discrimination performance (Area under the ROC curve or AUROC) nearing the STS-PROM score (0.83 vs 0.88 in MGH and 0.74 vs 0.80 in BWH). CXR-CTSurgery also had high performance in the 30% of patients undergoing procedures for which there is no STS-PROM score (MGH AUROC 0.87, BWH AUROC 0.73). In both datasets, CXR-CTSurgery had better calibration (as measured by Observed-Expected Ratio) in MGH (0.74 vs. 0.52) and BWH (0.91 vs. 0.73) testing cohorts.

Overall, we found that CXR-CTSurgery predicts postoperative mortality based on a CXR image with similar discrimination, but better calibration than STS-PROM. It can be applied to surgeries where the STS-PROM cannot be used.


## Installation
This inference code was tested on Ubuntu 18.04.3 LTS, conda version 4.8.0, python 3.7.7, fastai 1.0.61, cuda 10.2, pytorch 1.5.1 and cadene pretrained models 0.7.4. A full list of dependencies is listed in `environment.yml`. 

Inference can be run on the GPU or CPU, and should work with ~4GB of GPU or CPU RAM. For GPU inference, a CUDA 10 capable GPU is required.

For the model weights to download, Github's large file service must be downloaded and installed: https://git-lfs.github.com/ 

This example is best run in a conda environment:

```bash
git lfs clone https://github.com/vineet1992/CXRCT-Surgery/
cd location_of_repo
conda env create -n CXR -f environment.yml
conda activate CXR
python run_model.py dummy_datasets/test_images/ output/dummy_output.csv --surgery=dummy_datasets/Dummy_Dataset.csv
```

To generate saliency maps for each estimate, add "--saliency=path/to/output/saliency/maps". Next is a complete example of this command

```bash
python run_model.py dummy_datasets/test_images/ output/dummy_output.csv --surgery=dummy_datasets/Dummy_Dataset.csv --saliency=saliency_maps/
```
Dummy image files are provided in `dummy_datasets/test_images/;`. Weights for the CXR-CTSurgery model are in `development/models/STS_Mortality_Final_042420.pth`. 

## Image processing
Radiographs were converted from Native DICOM images from each institution to .png format using the following commands. We chose to first convert them to TIF using DCMTK v3.6.1, then to PNGs with a minimum dimension of 512 pixels through ImageMagick to maintain consistency with how the original CXR-Risk model was trained (https://github.com/michaeltlu/cxr-risk)

```bash
cd path/to/DICOM/images
for x in *.dcm; do dcmj2pnm -O +ot +G $x "${x%.dcm}".tif; done;
mogrify -path path/to/destination/PNGs/ -trim +repage -colorspace RGB -auto-level -depth 8 -resize 512x512^ -format png "*.tif"
```



## Acknowledgements
I thank the NCI and ACRIN for access to trial data that enabled us to develop this model and Massachusetts General and Brigham and Women's hospital for access to patient data. I would also like to thank the patients undergoing cardiac surgery at these institutions as well as the PLCO and NLST participants for their contribution to research. I would also like to thank the fastai and Pytorch communities. A GPU used for this research was donated as an unrestricted gift through the Nvidia Corporation Academic Program. The statements contained herein are mine alone and do not represent or imply concurrence or endorsements by the above individuals or organizations.


