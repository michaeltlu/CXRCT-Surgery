# CXR-CTSurgery: Deep learning to predict mortality after cardiothoracic surgery using preoperative chest radiographs

![CXR-Age Grad-CAM](/images/GradCAM_Github_020121.png)

[Raghu VK, Moonsamy P, Sundt TM, Ong CS, Singh S, Cheng A, Hou M, Denning L, Gleason TG, Aguirre AD, Lu MT. Deep learning to predict mortality after cardiothoracic surgery using preoperative chest radiographs. Ann Thorac Surg. 2022 May 21:S0003-4975(22)00722-6. doi: 10.1016/j.athoracsur.2022.04.056. Epub ahead of print. PMID: 35609650.](<https://pubmed.ncbi.nlm.nih.gov/35609650/>)



## Overview
Decisions to proceed with cardiothoracic surgery require an accurate estimate of risk. Currently, surgery departments rely on the Society of Thoracic Surgeons Predicted Risk of Mortality (STS-PROM) score - a regression model that outputs the probability that a patient will survive a surgical procedure. The STS-PROM score is only applicable to common procedures (e.g., coronary artery bypass, valve surgery) and is cumbersome to use, requiring manual entry of over 60 inputs.

Chest x-rays (radiographs or CXRs) are routinely done prior to cardiothoracic surgery. We and others have shown that deep learning models can extract useful information from CXRs to predict mortality and chronic disease risk. Here, we hypothesized that a deep learning convolutional neural network (CNN) could extract information from these x-rays to estimate a patient's risk of postoperative mortality. We call this model CXR-CTSurgery.

CXR-CTSurgery takes a single chest x-ray image as input and outputs a probability between 0 and 1 of postoperative mortality risk. CXR-CTSurgery was developed in two steps using transfer learning from our previous CXR-Risk model (https://github.com/michaeltlu/cxr-risk). First, it was fine tuned to estimate any postoperative adverse event using images and follow-up data from 9,283 patients from Mass General Hospital. Then, the model was fine-tuned to estimate mortality risk alone. 

CXR-CTSurgery was tested in an independent cohort of 3,615 patients undergoing surgery at Mass General, and externally tested in 2,840 patients undergoing surgery at Brigham and Women's Hospital. CXR-CTSurgery predicted post-operative mortality with discrimination performance (Area under the ROC curve or AUROC) nearing the STS-PROM score (0.83 vs 0.88 in MGH and 0.74 vs 0.80 in BWH). CXR-CTSurgery also had high performance in the 30% of patients undergoing procedures for which there is no STS-PROM score (MGH AUROC 0.87, BWH AUROC 0.73). In both datasets, CXR-CTSurgery had better calibration (as measured by Observed-Expected Ratio) in MGH (0.74 vs. 0.52) and BWH (0.91 vs. 0.73) testing cohorts.

Overall, we found that CXR-CTSurgery predicts postoperative mortality based on a CXR image with similar discrimination, but better calibration than STS-PROM. It can be applied to surgeries where the STS-PROM cannot be used.

**Central Illustration of CXR-Age**
![CXR-Age Central Illustration](/images/Central_Illustration.png)

This repo contains data intended to promote reproducible research. It is not for clinical care or commercial use. 

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
python run_model.py dummy_datasets/test_images/ development/models/ output/output.csv --modelarch=inceptionv4 --type=discrete --size=224
```

To generate saliency maps for each estimate, add "--saliency=path/to/output/saliency/maps". Next is a complete example of this command

```bash
python run_model.py dummy_datasets/test_images/ development/models/PLCO_Fine_Tuned_120419 output/output.csv --modelarch=age --type=continuous --size=224 --saliency=saliency_maps
```
Dummy image files are provided in `dummy_datasets/test_images/;`. Weights for the CXR-Age model are in `development/models/PLCO_Fine_Tuned_120419.pth`. 

## Datasets
PLCO (NCT00047385) data used for model development and testing are available from the National Cancer Institute (NCI, https://biometry.nci.nih.gov/cdas/plco/). NLST (NCT01696968) testing data is available from the NCI (https://biometry.nci.nih.gov/cdas/nlst/) and the American College of Radiology Imaging Network (ACRIN, https://www.acrin.org/acrin-nlstbiorepository.aspx). Due to the terms of our data use agreement, we cannot distribute the original data. Please instead obtain the data directly from the NCI and ACRIN.

The `data` folder provides the image filenames and the CXR-Age estimates. "File" refers to image filenames and "CXR-Age" refers to the CXR-Age estimate: 
* `PLCO_Age_Estimates.csv` contains the CXR-Age estimates in the PLCO testing dataset.
* `NLST_Age_Estimates.csv` contains the CXR-Age estimate in the NLST testing dataset. The format for "File" is (original participant directory)_(original DCM filename).png


## Image processing
PLCO radiographs were provided as scanned TIF files by the NCI. TIFs were converted to PNGs with a minimum dimension of 512 pixels with ImageMagick v6.8.9-9. 

Many of the PLCO radiographs were rotated 90 or more degrees. To address this, we developed a CNN to identify rotated radiographs. First, we trained a CNN using the resnet34 architecture to identify synthetically rotated radiographs from the [CXR14 dataset](http://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf). We then fine tuned this CNN using 11,000 manually reviewed PLCO radiographs. The rotated radiographs identified by this CNN in `preprocessing/plco_rotation_github.csv` were then corrected using ImageMagick. 

```bash
cd path_for_PLCO_tifs
mogrify -path destination_for_PLCO_pngs -trim +repage -colorspace RGB -auto-level -depth 8 -resize 512x512^ -format png "*.tif"
cd path_for_PLCO_pngs
while IFS=, read -ra cols; do mogrify -rotate 90 "${cols[0]}"; done < /path_to_repo/preprocessing/plco_rotation_github.csv
```

NLST radiographs were provided as DCM files by ACRIN. We chose to first convert them to TIF using DCMTK v3.6.1, then to PNGs with a minimum dimension of 512 pixels through ImageMagick to maintain consistency with the PLCO radiographs:

```bash
cd path_to_NLST_dcm
for x in *.dcm; do dcmj2pnm -O +ot +G $x "${x%.dcm}".tif; done;
mogrify -path destination_for_NLST_pngs -trim +repage -colorspace RGB -auto-level -depth 8 -resize 512x512^ -format png "*.tif"
```


The orientation of several NLST chest radiographs was manually corrected:

```
cd destination_for_NLST_pngs
mogrify -rotate "90" -flop 204025_CR_2000-01-01_135015_CHEST_CHEST_n1__00000_1.3.51.5146.1829.20030903.1123713.1.png
mogrify -rotate "-90" 208201_CR_2000-01-01_163352_CHEST_CHEST_n1__00000_2.16.840.1.113786.1.306662666.44.51.9597.png
mogrify -flip -flop 208704_CR_2000-01-01_133331_CHEST_CHEST_n1__00000_1.3.51.5146.1829.20030718.1122210.1.png
mogrify -rotate "-90" 215085_CR_2000-01-01_112945_CHEST_CHEST_n1__00000_1.3.51.5146.1829.20030605.1101942.1.png
```

## Acknowledgements
I thank the NCI and ACRIN for access to trial data, as well as the PLCO and NLST participants for their contribution to research. I would also like to thank the fastai and Pytorch communities as well as the National Academy of Medicine for their support of this work. A GPU used for this research was donated as an unrestricted gift through the Nvidia Corporation Academic Program. The statements contained herein are mine alone and do not represent or imply concurrence or endorsements by the above individuals or organizations.


