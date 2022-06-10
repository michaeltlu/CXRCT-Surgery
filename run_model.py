"""Script to run the CXR-CTSurgery model, optionally to produce saliency maps for the model

Surgery is a .csv file with two columns

Column 1 is the image file of interest

Column 2 is the surgery type


type corresponds to

0 - Non-STS Procedure
1 - Isolated Coronary Artery Bypass Grafting
2- Isolated Aortic Valve Replacement
3- Isolated Mitral Valve Replacement
4- Aortic Valve Replacement + Coronary Artery Bypass Grafting
5- Mitral Valve Replacement + Coronary Artery Bypass Grafting
6 - Mitral Valve Repair
7 - Mitral Valve Repair + Coronary Artery Bypass Grafting


Usage:
  run_model.py <image_dir> <output_file> [--saliency=SAL_DIR] [--surgery=SURGERY]
  run_model.py (-h | --help)
Examples:
  run_model.py /path/to/images /path/to/model /path/to/write/output.csv
Options:
  -h --help                    Show this screen.
  --saliency=SAL_DIR           Directory to write saliency maps to [Default:None]
  --surgery=SURGERY            What type of surgery is being performed? [Default: 1]
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from docopt import docopt
import pandas as pd
import fastai
from fastai.vision import *
import pretrainedmodels
from sklearn.metrics import *
from fastai.callbacks import *
import math
import time
import GradCAMUtils
from PIL import Image
from torchvision import transforms


###TODO Add optional checkpointing (optional result file to append to, skipping loop iteration if model exists)
tfms_test = get_transforms(do_flip = False,max_warp = None)



    
def _tta_only(learn:Learner, ds_type:DatasetType=DatasetType.Valid, activ:nn.Module=None, scale:float=1.35) -> Iterator[List[Tensor]]:
    "Computes the outputs for several augmented inputs for TTA"
    dl = learn.dl(ds_type)
    ds = dl.dataset
    old = ds.tfms
    #activ = ifnone(activ, _loss_func2activ(learn.loss_func))
    augm_tfm = [o for o in learn.data.train_ds.tfms if o.tfm not in
               (crop_pad, flip_lr, dihedral, zoom)]
    try:
        pbar = master_bar(range(8))
        for i in pbar:
            row = 1 if i&1 else 0
            col = 1 if i&2 else 0
            #flip = i&4
            d = {'row_pct':row, 'col_pct':col, 'is_random':False}
            tfm = [*augm_tfm, zoom(scale=scale, **d), crop_pad(**d)]
            #if flip: tfm.append(flip_lr(p=1.))
            #import pdb; pdb.set_trace()
            ds.tfms = tfm
            yield get_preds(learn.model, dl, pbar=pbar, activ=activ)[0]
    finally: ds.tfms = old


def _TTA(learn:Learner, beta:float=0.4, scale:float=1.35, ds_type:DatasetType=DatasetType.Valid, activ:nn.Module=None, with_loss:bool=False) -> Tensors:
    "Applies TTA to predict on `ds_type` dataset."
    preds,y = learn.get_preds(ds_type, activ=activ)
    all_preds = list(_tta_only(learn,ds_type=ds_type, activ=activ, scale=scale))
    avg_preds = torch.stack(all_preds).mean(0)
    sd_preds = torch.stack(all_preds).std(0)
    if beta is None: return preds,avg_preds,y,sd_preds
    else:
        final_preds = preds*beta + avg_preds*(1-beta)
        if with_loss:
            with NoneReduceOnCPU(learn.loss_func) as lf: loss = lf(final_preds, y)
            return final_preds, y, loss,sd_preds
        return final_preds, y,sd_preds

num_workers = 16
bs = 32
if __name__ == '__main__':

    arguments = docopt(__doc__)
    ###Grab image directory
    image_dir = arguments['<image_dir>']


    one = False
    
    #Location of the model
    mdl_path = "development/models/STS_Mortality_Final_042420"

    #Image size for model
    size = 224

    #Compute # of output nodes
    out_nodes = 2

    ###set model architecture
    if(arguments['--surgery']!="1"):
        ###Load Surgery type from csv
        surgery = pd.read_csv(arguments['--surgery'])
        output_df = surgery
    else:
        files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir,f))] 
        
        
        if(len(files)==1):
            one = True
            files.extend(files)
         ###Results
        output_df = pd.DataFrame(columns = ['File','Surgery','Dummy'])
        
        output_df['File'] = files
        
        print("No surgery types supplied... assuming all patients are undergoing isolated CABG")
        output_df['Surgery'] = int(arguments['--surgery'])
    output_df['Dummy'] = np.random.randint(0,2,output_df.shape[0])
    output_df.columns = ['File','Surgery','Dummy']

    col = 'Dummy'
    imgs = (ImageList.from_df(df=output_df,path=image_dir)
                                .split_none()
                                .label_from_df(cols=col)
                                .transform(tfms_test,size=size)
                                .databunch(num_workers = num_workers,bs=bs).normalize(imagenet_stats))
                                
                                
    
    
    def get_model(pretrained=True, model_name = 'inceptionv4', **kwargs ): 
        if pretrained:
            arch = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        else:
            arch = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained=None)
        return arch

    def get_cadene_model(pretrained=True, **kwargs ): 
        return fastai_inceptionv4
    custom_head = create_head(nf=2048*2, nc=37, ps=0.75, bn_final=False) 
    fastai_inceptionv4 = nn.Sequential(*list(children(get_model(model_name = 'inceptionv4'))[:-2]),custom_head) 
    
    
    learn = cnn_learner(imgs, get_cadene_model, metrics=accuracy)
        
    N = len(image_dir.split("/"))
    dir_fix = "../"*(N-1)
    learn.model_dir = "."
    learn.load(os.path.join(dir_fix,mdl_path))



    preds,y,sd_preds = _TTA(learn,ds_type = DatasetType.Fix,activ=nn.Softmax())
    
    ###output predictions as column with model name
    output_df['CXR_CTSurgery_RAW'] = np.array(preds[:,1])
    output_df['SD_Prediction'] = np.array(sd_preds[:,1])

    

    learn.data.batch_size = 1
    learn.data.valid_dl = imgs.train_dl.new(shuffle=False)
    learn.model.eval()

    rc = GradCAMUtils.ResnetCAM(learn.model)
    count = 0
    #Saliency maps
    if(arguments['--saliency'] is not None):
        for i in progress_bar(learn.data.valid_dl):
            img = i[0]
            tmp = img.resize(1,3,224,224).cuda()
            tmp.requires_grad_()
            pred = rc(tmp)

            prob = F.softmax(pred,dim=1)
            pred[:,1].backward()
            saliency,_ = torch.max(tmp.grad.data.abs(),dim=1)
            filename = output_df.iloc[count,0]

            img = Image.open(os.path.join(image_dir,filename)).convert('RGB')
            img = transforms.ToTensor()(img)
            new_img = rc.blendImage(saliency[0,:,:].detach().clone().cpu(),img.detach().clone(),alpha=0.5,cmap='hot')
        ###Because filenames include full path here
            if(len(filename.split("/"))>1):
                tmp_fname = filename.split("/")
                filename = tmp_fname[len(tmp_fname)-1]
        
            new_img.save(os.path.join(arguments['--saliency'],filename))
            count = count + 1
    if(one):
        output_df.drop([1])

    coefs = [0,-0.027707,-1.05982,0.09618,0.34556,0.54107,-13.91101,0.18578]
    curr_coef = [coefs[x] for x in output_df['Surgery']]
    logit = 30.36518*np.array(output_df['CXR_CTSurgery_RAW']) + curr_coef - 4.96393
    prob = np.exp(logit) / (1 + np.exp(logit))
    output_df['CXR_CTSurgery_Final'] = prob
    output_df = output_df.drop(["Dummy"],axis=1)
    output_df.to_csv(arguments['<output_file>'])
