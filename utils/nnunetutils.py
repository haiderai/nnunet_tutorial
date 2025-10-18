import os, shutil, glob
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.utilities.dataset_name_id_conversion import convert_dataset_name_to_id
import torch
from datetime import datetime

def fMakeDir(dirpath):
    os.makedirs(dirpath, exist_ok=True)
    return

def fNepochstoTrainername(nepochs):
    if nepochs == 1000:
        return 'nnUNetTrainer'
    nepochchoices = [1,10,20,50,100, 250, 500, 750, 2000, 4000, 8000]
    trainernames = [f'nnUNetTrainer_{t}epochs' for t in nepochchoices]
    if not(nepochs in nepochchoices):
        print(f"Training aborted: {nepochs} not one of {nepochchoices}")
        return ""
    print(f'**** Using Trainer {nepochchoices.index(nepochs)} ****')
    return trainernames[nepochchoices.index(nepochs)]
    
def fInitialize(rootdir, modelname):

    nnUNet_raw_data = os.environ['nnUNet_raw'] = os.path.join(rootdir, 'nnUNetv2_raw')
    preprocessing_output_dir = os.environ['nnUNet_preprocessed']=os.path.join(rootdir, 'nnUNetv2_preprocessed')
    network_training_output_dir = os.environ['nnUNet_results']=os.path.join(rootdir,"nnUNetv2_results")
    
    target_base = os.path.join(nnUNet_raw_data, modelname)
    target_imagesTr = os.path.join(target_base, "imagesTr")
    target_imagesTs = os.path.join(target_base, "imagesTs")
    target_labelsTs = os.path.join(target_base, "labelsTs")
    target_labelsTr = os.path.join(target_base, "labelsTr")
    fMakeDir(nnUNet_raw_data)
    fMakeDir(preprocessing_output_dir)
    fMakeDir(network_training_output_dir)
    fMakeDir(target_imagesTr)
    fMakeDir(target_labelsTs)
    fMakeDir(target_imagesTs)
    fMakeDir(target_labelsTr)

    print('\nENVIRONMENT VARIABLES')        
    print(f"nnUNet_raw:{os.environ['nnUNet_raw']}")
    print(f"nnUNet_preprocessed:{os.environ['nnUNet_preprocessed']}")
    print(f"nnUNet_results:{os.environ['nnUNet_results']}")
    
    return {"target_base": target_base, "target_imagesTr":target_imagesTr,"target_imagesTs":target_imagesTs,
            "target_labelsTr":target_labelsTr,
            "target_imagesTs":target_imagesTs,
            "target_labelsTs": target_labelsTs,
            "target_labelsTr":target_labelsTr
            }

def fPrepareFilesMP(dirdic, channel_names, labels, 
                    mSourceDir, imagefilenames, segfilename, region_class_order):
    
    prefix = "IM"  # file name prefix
    
    # get files
    filenames = []
    mFolders = [f.path for f in os.scandir(mSourceDir) if f.is_dir() ]
    mFolders.sort()

    
    for ii, mFolder in enumerate(mFolders):
              
        skipcase = False        
        
        # look for segmentation file
        msegfnmask = os.path.join(mFolder, segfilename)
        seglist = glob.glob(msegfnmask)
        if len(seglist)>=1:
            msegfn = seglist[0]  # if multiple seg matches take first one
        else:
            print(f'Cant find segmentation file {msegfnmask}, skipping exam')
            continue
        
        if not(os.path.exists(msegfn)): 
            print(f'Cant find segmentation file {msegfn}, skipping exam')
            continue

        # get image channels as a list
        mimfnlist = []
        for imagefilename in imagefilenames:
            mimfn = os.path.join(mFolder, imagefilename)
            # deal with wildcards
            if not(os.path.exists(mimfn)):
                print(f'Missing {mimfn}, exam skipped')
                skipcase = True
                break  # go to next exam
            mimfnlist.append(mimfn)
        if skipcase: continue
        # at this stage there is a list of channel filenames in mimfnlist
        filenames.append({'image': mimfnlist, 'label': msegfn}) 
        # end of exam loop

    # at this point filenames contains a list of filenames with a list of image channels and a labelfile for each
    
    # partition files - in this case all fields are used in folds
    train_files = filenames
    train_set_size = len(train_files)
    print(f'A total of {train_set_size} of {len(mFolders)} are for training')

    serial_number = 0
    # copy training files
    for jj, p in enumerate(train_files):
        curfiles = p['image'] # all channels
        serial_number = jj+1
                            
        labelfn = f'{prefix}_{serial_number:03d}.nii.gz'
        labelfn = os.path.join(dirdic["target_labelsTr"],labelfn)
        if not(os.path.exists(labelfn)):
            oldlabelfn = p['label']
            print(f'Copy {oldlabelfn} to {labelfn}')
            shutil.copy(oldlabelfn, labelfn)

        for channel_number, curfile in enumerate(curfiles):
            print(f'{serial_number}/{train_set_size}: {curfile}')
            imfn = f'{prefix}_{serial_number:03d}_{channel_number:04d}.nii.gz'
            imfn = os.path.join(dirdic["target_imagesTr"],imfn)
            if not(os.path.exists(imfn)):
                oldimfn = p['image'][channel_number]
                print(f'Copy {oldimfn} to {imfn}')
                shutil.copy(oldimfn, imfn)
    

    generate_dataset_json(dirdic["target_base"],channel_names,
                            labels,
                            train_set_size, regions_class_order=region_class_order,
                            file_ending='.nii.gz')
    return

def fPreProcess(modelname):
    
        taskno = convert_dataset_name_to_id(modelname)
        execstr = f'nnUNetv2_plan_and_preprocess -d {taskno} --verify_dataset_integrity'
        print(f'Executing command line: {execstr}')
        os.system(execstr)
        return
    
    
def fTrain(modelname, configurations, folds=[0,1,2,3,4], gpuid=0, nepochs=10):

    """
    gpuid - id of GPU to run training on, only use if multiple GPU's and want to train on a sepecific one

    """  
    taskno = convert_dataset_name_to_id(modelname)
    trainername = fNepochstoTrainername(nepochs)
    
    # get cuda visible devices
    ngpus = torch.cuda.device_count()
    if ngpus == 0:
        print("No GPUS aborting training")
        return
    if gpuid > (ngpus-1):
        print("ERROR, invlaid GPUID")
        return        

    print(f"Setting CUDA_VISIBLE_DEVICES to {gpuid}")
    os.environ["CUDA_VISIBLE_DEVICES"]= str(gpuid)
        
    for config in configurations:
        for fold in folds:        
            execstr = f'nnUNetv2_train {modelname} {config} {fold} -tr {trainername}' 
            print(f'Executing command line: {execstr}')
            os.system(execstr)
    return


def fPredict(rootdir, modelname, filestopredict, segfilename, configuration, folds, nepochs=10):
  
        trainername = fNepochstoTrainername(nepochs)
        
        # make dummy folders
        runname = f'{modelname}-{datetime.now().strftime("%Y%m%d%H%M%S%f")}'
        casefolder = os.path.join(rootdir, 'temp', f'{modelname}_predict-in_{runname}')
        predfolder = os.path.join(rootdir, 'temp', f'{modelname}_predict-out_{runname}')
        fMakeDir(casefolder)
        fMakeDir(predfolder)

        prefix = "IM"
        
        predfilelist = []
        for jj, filetopredict in enumerate(filestopredict):
            # reformat input file into correct format
            predfile = os.path.join(casefolder, f'{prefix}_001_{jj:04d}.nii.gz')
            
            print(f'copying {filetopredict} to {predfile}')
            shutil.copyfile(filetopredict, predfile)
            predfilelist = predfilelist + [predfile]

        exestr = f'nnUNetv2_predict -i {casefolder} -o {predfolder} -d {modelname} -c {configuration} -tr {trainername} -f {folds} -p nnUNetPlans'
       
        # Execute prediction
        print(f'\nExcuting command: {exestr}\n')
        os.system(exestr)
        
        smsegfilename = None
        
        outfile = os.path.join(predfolder, f'{prefix}_001.nii.gz')
        print(f'Copying segmentation result {outfile} to {segfilename}')
    
        try:
            shutil.copyfile(outfile, segfilename)
        except:
            print("FAILED to copy file")
            return
            
        shutil.rmtree(casefolder)
        shutil.rmtree(predfolder)
        return segfilename, smsegfilename
        

