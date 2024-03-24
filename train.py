import torch
from Trainer.head_trainer import Trainer
from utils.dataloader import CreateDataset
import pandas as pd

# If you want to use wandb, uncomment the lines that are commented

# import wandb
# wandb.login(key="b46a760f71842e87d8ac966f77b2db06d0a7085a")

architectures=["linear"]
bert_name="FacebookAI/xlm-roberta-base"

train_path = "dataset/cleaned_data/training/train.csv"
eval_path = "dataset/cleaned_data/testing/localch.csv"

train_set = pd.read_csv(train_path,encoding='utf-8',usecols=[1,2])
train_set.dropna(inplace=True)
test_set = pd.read_csv(eval_path,encoding='utf-8',usecols=[1,2])
test_set.dropna(inplace=True)

is_smart = True
percentages = [1]
for architecture in architectures:
  for p in percentages :
    # If `extract` = True, the model will be loaded from a checkpoint, and you have to pass
    # the checkpoint path to the `varient` parameter in Trainer
    extract = False
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("___________",bert_name,"____________")
    # wandb.init(
    #   project = "2-Head_Bert",
    #   name = bert_name + architecture + "2-head_vfsc" + "smart",
    # )
    batch_size = 128
  
    epochs = 40
    # if "large" in model_name:
    #   batch_size=4
    
    # varient='Epoch-17-2-head-linear-smart-5_5-3head'
    train_data_loader = CreateDataset(train_set['name'], train_set['label'], bert_name, batch_size=batch_size).todataloader()
    test_data_loader  = CreateDataset(test_set['name'], test_set['label'], bert_name, batch_size=batch_size).todataloader()
    bertcnn=Trainer(bert_name,  train_data_loader, test_data_loader, model=architecture,is_smart=is_smart,extract=extract,batch_size =batch_size)
    bertcnn.fit(schedule=True,epochs=epochs,report=False,name=f"{architecture}-victsd",percentage= p)
    # wandb.finish()

    del bertcnn
    del train_data_loader
    del test_data_loader
    torch.cuda.empty_cache()

    print("_______________End__________________")