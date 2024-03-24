from architecture.simplebert import BertLinear1HEAD
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score,f1_score, precision_score, recall_score

class Predict:
    def __init__(self,name, test_data_loader,model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bertcnn=BertLinear1HEAD(name).to(self.device)
        self.bertcnn.load_state_dict(torch.load(model_path))  # Load the state dictionary
        self.bertcnn.eval()
        self.test_data_loader  = test_data_loader

    def predictions_labels(self,preds,labels):
        pred = np.argmax(preds,axis=1).flatten()
        label = labels.flatten()
        return pred,label

    def eval(self):
        all_true_sent = []
        all_pred_sent = []

        with torch.no_grad():
            for batch in tqdm(self.test_data_loader):
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_sent = batch[2].to(self.device)
                #b_clas = batch[3].to(self.device)

                self.sent_predictions = self.bertcnn(b_input_ids,b_input_mask)
                #loss1 = self.criterion(sent_predictions, b_sent)
                #loss2 = self.criterion(clas_predictions, b_clas)

                #t_loss = loss1

                self.sent_predictions = self.sent_predictions.detach().cpu().numpy()
                #clas_predictions = clas_predictions.detach().cpu().numpy()

                label_sent = b_sent.to('cpu').numpy()
                #label_clas = b_clas.to('cpu').numpy()

                pred1, true1 = self.predictions_labels(self.sent_predictions,label_sent)
                #pred2, true2 = self.predictions_labels(clas_predictions,label_clas)

                all_pred_sent.extend(pred1)
                #all_pred_clas.extend(pred2)

                all_true_sent.extend(true1)
                #all_true_clas.extend(true2)

            val_accuracy_sent = accuracy_score(all_pred_sent,all_true_sent)
            #val_accuracy_clas = accuracy_score(all_pred_clas,all_true_clas)

            sent_f1_score = f1_score(all_true_sent,all_pred_sent,average='macro')
            sent_f1_scorew = f1_score(all_true_sent,all_pred_sent,average='weighted')
            #clas_f1_score = f1_score(all_pred_clas,all_true_clas,average='macro')
            #clas_f1_scorew = f1_score(all_pred_clas,all_true_clas,average='weighted')
            precision_score_sent = precision_score(all_true_sent,all_pred_sent)
            recall_score_sent = recall_score(all_true_sent,all_pred_sent)
            accs = (val_accuracy_sent)
            f1s= (sent_f1_score,sent_f1_scorew)
            # cm = confusion_matrix(all_true_sent, all_pred_sent)
            # disp = ConfusionMatrixDisplay(confusion_matrix=cm)


        return  accs,  f1s, precision_score_sent, recall_score_sent