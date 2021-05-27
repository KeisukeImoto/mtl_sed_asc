import sys, os
import numpy as np
import matplotlib.pyplot as plot

np.set_printoptions(suppress=True)
sys.path.append(os.pardir)

class SEDresult():
    def __init__(self,output,label,params,setting):
        self.setting = setting
        self.mode = setting['thresmode']
        self.threshold = setting['threshold']
        self.startthres = setting['startthres']
        self.endthres = setting['endthres']
        self.nevent = params['nevent']
        self.output = output
        self.label = label
        self.prediction = []
        self.ntp = []
        self.ntn = []
        self.nfp = []
        self.nfn = []
        self.ntp_event = []
        self.ntn_event = []
        self.nfp_event = []
        self.nfn_event = []
        self.recall = []
        self.precision = []
        self.fpr = []
        self.micro_fscore = []
        self.macro_fscore = []
        self.micro_er = []
        self.macro_er = []
        self.recall_event = []
        self.precision_event = []
        self.fpr_event = []
        self.fscore_event = []
        self.er_event = []
        self.prauc = 0.0
        self.rocauc = 0.0
        self.prauc_event = np.zeros(self.nevent)
        self.rocauc_event = np.zeros(self.nevent)

    def detection(self,threshold=0.5,interval=0.02):

        # thresholding and detecting sound events
        threslist = []
        if self.mode in ['fixed','auc']:
            if type(threshold) is list:
                prediction = np.asarray((self.output>=threshold).astype(int))
            elif type(threshold) in (int, float):
                prediction = np.asarray((self.output>=threshold).astype(int))
        elif self.mode == 'adaptive':
            delta = 1.0e-10; prediction = np.zeros(np.shape(self.output),dtype=int)
            tmpfscore = np.zeros((int((self.endthres-self.startthres)/interval+1),self.nevent),dtype=float)
            for ii in range(int((self.endthres-self.startthres)/interval+1)):
                tmpprediction = np.asarray((self.output>=((self.startthres+interval*ii))).astype(int))
                tmpntp = np.sum(tmpprediction&self.label,axis=0)
                tmpnfp = np.sum(tmpprediction-(tmpprediction&self.label),axis=0)
                tmpnfn = np.sum(self.label-(tmpprediction&self.label),axis=0)
                tmpfscore[ii,:] = (2*tmpntp)/(2*tmpntp+tmpnfp+tmpnfn+delta)

            maxind = np.argmax(tmpfscore,axis=0)
            maxval = np.max(tmpfscore,axis=0)
            for ii in range(self.nevent):
                prediction[:,ii] = np.asarray((self.output[:,ii]>=(self.startthres+interval*maxind[ii])).astype(int))
                threslist.append(self.startthres+interval*maxind[ii])
            self.threslist = threslist

        self.prediction.append(prediction)

    def calc_fscore(self):

        # calculate # true positive, true negative, false positive, and false negative frames
        delta = 1.0e-10
        self.ntp.append(np.sum(self.prediction[-1]&self.label)) # # true positive frames in test dataset
        self.ntn.append(np.sum((~self.prediction[-1]+2)&(~self.label+2))) # # true negative frames in test dataset
        self.nfp.append(np.sum(self.prediction[-1]-(self.prediction[-1]&self.label))) # # false positive frames in test dataset
        self.nfn.append(np.sum(self.label-(self.prediction[-1]&self.label))) # # false negative frames in test dataset
        self.ntp_event.append(np.sum(self.prediction[-1]&self.label, axis=0)) # # true positive frames for each sound event
        self.ntn_event.append(np.sum((~self.prediction[-1]+2)&(~self.label+2), axis=0)) # # true negative frames for each sound event
        self.nfp_event.append(np.sum(self.prediction[-1]-(self.prediction[-1]&self.label), axis=0)) # # false positive frames for each sound event
        self.nfn_event.append(np.sum(self.label-(self.prediction[-1]&self.label), axis=0)) # # false negative frames for each sound event
        Nsys = np.sum(self.prediction[-1],axis=1) # # sound events detected by system in each frame
        Nref = np.sum(self.label,axis=1) # # sound events labeled in each frame
        tpframe = np.sum(self.prediction[-1]&self.label,axis=1) # # true positive events for each time frame

        # calculate substitutions, deletions, and insertions (https://tutcris.tut.fi/portal/files/6973737/applsci_06_00162.pdf)
        S = np.minimum(Nref,Nsys) - tpframe
        D = np.maximum(np.zeros(len(Nref)),Nref-Nsys)
        I = np.maximum(np.zeros(len(Nref)),Nsys-Nref)
        nlabel = np.sum(Nref); self.prediction[-1] = np.sum(Nsys)
        totalS = np.sum(S); totalD = np.sum(D); totalI = np.sum(I)

        # calculate average recall, precision, fscore, and error rate
        self.recall.append(self.ntp[-1]/(self.ntp[-1]+self.nfn[-1]+delta))
        self.precision.append(self.ntp[-1]/(self.ntp[-1]+self.nfp[-1]+delta))
        self.fpr.append(self.nfp[-1]/(self.nfp[-1]+self.ntn[-1]+delta))
        #self.micro_fscore.append(2*self.recall*self.precision/(self.recall+self.precision+delta))
        self.micro_fscore.append((2*self.ntp[-1])/(2*self.ntp[-1]+self.nfp[-1]+self.nfn[-1]+delta))
        self.micro_er.append((totalS+totalD+totalI)/nlabel)

        # calculate classwise recall, precision, fscore, and error rate
        self.recall_event.append(self.ntp_event[-1]/(self.ntp_event[-1]+self.nfn_event[-1]+delta))
        self.precision_event.append(self.ntp_event[-1]/(self.ntp_event[-1]+self.nfp_event[-1]+delta))
        self.fpr_event.append(self.nfp_event[-1]/(self.nfp_event[-1]+self.ntn_event[-1]+delta))
        self.fscore_event.append((2*self.ntp_event[-1])/(2*self.ntp_event[-1]+self.nfp_event[-1]+self.nfn_event[-1]+delta))
        self.er_event.append((self.nfp_event[-1]+self.nfn_event[-1])/np.sum(self.label,axis=0))
        self.macro_fscore.append(np.mean(self.fscore_event[-1]))
        self.macro_er.append(np.mean(self.er_event[-1]))

    def sed_evaluation(self,printflag=False,saveflag=False,path='None'):

        if self.mode in ['fixed','adaptive','learned']:
            self.detection(self.threshold)
            self.calc_fscore()

            if printflag:
                print('recall (frame-level metric) = ' + str(self.recall[-1]*100.0) + '%')
                print('precision (frame-level metric) = ' + str(self.precision[-1]*100.0) + '%')
                print('micro fscore (frame-level metric) = ' + str(self.micro_fscore[-1]*100.0) + '%')
                print('macro fscore (sound event, frame-level metric) = ' + str(self.macro_fscore[-1]*100.0) + '%')
                print('micro error rate (frame-level metric) = ' + str(self.micro_er[-1]))
                print('macro error rate (frame-level metric) = ' + str(self.macro_er[-1]))
                print('fscore for each acoustic event (frame-level metric) = ' + str(self.fscore_event[-1]*100.0) + '%')
                print('error rate for each acoustic event (frame-level metric) = ' + str(self.er_event[-1]))

            # save SED performance
            if saveflag:
                with open(path + '.accuracy','w') as fid:
                    fid.write('recall (frame-level metric) = ' + str(self.recall[-1]*100.0) + '%\n')
                    fid.write('precision (frame-level metric) = ' + str(self.precision[-1]*100.0) + '%\n')
                    fid.write('micro fscore (frame-level metric) = ' + str(self.micro_fscore[-1]*100.0) + '%\n')
                    fid.write('macro fscore (sound event, frame-level metric) = ' + str(self.macro_fscore[-1]*100.0) + '%\n')
                    fid.write('micro error rate (frame-level metric) = ' + str(self.micro_er[-1]) + '\n')
                    fid.write('macro error rate (frame-level metric) = ' + str(self.macro_er[-1]) + '\n')
                    fid.write('fscore for each acoustic event (frame-level metric) = ' + str(self.fscore_event[-1]*100.0).replace('\n','') + '%\n')
                    fid.write('error rate for each acoustic event (frame-level metric) = ' + str(self.er_event[-1]).replace('\n',''))

        elif self.mode == 'auc':

            for ii in np.arange(0.0,1.0,0.02):
                self.detection(float(ii))
                self.calc_fscore()

            # calculate precision-recall AUC
            for ii in range(len(self.recall)-1):
                self.prauc = self.prauc + (self.recall[ii]-self.recall[ii+1])*(self.precision[ii]+self.precision[ii+1])/2.0
                for jj in range(self.nevent):
                    self.prauc_event[jj] = self.prauc_event[jj] + (self.recall_event[ii][jj]-self.recall_event[ii+1][jj])*(self.precision_event[ii][jj]+self.precision_event[ii+1][jj])/2.0

            # calculate ROC AUC
            for ii in range(len(self.fpr)-1):
                self.rocauc = self.rocauc + (self.fpr[ii]-self.fpr[ii+1])*(self.recall[ii]+self.recall[ii+1])/2.0
                for jj in range(self.nevent):
                    self.rocauc_event[jj] = self.rocauc_event[jj] + (self.fpr_event[ii][jj]-self.fpr_event[ii+1][jj])*(self.recall_event[ii][jj]+self.recall_event[ii+1][jj])/2.0

            if printflag:
                print('micro precision-recall AUC (total) = ' + str(self.prauc*100.0) + '%')
                print('macro precision-recall AUC (total) = ' + str(np.mean(self.prauc_event)*100.0) + '%')
                print('precision-recall AUC (each_event) = ' + str(self.prauc_event*100.0) + '%')
                print('micro ROC AUC (total) = ' + str(self.rocauc*100.0) + '%')
                print('macro ROC AUC (total) = ' + str(np.mean(self.rocauc_event)*100.0) + '%')
                print('ROC AUC (each_event) = ' + str(self.rocauc_event*100.0) + '%')

            # save SED performance
            if saveflag:
                with open(path + '.auc','w') as fid:
                    fid.write('micro precision-recall AUC (total) = ' + str(self.prauc*100.0) + '%\n')
                    fid.write('macro precision-recall AUC (total) = ' + str(np.mean(self.prauc_event)*100.0) + '%\n')
                    fid.write('precision-recall AUC (each_event) = ' + str(self.prauc_event*100.0).replace('\n','') + '%\n')
                    fid.write('micro ROC AUC (total) = ' + str(self.rocauc*100.0) + '%\n')
                    fid.write('macro ROC AUC (total) = ' + str(np.mean(self.rocauc_event)*100.0) + '%\n')
                    fid.write('ROC AUC (each_event) = ' + str(self.rocauc_event*100.0).replace('\n','') + '%')

            self.fpr_event = np.array(self.fpr_event)
            self.recall_event = np.array(self.recall_event)

def asc_evaluation(prediction,label,printflg=False,saveflag=False,path='None'):

    delta = 1.0e-10
    nscene = max(label)+1
    confusion = np.zeros((nscene,nscene),dtype=int)
    for ii in range(len(prediction)):
        confusion[label[ii],prediction[ii]] = confusion[label[ii],prediction[ii]] + 1

    if printflg:
        print('confusion matrix (acoustic scene):')
        print(confusion)

    # calculate recall, precision, macro fscore, and micro fscore
    recall = np.diag(confusion)/(np.sum(confusion,axis=1)+delta)
    precision = np.diag(confusion)/(np.sum(confusion,axis=0)+delta)
    macro_fscore = 2*recall*precision/(recall+precision+delta)
    macro_fscore = np.mean(macro_fscore)
    micro_fscore = np.sum(np.diag(confusion))/(np.sum(confusion)+delta)

    if printflg:
        print('recall (acoustic scene) = ' + str(recall*100.0) + '%')
        print('precision (acoustic scene) = ' + str(precision*100.0) + '%')
        print('macro fscore (acoustic scene) = ' + str(macro_fscore*100.0) + '%')
        print('micro fscore (acoustic scene) = ' + str(micro_fscore*100.0) + '%')

    # save ASC performance
    if saveflag:
        with open(path + '_scene.accuracy','w') as fid:
            fid.write('recall (acoustic scene) = ' + str(recall*100.0) + '%\n')
            fid.write('precision (acoustic scene) = ' + str(precision*100.0) + '%\n')
            fid.write('macro fscore (acoustic scene) = ' + str(macro_fscore*100.0) + '%\n')
            fid.write('micro fscore (acoustic scene) = ' + str(micro_fscore*100.0) + '%\n')
            fid.write('confusion matrix (acoustic scene):\n')
            fid.write(str(confusion) + '\n')
