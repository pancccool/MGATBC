function [BrainNetSet]=PC(BOLD,lambda)
%% Basic parameter 
nTime=size(BOLD{1},1);
nSubj=length(BOLD);  
RegionNum=size(BOLD{1},2);
Total_Data=zeros(nSubj,nTime,RegionNum);
for SubjectID=1:nSubj 
    tmp=BOLD{SubjectID};
    subject=tmp(:,1:RegionNum); 
    subject=subject-repmat(mean(subject),nTime,1); 
    subject=subject./(repmat(std(subject),nTime,1)); 
    Total_Data(SubjectID,:,:)=subject; 
end
%% Network construction
BrainNetSet=cell(length(lambda),1);
for SubjectID=1:nSubj
    for l1=1:size(lambda,2)
        param=lambda(l1);
        BrainNet=zeros(RegionNum,RegionNum);
        tmp=zeros(size(Total_Data,2),size(Total_Data,3));
        tmp(:,:)=Total_Data(SubjectID,:,:);
        currentNet=corrcoef(tmp); 
        currentNet=currentNet-diag(diag(currentNet));
        threhold=prctile(abs(currentNet(:)),param); 
        currentNet(abs(currentNet)<=threhold)=0;
        BrainNet(:,:)=currentNet;
        
        BrainNetSet{l1,1}(SubjectID,:,:)=BrainNet;
        fprintf('Done the %d subject networks with lamda1 equal to %d!\n',SubjectID,l1);
    end
end
save('data\BrainNetSet_HC_SZ_PC.mat','BrainNetSet');