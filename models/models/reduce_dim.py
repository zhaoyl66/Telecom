import pickle
import json
import numpy as np
from sklearn.decomposition import PCA

token_list = []

for i in range(564):
    with open("train/train"+str(i)+".pkl","rb+") as f1:
        token = pickle.load(f1)
        # print(token[0][0].size())
        # break
        for j in range(16):
            try:
                t=token[0][j].cpu().detach().numpy()
                pca = PCA(n_components=100)
                reduced_data = pca.fit_transform(t)
                reduced_rows = 50
                reduced_matrix = reduced_data[:reduced_rows,:]
                token_list.append(reduced_matrix)
            except IndexError:
                break

print(len(token_list))
print(token_list[0].shape)


# with open("train.txt",'w') as fx:
#     for i in range(len(token_list)):
#         fx.write(str(token_list[i])+'\n')

# print("write label")
label_list = []

with open ("/data/zyl2023/hpt/data/Telecom1/Telecom1_trainseg1_clear.json","r",encoding="utf-8") as f2:
    for line in f2:
        data = json.loads(line)
        label = data['label']
        label_list.append(label)

print(len(label_list))

# with open("train_label.pkl","wb") as f3:
    # pickle.dump(label_list,f3)

with open("train_dem.pkl","wb") as fs:
    pickle.dump(token_list,fs)
    pickle.dump(label_list,fs)
