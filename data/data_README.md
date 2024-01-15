Multi-turn Dialogue Responsibility Assignment Dataset

Multi-turn Dialogue Responsibility Assignment dataset is specifically designed for analyzing the process of assigning responsibility in multi-turn telecommunications complaint conversations.

# Data Source
These conversations are transcribed from spoken dialogues between customer service agents from Guangzhou Telecom Company and customers who have raised complaints about their telecommunications services. The dialogues in the dataset are conducted over the phone, typically following a customer's complaint in Chinese about telecommunications services. The customer service agents participating in these conversations are employees of Guangzhou Telecom Company, while the customers are individuals who have lodged complaints regarding their telecommunications services.

# Data Preprocessing
During the data preprocessing phase, we have implemented measures to protect the privacy of the users by encrypting personally identifiable information, such as phone numbers, to ensure that the data is anonymized and in compliance with privacy regulations. This approach is critical to safeguard the sensitive data of both the customers and the service agents involved in the conversations.

# Data Characteristics
We present the Multi-turn Dialogue Responsibility Assignment dataset to facilitate the study of complaint responsibility assignment in multi-turn dialogues between telecommunication complaint users and customer service personnel. It differs from existing dialogue datasets in two aspects: 
1) Hierarchical Responsibility Labels: Unlike many other dialogue datasets that may have flat or simple labels, this dataset provides a hierarchical structure for categorizing the complaints, ranging from general fee issues to specific problems such as incorrect value recharge; 
2) Progressive Refinement of Labels: As the conversation between the customer service agent and the customer unfolds, the responsibility labels become progressively refined.
 
# Data Format
The telecommunication customer service multi-turn dialogue complaint responsibility dataset contains a total of 14,315 data entries, including training, development, and testing sets. 
The dataset includes dialogue texts and hierarchical responsibility labels.
1) The structure of dialogue data is as follows:
[CLS]first turn[EOS]second turn[EOS]……[EOS]last turn[SEP]
Due to the inclusion of personal information in the dialogue content, the dialogue texts are encrypted and processed using the BERT-base-Chinese model for embedding, followed by dimensionality reduction. 
2) The hierarchical responsibility labels are encrypted, with different numbers representing different categories of labels. The hierarchical structure of the labels is preserved in the 'slot.pt' file.

|                          |     Train     |       Val      |       Test      |
| ------------------------ | ------------- | -------------- | --------------- |
| Dialogue-label pairs     |     9018      |      2270      |      2832       |
| Avg. token per session   |      534      |       551      |       538       |
| Min token per session    |       13      |        22      |        19       |
| Max token per session    |     5732      |      5433      |      6092       |
| Avg. turn per session    |        6      |         6      |         6       |
| Min turn per session     |        2      |         2      |         2       |
| Max turn per session     |       47      |        50      |        46       |