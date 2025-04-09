**Uses Fl4health packages:**  
client and server should be used in example.basic_example  
fraud_full.csv is the synthetic data set used for training

**Things attempted for troubleshooting:**  
1. manual inspection of TensorDataset for train and val to ensure they properly capture the original dataframe  
2. check to ensure all dataloading methods from the original class are present (get_data_loader, get_test_data_loader)  
3. update model to have sigmoid activation at the end instead of raw logits to match the accuracy calculation method  
4. modified 0,0,0,0 to localhost to work with windows  
