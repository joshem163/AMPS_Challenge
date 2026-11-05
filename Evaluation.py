from modules import *
#EvaluationData_dir = "./dataset/Evaluation/Phase1Evaluation"
EvaluationData_dir = "./EvaluationData"
folder_dicts=[]
for idx, folder in enumerate(os.listdir(EvaluationData_dir)):
    folder_path = os.path.join(EvaluationData_dir, folder)
    if not os.path.isdir(folder_path):
        continue
    sensorDataFile = os.path.join(folder_path, "SensorData.mat")
    folder_dicts.append({"file":folder, "sensorDataFile":sensorDataFile})


def attack_detector(sensorDataFile,model,num_class):
    sensorDict = loadmat(sensorDataFile)
    pmu_va = sensorDict["PMU_Va"]
    pmu_vm = sensorDict["PMU_Vm"]
    pmu_f = sensorDict["PMU_f"]
    pmu_vm_normalized = MinMaxScaler().fit_transform(pmu_vm)
    Xseq = make_sliding_window_test(pmu_vm_normalized, window_size=5, stride=5)
    X = torch.tensor(Xseq, dtype=torch.float32)
    # Test on new data
    with torch.no_grad():
        new_output = model(X.unsqueeze(0))  # new_X should be preprocessed like the original training data
        #print(len(new_output[0]))
        _, new_predictions = torch.max(new_output, 1)
        print(new_predictions)
        #probs = torch.sigmoid(new_output)
        one_hot_label = np.zeros(num_class)
        if new_predictions!=0:
            one_hot_label[new_predictions] = 0.99
    upsampled_preds = np.repeat(one_hot_label, repeats=5, axis=0)
    return upsampled_preds

# Define input and output dimensions (example placeholders)
input_dim = 340
num_timesteps=1200
hidden_dim = 256
output_dim = 981
num_class=981
n_heads = 2
n_layers = 2
# Re-initialize the model with the same architecture
model = TransformerClassifier(input_dim, hidden_dim, output_dim, n_heads, n_layers, num_timesteps,drop_out=0.3)

# Load the saved model weights
#model.load_state_dict(torch.load('Attack_detector.pth'))
model.load_state_dict(torch.load('Attack_detector_single.pth', map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode
detection_dict={}
for _file_dict in  folder_dicts:
    detection_dict[_file_dict["file"]]=attack_detector(_file_dict["sensorDataFile"],model,num_class)
detection_df=pd.DataFrame.from_dict(detection_dict)
detection_df.to_csv("DetectionDataFrame.csv")