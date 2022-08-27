import torch
import numpy as np
import time
import network
import dataset
import metrics
from tqdm import tqdm
from torch.utils.data import SubsetRandomSampler
import sklearn.ensemble, sklearn.linear_model, sklearn.dummy
from sklearn import preprocessing

start_time = time.time()

# Run logistic regression classifiers on Z to predict Y and to predict P
def evaluate(name,dataset_type,latent_dim):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cpu':
        debugging = True
    else:
        debugging = False

    torch.manual_seed(2025)
    np.random.seed(2025)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    numworkers = 32 if torch.cuda.is_available() else 0

    if dataset_type == 0:  # CelebA sensitive ITA (Skin Tone)
        img_dim = 128
        train_set, test_set = dataset.get_celeba(debugging=debugging)
    elif dataset_type == 3:  # CelebA sensitive Gender
        img_dim = 128
        train_set, test_set = dataset.get_celeba_gender(debugging=debugging)
    elif dataset_type == 1:  # EyePACS
        img_dim = 256
        train_set, test_set = dataset.get_eyepacs()

    model = network.RFPIB(latent_dim, img_dim=img_dim).to(device)

    if device == 'cuda':
        model.load_state_dict(torch.load(f'../results/{name}'))
    else:
        model.load_state_dict(torch.load(f'../results/{name}',map_location=torch.device('cpu')))
    model.eval()

    ''' Predict  P'''
    predictor = sklearn.linear_model.LogisticRegression(solver='liblinear')
    model.eval()

    with torch.no_grad():
        # Train
        traindataloader = torch.utils.data.DataLoader(train_set, batch_size=128,
                                                      shuffle=True, num_workers=numworkers)
        p_list = []
        z_list = []

        for x, y, s, p in tqdm(traindataloader, disable=not (debugging)):
            x = x.to(device).float()
            p = p.to(device).float()

            z, mu, logvar = model.getz(x)

            p_list.append(p)
            z_list.append(z)

        Z_train = torch.cat(z_list, dim=0)
        P_train = torch.cat(p_list, dim=0)

        Z_train = Z_train.cpu()
        P_train = P_train.cpu()

        scaler = preprocessing.StandardScaler().fit(Z_train)
        Z_scaled = scaler.transform(Z_train)

        Z_scaled = np.where(np.isnan(Z_scaled), np.zeros_like(Z_scaled), Z_scaled)
        Z_scaled = np.where(np.isinf(Z_scaled), np.zeros_like(Z_scaled), Z_scaled)
        predictor.fit(Z_scaled, P_train)

        # Test
        testdataloader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=numworkers)

        z_list = []
        p_list = []

        for x, y, s, p in tqdm(testdataloader, disable=not (debugging)):
            x = x.to(device).float()
            p = p.to(device).float()

            z, mu, logvar = model.getz(x)

            z_list.append(z)
            p_list.append(p)

        Z_test = torch.cat(z_list, dim=0)
        P_test = torch.cat(p_list, dim=0)

        Z_test = Z_test.cpu()
        Z_scaled = scaler.transform(Z_test)
        Z_scaled = np.where(np.isnan(Z_scaled), np.zeros_like(Z_scaled), Z_scaled)
        Z_scaled = np.where(np.isinf(Z_scaled), np.zeros_like(Z_scaled), Z_scaled)

        predictions = predictor.predict_proba(Z_scaled)
        predictions = np.argmax(predictions, 1)
        p_ground_truth = P_test.cpu().detach().numpy()
        accuracy = metrics.get_accuracy(predictions, p_ground_truth)

        print(f"logistic accuracy predicting p from z = {accuracy}")
        p_accuracy = accuracy

    ''' predict Y'''
    predictor = sklearn.linear_model.LogisticRegression(solver='liblinear')
    model.eval()
    with torch.no_grad():
        # Train
        traindataloader = torch.utils.data.DataLoader(train_set, batch_size=128,
                                                      shuffle=True, num_workers=numworkers)
        y_list = []
        z_list = []

        for x, y, s, p in tqdm(traindataloader, disable=not (debugging)):
            x = x.to(device).float()
            y = y.to(device).float()

            z, mu, logvar = model.getz(x)

            z_list.append(z)
            y_list.append(y)

        Z_train = torch.cat(z_list, dim=0)
        Y_train = torch.cat(y_list, dim=0)

        Z_train = Z_train.cpu()
        Y_train = Y_train.cpu()

        scaler = preprocessing.StandardScaler().fit(Z_train)
        Z_scaled = scaler.transform(Z_train)


        Z_scaled = np.where(np.isnan(Z_scaled), np.zeros_like(Z_scaled), Z_scaled)
        Z_scaled = np.where(np.isinf(Z_scaled), np.zeros_like(Z_scaled), Z_scaled)
        predictor.fit(Z_scaled, Y_train)

        # Test
        testdataloader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=numworkers)

        s_list = []
        y_list = []
        z_list = []

        for x, y, s, p in tqdm(testdataloader, disable=not (debugging)):
            x = x.to(device).float()
            y = y.to(device).float()
            z, mu, logvar = model.getz(x)

            s_list.append(s)
            z_list.append(z)
            y_list.append(y)

        Z_test = torch.cat(z_list, dim=0)
        S_test = torch.cat(s_list, dim=0)
        Y_test = torch.cat(y_list, dim=0)

        Z_test = Z_test.cpu()
        Z_scaled = scaler.transform(Z_test)
        Z_scaled = np.where(np.isnan(Z_scaled), np.zeros_like(Z_scaled), Z_scaled)
        Z_scaled = np.where(np.isinf(Z_scaled), np.zeros_like(Z_scaled), Z_scaled)

        predictions = predictor.predict_proba(Z_scaled)
        predictions = np.argmax(predictions, 1)
        s = S_test.cpu().detach().numpy()
        y = Y_test.cpu().detach().numpy()
        accuracy = metrics.get_accuracy(predictions, y)

        accgap = metrics.get_acc_gap(predictions, y, s)
        dpgap = metrics.get_discrimination(predictions, s)
        eqoddsgap = metrics.get_equalized_odds_gap(predictions, y, s)
        accmin0, accmin1 = metrics.get_min_accuracy(predictions, y, s)

        print(f"logistic accuracy predicting y from z = {accuracy}")
        print(f"logistic accgap = {accgap}")
        print(f"logistic dpgap = {dpgap}")
        print(f"logistic eqoddsgap = {eqoddsgap}")
        print(f"logistic acc_min_0 = {accmin0}")
        print(f"logistic acc_min_1 = {accmin1}")

    return np.array([round(p_accuracy, 4), round(accuracy, 4), round(accgap, 4), round(dpgap, 4), round(eqoddsgap, 4),
                 round(accmin0, 4),round(accmin1,4)])