from sklearn import datasets, model_selection, cluster
from blackbox import *
from surrogate import *
from transport import *
from utility import *
import random
import time
import copy

# X - (unlabeled) data to be used in fusion, and upon which the post-fusion models will be tested.
# YY - ground truth labels for the data in X. not actly needed for fusion, only here for testing fused model quality.
# return value: Y (2d array). each elem corresponds to a datum in X, and contains each surrogate's post-fusion prediction of the label for that datum.
def fuse_surrogate(exp_name, sur, X, YY, n_iter=100, lr=2):
    Y = np.zeros((X.shape[0], len(sur)))
    g = open(result_folder() + exp_name + "_test.txt", "w+")
    h = open(result_folder() + exp_name + "_truth.txt", "w+")
    tm = open(result_folder() + exp_name + "_dectime.txt", "w+")
    dec_time = 0
    lin_time = 0
    for i in range(X.shape[0]): # for each datum
        # init y-candidates
        t1 = time.time()
        f = open(result_folder() + exp_name + "_profile" + str(i) + ".txt", "w+")
        x = X[i].reshape((1, X.shape[1]))
        for j in range(X.shape[1]): # prints the features of the current input datum
            g.write(str(x[0, j]))
            if j == X.shape[1] - 1:
                g.write("\n")
            else:
                g.write(",")
        h.write(str(YY[i, 0]) + "\n")
        ysur = np.zeros(len(sur))   # array of all the surrogates' predictions for this input
        print("Fusing prediction for test point #" + str(i))
        t2 = time.time()
        for j in range(len(sur)):
            #ysur[j] = 100 * random.random()
            #ysur[j] = sur[j].predict(x)
            ysur[j] = random.gauss(sur[j].predict(x), 2.0)
            f.write(str(ysur[j]))
            if j == len(sur) - 1:
                f.write("\n")
            else:
                f.write(",")
        print("Initial guess " + str(ysur) + " Truth = " + str(YY[i,0]))
        
        # this is the fusion part. note that the surrogates themselves never change. the fusion is
        # literally just asking each surrogate to make a prediction in ysur, then slowly nudging those
        # predictions until they converge. for each prediction y (aka ysur[i]), y is nudged in the direction that
        # leads to the fastest increase in p(y|x,...); in other words, the direction of the gradient.
        # in this manner, by the time the predictions converge, we'll have made it such that the new
        # converged prediction has a higher probability of being correct than the original predictions.
        for t in range(n_iter):
            dysur = np.zeros(len(sur)) # global gradient. it's an array because each surrogate's prediction will be updated using a diff global gradient.
            for j1 in range(len(sur)):
                for j2 in range(len(sur)):
                    # the global gradient for a surrogate (j1) depends on the gradients of all surrogates at the point of j1's prediction for the current input.
                    # essentially for each prediction ysur[j1], we ask all models: how should we change ysur[j1] so that p(ysur[j1] | x...) is maximized?
                    # (i.e. what's the gradient?) then we sum all the gradients to get our global gradient, and nudge ysur[j1] in that direction.
                    # in this way, each prediction will slowly be changed into a new prediction that's more likely to be correct given x.
                    grad = sur[j2].dy(x, ysur[j1]) # note that dy takes on more extreme values when j2 and j1's predictions are more different.
                    dysur[j1] += grad
            for j in range(len(sur)):
                #print(str(j) + " " + str(ysur[j]) + " " + str(dysur[j]))
                ysur[j] += lr * dysur[j] / (np.linalg.norm(dysur) * (t + 1) ** 0.5)
                f.write(str(ysur[j]))
                if j == len(sur) - 1:
                    f.write("\n")
                else:
                    f.write(",")
            #print(str(ysur)  + " Truth = " + str(YY[i,0]))
        
        # fusion done.

        print("Post-fusion " + str(ysur) + " Truth = " + str(YY[i,0]))
        for j in range(len(sur)):
            Y[i, j] = ysur[j]
        f.close()
        t3 = time.time()
        dec_time += (t2 - t1) + (t3 - t2) / len(sur)
        lin_time += t3 - t1

    tm.write(str(dec_time) + " " + str(lin_time))
    tm.close()
    h.close()
    g.close()

    return Y

def fuse_surrogate2(exp_name, sur, X, YY, n_iter=40):
    Y = np.zeros((X.shape[0], len(sur)))
    g = open(result_folder() + exp_name + "_test.txt", "w+")

    for j in range(X.shape[0]):
        print("Fusing surrogates at test point # " + str(j))
        f = open(result_folder() + exp_name + "_profile" + str(j) + ".txt", "w")
        x = X[j].reshape((1, X.shape[1]))
        t1 = time.time()
        for t in range(X.shape[1]):
            g.write(str(x[0, t]))
            if t == X.shape[1] - 1: g.write("\n")
            else: g.write(",")
        t2 = time.time()
        for s in range(len(sur)):
            Y[j, s] = sur[s].predict(x)
            #Y[j, s] = random.gauss(sur[s].predict(x), 0.0)
            f.write(str(Y[j, s]))
            if s == len(sur) - 1: f.write("\n")
            else: f.write(",")

        for i in range(n_iter):
            dw = np.zeros((len(sur), sur[0].n_params))
            dy = np.zeros(len(sur))
            for s in range(len(sur)):
                dy[s] = 0.0
            for s in range(len(sur)):
                for t in range(len(sur)):
                    dw[s] += sur[t].dw(x, Y[j, s], w_ext=sur[s].w)
                    dy[s] += sur[t].dy(x, Y[j, s])
            for s in range(len(sur)):
                dw[s] = dw[s] / np.linalg.norm(dw[s])
                dy[s] = dy[s] / np.linalg.norm(dy[s])
            for s in range(len(sur)):
                sur[s].update(i, dw[s])
                Y[j, s] += dy[s] / (2 * np.sqrt((i + 1) * n_iter))
                f.write(str(Y[j, s]))
                if s == len(sur) - 1:
                    f.write("\n")
                else:
                    f.write(",")
        t3 = time.time()

        f.close()
    g.close()

    return Y

# the X here is actly Xtu.
def fit_surrogate(bb, su, tr, X, ne=1, n_iter=20):
    for i in range(n_iter):
        start = time.time()
        print("Iter #" + str(i))
        ww = np.zeros(su.w.shape[0])
        uu = np.zeros(tr.u.shape[0])
        for t in range(su.w.shape[0]):
            ww[t] = np.exp(su.w[t])
        for t in range(tr.u.shape[0]):
            uu[t] = np.exp(tr.u[t])
        
        # ww contains e^x for x in su's weights. same for uu and tr's u.
        print(str(ww))
        print(str(uu))

        du = np.zeros(2)
        dw = np.zeros(su.n_params)
        
        # these are the epsilon noise terms in the paper. all sampled from N(0,1)        
        epw = np.zeros(ne)
        epu = np.zeros(ne)
        for j in range(ne):
            epw[j] = random.gauss(0, 1)
            epu[j] = random.gauss(0, 1)

        # for each datum
        for j in range(X.shape[0]):
            x = X[j].reshape(1, X.shape[1]) # flatten datum into a 1d vector

            # Compute dL/du = dL/dy * dy/du
            for eps in epu:
                y = tr.h(eps)           # makes sense, paper substitutes y with h(eps; u)
                dyp = bb.dy(x, y)       # "given this x and y, how shld i nudge y to have the fastest increase in p(y|x)?"
                dyq = su.dy(x, y)
                du += (dyq - dyp) * tr.du(eps) / X.shape[0] # bracketed term is dL/dy; tr.du(eps) is dy/du.
            du /= ne

            # Compute dL/dw
            for eps in epw:
                y = tr.h(eps)
                dw += su.dw(x, y) / (ne * X.shape[0])

        print(str(du))
        print(str(dw))
        du = du / np.linalg.norm(du)
        dw = dw / np.linalg.norm(dw)

        su.update(i, dw)
        tr.update(i, du)
        end = time.time()
        print(str(end - start))
    return su, tr

def experiment_9():
    X, Y = load_traffic()
    nb = 10
    box_type = []
    trp_type = []
    sur_type = []
    for i in range(nb):
        box_type.append("SGP")
        trp_type.append("Aff")
        sur_type.append("FGP")
    colbi2(X, Y, "colbi_aimpeak_timeslice_sgpbb", box_type, trp_type, sur_type)


def experiment_10():
    X, Y = load_traffic()
    nb = 10
    box_type = []
    trp_type = []
    sur_type = []
    for i in range(nb):
        box_type.append("SGP")
        trp_type.append("Aff")
        sur_type.append("FGP")
    colbi2(X, Y, "colbi_aimpeakfull_timeslice_sgpbb", box_type, trp_type, sur_type, box_size=200)


def experiment_8():
    X, Y = load_traffic()
    nb = 10
    box_type = []
    trp_type = []
    sur_type = []
    for i in range(nb):
        if i < 5:
            box_type.append("SGP")
        else:
            box_type.append("BRR")
        trp_type.append("Aff")
        sur_type.append("FGP")
    colbi(X, Y, "colbi_fullaimpeak_mixbb", box_type, trp_type, sur_type)


def experiment_7():
    X, Y = load_traffic()
    nb = 10
    box_type = []
    trp_type = []
    sur_type = []
    for i in range(nb):
        box_type.append("SGP")
        trp_type.append("Aff")
        sur_type.append("FGP")
    colbi(X, Y, "colbi_fullaimpeak_sgpbb", box_type, trp_type, sur_type)

def experiment_6():
    X, Y = load_protein()
    nb = 10
    box_type = []
    trp_type = []
    sur_type = []
    for i in range(nb):
        if i < 5:
            box_type.append("SGP")
        else:
            box_type.append("BRR")
        trp_type.append("Aff")
        sur_type.append("FGP")
    colbi(X, Y, "colbi_protein_mixbb", box_type, trp_type, sur_type)

def experiment_5():
    X, Y = load_protein()
    nb = 10
    box_type = []
    trp_type = []
    sur_type = []
    for i in range(nb):
        box_type.append("SGP")
        trp_type.append("Aff")
        sur_type.append("FGP")
    colbi(X, Y, "colbi_protein_sgpbox", box_type, trp_type, sur_type)

# AIMPEAK, only SGP blackboxes
def experiment_4():
    X, Y = load_traffic()
    nb = 10
    box_type = []
    trp_type = []
    sur_type = []
    for i in range(nb):
        if i < 5:
            box_type.append("SGP")
        else:
            box_type.append("BRR")
        trp_type.append("Aff")
        sur_type.append("FGP")
    colbi(X, Y, "colbi_aimpeak_mixbb", box_type, trp_type, sur_type)



# AIMPEAK, only SGP blackboxes
def experiment_3():
    X, Y = load_traffic()
    nb = 10
    box_type = []
    trp_type = []
    sur_type = []
    for i in range(nb):
        box_type.append("SGP")
        trp_type.append("Aff")
        sur_type.append("FGP")
    colbi(X, Y, "colbi_aimpeak_sgpbb", box_type, trp_type, sur_type)

# Diabetes, mixed black-boxes (BRR + SGP)
def experiment_2():
    X, Y = datasets.load_diabetes(return_X_y=True)
    nb = 10
    box_type = []
    trp_type = []
    sur_type = []
    for i in range(nb):
        if i < 5:
            box_type.append("SGP")
        else:
            box_type.append("BRR")
        trp_type.append("Aff")
        sur_type.append("FGP")
    colbi(X, Y, "colbi_diabetes_mixbb", box_type, trp_type, sur_type)


# Diabetes, only SGP black-boxes
def experiment_1():
    X, Y = datasets.load_diabetes(return_X_y=True)
    nb = 10
    box_type = []
    trp_type = []
    sur_type = []
    for i in range(nb):
        box_type.append("SGP")
        trp_type.append("Aff")
        sur_type.append("FGP")
    colbi(X, Y, "colbi_diabetes_sgpbb", box_type, trp_type, sur_type)


def colbi2(X, Y, exp_name, box_type, trp_type, sur_type, box_size=50, test_size=500):
    nb = len(box_type)
    #kf = model_selection.KFold(n_splits = nb + 2)
    Xb = []
    Yb = []
    slice_size = 775
    slice_per_box = 5
    offset = 0
    for i in range(nb):
        train_index = list(range(offset, offset + slice_size * slice_per_box))
        Xb.append(X[train_index])
        Yb.append(Y[train_index].reshape((slice_size * slice_per_box, 1)))
        offset += slice_size * slice_per_box
    test_index = list(range(offset, offset + slice_size * 2))
    Xt = copy.deepcopy(X[test_index])
    Yt = copy.deepcopy(Y[test_index].reshape((slice_size * 2, 1)))
    offset += slice_size * 2
    Xs = copy.deepcopy(X[offset:])

    '''
    fold_no = 0
    for train_index, test_index in kf.split(X):
        if fold_no < nb:
            Xb.append(X[test_index])
            Yb.append(Y[test_index].reshape((len(test_index), 1)))
        elif fold_no == nb:
            Xs = X[test_index]
            Ys = Y[test_index].reshape((len(test_index), 1))
        else:
            Xt = X[test_index]
            Yt = Y[test_index].reshape((len(test_index), 1))
        fold_no += 1
    '''
    id = np.random.choice(range(Xs.shape[0]), min(box_size, Xs.shape[0]), replace=False)
    Xs = Xs[id, :]

    box = []
    sur = []
    trp = []
    for b in range(nb):
        print("Fitting Surrogate #" + str(b))
        if box_type[b] == "SGP":
            id = np.random.choice(range(Xb[b].shape[0]), min(box_size, Xb[b].shape[0]), replace=False)
            Xsu = Xb[b][id, :]
            Ysu = Yb[b][id, :]
            bx = SGP_BlackBox(Xsu, Ysu, Xs)
        elif box_type[b] == "BRR":
            id = np.random.choice(range(Xb[b].shape[0]), min(box_size, Xb[b].shape[0]), replace=False)
            Xsu = Xb[b][id, :]
            Ysu = Yb[b][id, :]
            bx = BRR_BlackBox(Xsu, Ysu.reshape(Ysu.shape[0]))

        if sur_type[b] == "Lin":
            su = LinearSurrogate(X.shape[1])
            su.sigma = np.log(2.0)
            su.bias = 5
            su.unfold()
        elif sur_type[b] == "FGP":
            id = np.random.choice(range(Xb[b].shape[0]), min(box_size, Xb[b].shape[0]), replace=False)
            Xsu = Xb[b][id, :]
            Ysu = Yb[b][id, :]
            su = GPSurrogate(Xb[b].shape[1] + 2, Xsu, Ysu)
            su.precompute_inv()

        if trp_type[b] == "Aff":
            tr = AffineTransport(u=np.asarray([np.exp(1), 1]))

        id = np.random.choice(range(Xt.shape[0]), min(box_size, Xt.shape[0]), replace=False)
        Xtu = Xt[id, :]

        box.append(bx)
        sur.append(su)
        trp.append(tr)

        sur[b], trp[b] = fit_surrogate(box[b], sur[b], trp[b], Xtu)
    f1 = open(result_folder() + exp_name + "_disagreement.txt", "w+")
    f2 = open(result_folder() + exp_name + "_fusion_rmse.txt", "w+")

    id = np.random.choice(range(Xt.shape[0]), min(test_size, Xt.shape[0]), replace=False)
    Xtu = Xt[id, :]
    Ytu = Yt[id, :]
    '''
    box.append(copy.deepcopy(box[0]))
    sur.append(copy.deepcopy(sur[0]))
    '''

    pre_sur_rmse = []
    for b in range(len(sur)):
        print("Testing Surrogate #" + str(b) + " Alignment")
        rmse = 0.0
        box_rmse = 0.0
        sur_rmse = 0.0
        for j in range(Xtu.shape[0]):
            x = Xtu[j].reshape(1, Xtu.shape[1])
            box_pred = box[b].predict(x)
            sur_pred = sur[b].predict(x)
            rmse += (box_pred - sur_pred) ** 2
            box_rmse += (box_pred - Ytu[j, 0]) ** 2
            sur_rmse += (sur_pred - Ytu[j, 0]) ** 2
        rmse = (rmse / Xt.shape[0]) ** 0.5
        box_rmse = (box_rmse / Xtu.shape[0]) ** 0.5
        print("Disagreement = " + str(rmse))
        print("Box RMSE = " + str(box_rmse))
        print("Sur RMSE = " + str((sur_rmse / Xtu.shape[0]) ** 0.5))
        pre_sur_rmse.append((sur_rmse / Xtu.shape[0]) ** 0.5)
        f1.write(str(box_rmse) + "," + str(pre_sur_rmse[b]) + "," + str(rmse) + "\n")

    pred = fuse_surrogate(exp_name, sur, Xtu, Ytu)
    post_sur_rmse = []
    for b in range(len(sur)):
        print("Testing Surrogate #" + str(b) + " Post Fusion")
        sur_rmse = 0.0
        for j in range(Xtu.shape[0]):
            #print(str(pred[j,b]) + " " + str(Ytu[j, 0]))
            sur_rmse += (pred[j, b] - Ytu[j, 0]) ** 2
        print("Sur RMSE = " + str((sur_rmse / Xtu.shape[0]) ** 0.5))
        post_sur_rmse.append((sur_rmse / Xtu.shape[0]) ** 0.5)

    for b in range(len(sur)):
        f2.write(str(pre_sur_rmse[b]) + "," + str(post_sur_rmse[b]) + "\n")

    print("Average Sur RMSE pre-fusion = " + str(sum(pre_sur_rmse) / len(sur)))
    print("Average Sur RMSE post-fusion = " + str(sum(post_sur_rmse) / len(sur)))


# box_type, trp_type, sur_type are types of blackbox model, transport func (h(eps; u)), and surrogate. all are lists, each elem represents one model.
# i think box_size is num of data used to train blackbox?
def colbi(X, Y, exp_name, box_type, trp_type, sur_type, box_size=50, test_size=500):
    print(f'Beginning colbi() function')

    nb = len(box_type)
    kf = model_selection.KFold(n_splits = nb + 2)
    # Xb, Yb, Xs, Ys, Xt, Yt are *lists* of test data
    # i think these correspond to blackbox, surrogate, transportfunc. or does t stand for test?
    Xb = []
    Yb = []
    fold_no = 0
    
    for train_index, test_index in kf.split(X): # note that train_index, test_index are arrays
        # we have 1-fold worth of data for the surrogate/transport, but nb folds worth for the blackbox
        if fold_no < nb:
            Xb.append(X[test_index])
            Yb.append(Y[test_index].reshape((len(test_index), 1)))
        elif fold_no == nb:
            Xs = X[test_index]
            Ys = Y[test_index].reshape((len(test_index), 1))
        else:
            Xt = X[test_index]
            Yt = Y[test_index].reshape((len(test_index), 1))
        fold_no += 1

    id = np.random.choice(range(Xs.shape[0]),min(box_size, Xs.shape[0]), replace=False) # id is an array
    Xs = Xs[id, :]

    # lists of blackboxes, surrogates, transport funcs
    box = []
    sur = []
    trp = []

    # for each blackbox (this loop creates blackboxes, surrogates, and transportfuncs)
    for b in range(nb):
        print("Fitting Surrogate #" + str(b))
        # create the black boxes.
        if box_type[b] == "SGP":
            id = np.random.choice(range(Xb[b].shape[0]), min(box_size, Xb[b].shape[0]), replace=False)
            Xsu = Xb[b][id, :]
            Ysu = Yb[b][id, :]
            bx = SGP_BlackBox(Xsu, Ysu, Xs)
        elif box_type[b] == "BRR":
            id = np.random.choice(range(Xb[b].shape[0]), min(box_size, Xb[b].shape[0]), replace=False)
            Xsu = Xb[b][id, :]
            Ysu = Yb[b][id, :]
            bx = BRR_BlackBox(Xsu, Ysu.reshape(Ysu.shape[0]))

        # create surrogates
        if sur_type[b] == "Lin":
            su = LinearSurrogate(X.shape[1])
            su.sigma = np.log(2.0)
            su.bias = 5
            su.unfold()
        elif sur_type[b] == "FGP":
            id = np.random.choice(range(Xb[b].shape[0]), min(box_size, Xb[b].shape[0]), replace=False) # sample some data from Xb[b]
            Xsu = Xb[b][id, :]
            Ysu = Yb[b][id, :]
            su = GPSurrogate(Xb[b].shape[1] + 2, Xsu, Ysu)
            su.precompute_inv()

        # create transportfuncs
        if trp_type[b] == "Aff":
            tr = AffineTransport(u=np.asarray([np.exp(1), 1]))

        # Xtu is a subset of Xt. it's an array because id is an array.
        id = np.random.choice(range(Xt.shape[0]), min(box_size, Xt.shape[0]), replace=False)
        Xtu = Xt[id, :]

        box.append(bx)
        sur.append(su)
        trp.append(tr)

        sur[b], trp[b] = fit_surrogate(box[b], sur[b], trp[b], Xtu)

    # by this point, the blackboxes, surrogates, transportfuncs have been created

    f1 = open(result_folder() + exp_name + "_disagreement.txt", "w+")
    f2 = open(result_folder() + exp_name + "_fusion_rmse.txt", "w+")

    # Xtu and Ytu are *subsets* of Xt and Yt. note that both Xtu and Ytu are arrays because id is an array.
    # Xtu represents (unlabeled) data which will be used in the surrogate fusion, and upon which the fused
    # model's quality will be tested. Ytu is the labels for Xtu; it's not needed in the fusion, but it's
    # useful for determining fused model accuracy.
    id = np.random.choice(range(Xt.shape[0]), min(test_size, Xt.shape[0]), replace=False)
    Xtu = Xt[id, :]
    Ytu = Yt[id, :]
    '''
    box.append(copy.deepcopy(box[0]))
    sur.append(copy.deepcopy(sur[0]))
    '''

    pre_sur_rmse = []
    for b in range(len(sur)):
        print("Testing Surrogate #" + str(b) + " Alignment")
        rmse = 0.0
        box_rmse = 0.0
        sur_rmse = 0.0
        for j in range(Xtu.shape[0]):
            x = Xtu[j].reshape(1, Xtu.shape[1])
            box_pred = box[b].predict(x)
            sur_pred = sur[b].predict(x)
            rmse += (box_pred - sur_pred) ** 2
            box_rmse += (box_pred - Ytu[j, 0]) ** 2
            sur_rmse += (sur_pred - Ytu[j, 0]) ** 2
        rmse = (rmse / Xt.shape[0]) ** 0.5
        box_rmse = (box_rmse / Xtu.shape[0]) ** 0.5
        print("Disagreement = " + str(rmse))
        print("Box RMSE = " + str(box_rmse))
        print("Sur RMSE = " + str((sur_rmse / Xtu.shape[0]) ** 0.5))
        pre_sur_rmse.append((sur_rmse / Xtu.shape[0]) ** 0.5)
        f1.write(str(box_rmse) + "," + str(pre_sur_rmse[b]) + "," + str(rmse) + "\n")

    pred = fuse_surrogate(exp_name, sur, Xtu, Ytu)
    post_sur_rmse = []
    for b in range(len(sur)):
        print("Testing Surrogate #" + str(b) + " Post Fusion")
        sur_rmse = 0.0
        for j in range(Xtu.shape[0]):
            #print(str(pred[j,b]) + " " + str(Ytu[j, 0]))
            sur_rmse += (pred[j, b] - Ytu[j, 0]) ** 2
        print("Sur RMSE = " + str((sur_rmse / Xtu.shape[0]) ** 0.5))
        post_sur_rmse.append((sur_rmse / Xtu.shape[0]) ** 0.5)

    for b in range(len(sur)):
        f2.write(str(pre_sur_rmse[b]) + "," + str(post_sur_rmse[b]) + "\n")

    print("Average Sur RMSE pre-fusion = " + str(sum(pre_sur_rmse) / len(sur)))
    print("Average Sur RMSE post-fusion = " + str(sum(post_sur_rmse) / len(sur)))


def main():
    random.seed(2010)
    #experiment_1()
    #experiment_2()
    #experiment_3()
    #experiment_4()
    #experiment_5()
    #experiment_6()
    #experiment_7()
    #experiment_8()
    #experiment_9()
    experiment_10()

if __name__ == "__main__":
    main()
