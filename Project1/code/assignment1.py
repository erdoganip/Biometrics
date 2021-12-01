##Ipek Erdogan
##2019700174
##Biometrics HW1
##
import numpy as np
import matplotlib.pyplot as plt

def plot_far_frr(far_,frr_,th_):
    plt.plot(th_, far_, 'r')
    plt.plot(th_, frr_, 'b')
    plt.xlabel('Threshold')
    plt.title('FAR-FRR Graph')
    plt.show()

def plot_roc(far_,gmr_):
    plt.plot(far_, gmr_, 'b')
    plt.xlabel('FAR')
    plt.ylabel('GMR')
    plt.title('ROC Curve')
    plt.show()

def plot_dist(gen,imp):
    _, bins, _ = plt.hist(gen, bins=50, label='Genuine', density=True)
    _ = plt.hist(imp, bins=bins, alpha=0.5, label='Impostor', density=True)
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution')
    plt.legend(loc='upper right')
    plt.show()

def dist_calc(file_name):
    impostor_dist = []
    genuine_dist = []
    similarity_mat = np.loadtxt(file_name, dtype='f', delimiter=',')
    for i in range(1000):
        if (i % 10 == 0):
            base = i  # It will be 0, 10,20,30,40 etc.
        for j in range(1000):
            temp = similarity_mat[i][j]
            if (np.isnan(temp)):
                continue
            elif ((not np.isnan(temp)) and (base <= j <= base + 9)):
                genuine_dist.append(temp)
            else:
                impostor_dist.append(similarity_mat[i][j])
    return genuine_dist,impostor_dist

def create_guess_matrix(similarity_mat,threshold):
    guess_matrix = np.zeros((1000, 1000))
    for i in range(1000):
        for j in range(1000):
            temp=similarity_mat[i][j]
            if (not np.isnan(temp)):
                if(temp>threshold):
                    guess_matrix[i][j]=1
            else:
                guess_matrix[i][j]=np.nan
    return guess_matrix

def metric_calculate(file_name, threshold_se,threshold_step):
    far_list=[]
    frr_list=[]
    gmr_list=[]
    threshold_start = threshold_se[0]
    threshold_end = threshold_se[1]
    thresholds = np.arange(threshold_start, threshold_end, threshold_step)
    similarity_mat = np.loadtxt(file_name, dtype='f', delimiter=',')
    for threshold in thresholds:
        false_positive = 0
        true_positive = 0
        false_negative = 0
        true_negative = 0
        guess_matrix = create_guess_matrix(similarity_mat, threshold)
        for i in range(1000):
            if (i % 10 == 0):
                base = i  # It will be 0, 10,20,30,40 etc.
            for j in range(1000):
                temp = guess_matrix[i][j]
                if (np.isnan(temp)):
                    continue
                elif ((not np.isnan(temp)) and (base <= j <= base + 9)):
                    if (temp == 1):
                        true_positive += 1
                    else:
                        false_negative += 1
                else:
                    if (temp == 0):
                        true_negative += 1
                    else:
                        false_positive += 1
        far = false_positive / 990000
        far_list.append(far)
        gmr = true_positive / 9000
        gmr_list.append(gmr)
        frr = 1 - gmr
        frr_list.append(frr)
    return far_list,frr_list,gmr_list,thresholds

def find_frr(fr_l,fa_l):
    frr_1=0
    frr_01=0
    frr_001=0
    for index, frr_value in enumerate(fr_l):
        if (frr_value-0.1<0.001): #I determined 0.001 as an error rate
            frr_1= fa_l[index]
        elif (frr_value-0.01<0.001): #I determined 0.001 as an error rate
            frr_01= fa_l[index]
        elif (frr_value-0.001<0.0001): #I determined 0.0001 as an error rate
            frr_001 = fa_l[index]
        else:
            continue
    return frr_1,frr_01,frr_001

def find_err(fr_l,fa_l):
    err=0
    for index,frr_value in enumerate(fr_l):
        far_value=fa_l[index]
        if(abs(far_value-frr_value)<=0.001):  #I determined 0.001 as an error rate
            err=(far_value+frr_value)/2
            return err,index
    return 0,0

#1=True 0=False
if __name__ == '__main__':
    filenames=["data1_SM.txt","data2_SM.txt","data3_SM.txt","data4_SM.txt"]
    thresholds_=[[-2200,200],[0,1.2],[-1,1],[-0.1,1]]
    threshold_steps=[1,0.01,0.01,0.01]
    for index, file in enumerate(filenames):
        print("Results of",file)
        fa_l,fr_l,gm_l,th=metric_calculate(file,thresholds_[index],threshold_steps[index])
        frr_1,frr_01,frr_001=find_frr(fr_l,fa_l)
        print("FRR value at 0.1 FAR: ",frr_1)
        print("FRR value at 0.01 FAR: ", frr_01)
        print("FRR value at 0.001 FAR: ", frr_001)
        err_value, err_index = find_err(fr_l, fa_l)
        err_threshold = th[err_index]
        print("ERR value:", err_value)
        print("ERR threshold: ", err_threshold)
        print("\n")
        gen_dist, imp_dist = dist_calc(file)
        plot_dist(gen_dist,imp_dist)
        plot_far_frr(fa_l,fr_l,th)
        plot_roc(fa_l,gm_l)

