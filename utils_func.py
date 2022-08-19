import numpy as np


#Project adv example on l_ inf adv_bound ball centered on clean example
def clip_adv(X_adv, X_test, indices_test, adv_bound):
    X_clip_adv = np.zeros(X_test[indices_test].shape)
    j = 0
    for i in indices_test:
        diff = X_adv[j] - X_test[i]
        X_clip_adv[j] = X_test[i] + np.clip(diff, -adv_bound, adv_bound)
        j = j +1
    return(X_clip_adv)
    

#Project adv example on l_2 adv_bound ball centered on clean example
def clip_adv_l2(X_adv, X_test, indices_test, adv_bound):
    X_clip_adv = np.zeros(X_test[indices_test].shape)
    j = 0
    for i in indices_test:
        diff = X_adv[j] - X_test[i]
        norm = np.sqrt(np.maximum(1e-12, np.sum(np.square(diff))))
        # We must *clip* to within the norm ball, not *normalize* onto the# surface of the ball
        factor = np.minimum(1., np.divide(adv_bound, norm))
        diff = diff * factor
        X_clip_adv[j] = X_test[i] + diff
        j = j +1
    return(X_clip_adv)
    

#Function which returns the adversarial accuracy, the number of successful adversarial examples, l_2,l_inf, l_1 and l_0 distances between successful adversarial examples and clean observations
def metrics(model, X_adv, X_test, y_pred, indices_test):    
    adv_pred = np.argmax(model.predict(X_adv), axis = 1)
    adv_acc =  np.mean(np.equal(adv_pred, y_pred[indices_test]))
    l2_distort_success = 0
    linf_distort_success = 0
    l1_distort_success = 0
    l0_distort_success = 0
    l2_distort_fail = 0
    linf_distort_fail = 0
    l1_distort_fail = 0
    l0_distort_fail = 0
    nb_success = 0
    j = 0
    for i in indices_test:
        if (adv_pred[j] != y_pred[i]):
            l2_distort_success = l2_distort_success + np.linalg.norm(X_test[i] - X_adv[j])
            linf_distort_success = linf_distort_success + np.max(abs(X_test[i] - X_adv[j]))
            l1_distort_success = l1_distort_success + np.sum(np.abs(X_test[i] - X_adv[j]))
            l0_distort_success = l0_distort_success +  np.linalg.norm(X_test[i].flatten() - X_adv[j].flatten(), ord=0)
            nb_success = nb_success + 1     
        if (adv_pred[j] == y_pred[i]):
            l2_distort_fail = l2_distort_fail + np.linalg.norm(X_test[i] - X_adv[j])
            linf_distort_fail = linf_distort_fail + np.max(abs(X_test[i] - X_adv[j]))
            l1_distort_fail = l1_distort_fail + np.sum(np.abs(X_test[i] - X_adv[j]))
            l0_distort_fail = l0_distort_fail +  np.linalg.norm(X_test[i].flatten() - X_adv[j].flatten(), ord=0)
        j = j+1        
    nb_fail = len(indices_test) - nb_success
    if ((nb_fail != 0) & (nb_success != 0)):
        return(adv_acc, nb_success, l2_distort_success/nb_success, linf_distort_success/nb_success, l1_distort_success/nb_success,
               l0_distort_success/nb_success, l2_distort_fail/nb_fail, linf_distort_fail/nb_fail, l1_distort_fail/nb_fail,
               l0_distort_fail/nb_fail)
    elif (nb_fail == 0):
        return(adv_acc, nb_success, l2_distort_success/nb_success, linf_distort_success/nb_success, l1_distort_success/nb_success,
               l0_distort_success/nb_success, "non", "non", "non", "non")
    elif (nb_success == 0):
        return(adv_acc, nb_success, "non", "non", "non", "non", l2_distort_fail/nb_fail, linf_distort_fail/nb_fail, l1_distort_fail/nb_fail,
               l0_distort_fail/nb_fail)
        
        
