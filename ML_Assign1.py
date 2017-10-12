# coding=utf-8
import math
from collections import OrderedDict, defaultdict
import sys, codecs, optparse, os
import pandas as pd

optparser = optparse.OptionParser()
optparser.add_option("-m", "--max_prob", dest='invoke_mle', default=False, help="Calculate MLE")
optparser.add_option("-p", "--predict", dest='invoke_classifier', default=False, help="naives bayes classifier")
optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'preprocessed_datasets.csv'), help="input file")
optparser.add_option("-c", "--targetclass", dest="target_class", default='GP_greater_than_0', help="Target Class")
optparser.add_option("-b", "--besselbias", dest="bessel_bias", default=False, help="Bessel_Bias")


(opts, _) = optparser.parse_args()


class NBClassifier:

    def __init__(self, training_data, testing_data, col_names):
        self.training = training_data
        self.testing = testing_data
        self.target = opts.target_class
        self.features = col_names
        self.mle_map = {}

        # calculating mean variance for class
        for val in self.training[self.target].unique():
            self.mle_map[val] = {}
            class_prob = len(self.training[self.training[self.target] == val]) / float(self.training[self.target].count())
            print val
            print len(self.training[self.training[self.target] == val])
            print len(self.training)
            self.mle_map[val]['prob'] = math.log10(class_prob)
            for f in self.features:
                self.mle_map[val][f] = calculate_mean_variance(self.training, f, cond_name=self.target, cond_val=val)

    def predict(self):

        pred = {}

        for idx, row in self.testing.iterrows():
            pred[row.id] = {}
            # calculate probabilities
            probs = self.get_gaussian_prob_value(row)

            # cal max_prob and classify its class
            pred[row.id] = self.classify(probs)
        print "accuracy ==> {}".format(self.get_accuracy(pred))

    def get_gaussian_prob_value(self, xi):

        probs = {}

        for key, val in self.mle_map.iteritems():

            # P(X) = ( 1 / SQRT( 2 * PI )* STD) * e ^(- (x-mean)^2 / 2 * var)
            #
            log_prob = 1
            for f in self.features:
                f_estimators = val[f]
                const_calc = (1 / (math.sqrt(2 * math.pi) * f_estimators['std']))
                exp_calc = math.exp(-(math.pow(xi[f] - f_estimators['mean'], 2) / float(2 * f_estimators['var'])))

                # const_calc = - (math.log10(math.sqrt(2 * math.pi)) + math.log10(f_estimators['std']))
                # exp_calc = -(math.pow((xi[f] - f_estimators['mean']), 2) / 2 * f_estimators['var'])

                log_prob = log_prob * (const_calc * exp_calc)
            probs[key] = log_prob

        return probs

    def classify(self, prob_map):

        max_prob = None
        max_prob_class = None

        for key, val in prob_map.iteritems():
            if max_prob is None or val > max_prob:
                max_prob = val
                max_prob_class = key

        return max_prob_class

    def get_accuracy(self, predicted_data):

        total_count = len(self.testing.index)
        total_match = 0
        print "total Dataset Count => {}".format(total_count)

        for idx, row in self.testing.iterrows():
            if row[self.target] is predicted_data[row.id]:
                total_match += 1

        print "Correctly predicted Count => {}".format(total_match)
        return (total_match / float(total_count)) * 100


def calculate_mean_variance(data, col_name, cond_name=None, cond_val=None):

    filtered_data = data
    if cond_name:
        filtered_data = data.loc[data[cond_name] == cond_val]
    # for normal distribution mean
    val_list = [r for r in filtered_data[col_name].tolist() if r is not None]
    if val_list:
        mean = sum(val_list) / float(len(val_list))

        var_attr = [pow((r - mean), 2) for r in val_list if r is not None]
        if opts.bessel_bias:
            var = sum(var_attr) / float(len(val_list) - 1)
        else:
            var = sum(var_attr) / float(len(val_list))

        return {'mean': mean, 'var': var, 'std': math.sqrt(var)}

    return None



df = pd.read_csv(opts.input)

if opts.invoke_mle:

    res = calculate_mean_variance(df, "Weight")

    print 'The maximum likelihood estimates for column weights'
    print ' mean  = {} and variance = {}'.format(res['mean'], res['var'])

    res = calculate_mean_variance(df, "Weight", cond_name=opts.target_class, cond_val='yes')

    print 'The maximum likelihood estimates for column weights conditional on GP > 0 being true'
    print ' mean  = {} and variance = {}'.format(res['mean'], res['var'])

    res = calculate_mean_variance(df, "Weight", cond_name=opts.target_class, cond_val='no')

    print 'The maximum likelihood estimates for column weights conditional on GP > 0 being false'
    print ' mean  = {} and variance = {}'.format(res['mean'], res['var'])

elif opts.invoke_classifier:

    training_df = df[df[u'DraftYear'].isin([1998, 1999, 2000])]
    testing_df = df[df[u'DraftYear'] == 2001]

    features = df._get_numeric_data().columns

    drop_class = [u'id', u'sum_7yr_GP', u'sum_7yr_TOI',u'DraftYear', u'po_PlusMinus', u'GP_greater_than_0']

    features = [f for f in features if f not in drop_class]

    classifier = NBClassifier(training_df, testing_df, features)

    classifier.predict()

