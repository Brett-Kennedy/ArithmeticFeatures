import numpy as np
import pandas as pd
from warnings import filterwarnings
import matplotlib.pyplot as plt
from datetime import datetime

# Sklearn model types tested
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel

import sys
sys.path.insert(0, 'C:\python_projects\ArithmeticFeatures_project\ArithmeticFeatures')
from ArithmeticFeatures import ArithmeticFeatures # todo: fix once have pip install

sys.path.insert(0, 'C:\python_projects\DatasetsEvaluator_project\DatasetsEvaluator')
import DatasetsEvaluator as de

filterwarnings('ignore')
np.random.seed(0)


# These specify how many datasets are used in the tests below. Ideally about 50 to 100 datasets would be used,
# but these may be set lower. Set to 0 to skip tests. 
NUM_DATASETS_CLASSIFICATION_DEFAULT = 100
NUM_DATASETS_CLASSIFICATION_PARAMETER_SEARCH = 50
NUM_DATASETS_REGRESSION_DEFAULT = 100
NUM_DATASETS_REGRESSION_DEFAULT_PARAMETER_SEARCH = 50
RUN_PARALLEL = True


# Model-based feature selection using RandomForest.
def get_features_rf_selects(X, y):
	rf = RandomForestClassifier().fit(X, y)
	model = SelectFromModel(rf, prefit=True)
	selected_arr = model.get_support()
	cols_used_list = [X.columns[i] for i in range(len(selected_arr)) if selected_arr[i]]
	return pd.DataFrame(X, columns=cols_used_list)


def print_header(test_name):
	stars = "*****************************************************"
	print(f"\n\n{stars}\n{test_name}\n{stars}")


def test_model_default_params(datasets_tester, estimators_arr, partial_result_folder, results_folder, accuracy_metric):
	summary_df_1, saved_file_name = datasets_tester.run_tests(
		estimators_arr=estimators_arr,
		feature_selection_func=None,
		num_cv_folds=5,
		show_warnings=False,
		partial_result_folder=partial_result_folder,
		results_folder=results_folder,
		run_parallel=RUN_PARALLEL)

	summary_df_2, saved_file_name = datasets_tester.run_tests(
		estimators_arr=estimators_arr,
		feature_selection_func=get_features_rf_selects,
		num_cv_folds=5,
		show_warnings=False,
		partial_result_folder=partial_result_folder,
		results_folder=results_folder,
		run_parallel=RUN_PARALLEL)

	summary_df_2['Feature Engineering Description'] = summary_df_2['Feature Engineering Description'] + " (w/ Feat. Sel.)"
	summary_df = summary_df_1.append(summary_df_2, ignore_index=True)
	datasets_tester.summarize_results(summary_df, accuracy_metric, saved_file_name, results_folder)
	datasets_tester.plot_results(summary_df, accuracy_metric, saved_file_name, results_folder)


def test_model_param_search(datasets_tester, estimators_arr, partial_result_folder, results_folder, accuracy_metric):
	orig_parameters = {}

	arith_parameters = {
		'arith__scale_data': (True, False),
		'arith__support_plus': (True, False),
		'arith__support_mult': (True, False),
		'arith__support_minus': (True, False),
		'arith__support_div': (True, False),
		'arith__support_min': (True, False),
		'arith__support_max': (True, False),
	}

	summary_df, saved_file_name = datasets_tester.run_tests_parameter_search(
		estimators_arr=estimators_arr,
		parameters_arr=[orig_parameters, arith_parameters],
		search_method='grid',
		num_cv_folds=5,
		show_warnings=False,
		partial_result_folder=partial_result_folder,
		results_folder=results_folder,
		run_parallel=RUN_PARALLEL)

	datasets_tester.summarize_results(summary_df, 'f1_macro', saved_file_name, results_folder)
	datasets_tester.plot_results(summary_df, 'f1_macro', saved_file_name, results_folder)

	# Plot summary statistics about the usage of each operator and the number of operators selected in the grid
	# search for each dataset.
	count_plus = 0
	count_mult = 0
	count_minus = 0
	count_div = 0
	count_min = 0
	count_max = 0
	operations_count_arr = []
	for r in range(len(summary_df)):
		h = summary_df.iloc[r]['Best Hyperparameters']
		if h == "{}":
			continue
		s = h.split()
		ops_curr_row = 0
		for s_idx in range(len(s)-1):
			val = ("True" in s[s_idx+1])
			if "plus" in s[s_idx]:
				count_plus += val
			if "mult" in s[s_idx]:
				count_mult += val
			if "minus" in s[s_idx]:
				count_minus += val
			if "div" in s[s_idx]:
				count_div += val
			if "min" in s[s_idx]:
				count_min += val
			if "max" in s[s_idx]:
				count_max += val
			ops_curr_row += val
		operations_count_arr.append(ops_curr_row)

	# Plot the number of times each operator is used
	fig, ax = plt.subplots()
	counts_arr = [count_plus, count_mult, count_minus, count_div, count_min, count_max]
	ax.bar(
		x=range(0, 6),
		height=counts_arr,
		tick_label=["Plus", "Times", "Minus", "Div", "Min", "Max"]
	)
	fig.tight_layout()
	ax.set_title("Counts of Operators")
	ax.set_yticks(range(max(counts_arr)+1))
	ax.set_yticklabels([str(x) for x in range(max(counts_arr)+1)])
	n = datetime.now()
	dt_string = n.strftime("%d_%m_%Y_%H_%M_%S")
	results_plot_filename = results_folder + "\\Operators_counts_" + dt_string + ".png"
	fig.savefig(results_plot_filename, bbox_inches='tight', dpi=150)

	# Plot the number of operators used per dataset
	fig, ax = plt.subplots()
	num_bars = max(operations_count_arr)+1
	heights_arr = [0]*num_bars
	for h in range(num_bars):
		heights_arr[h] = operations_count_arr.count(h)
	ax.bar(
		x=range(0, num_bars),
		height=heights_arr,
		tick_label=[str(x) for x in range(0,num_bars)]
	)
	fig.tight_layout()
	ax.set_title("Number of Operators Use By Dataset")
	ax.set_yticks(range(max(operations_count_arr)+1))
	ax.set_yticklabels([str(x) for x in range(max(operations_count_arr)+1)])
	results_plot_filename = results_folder + "\\Operators_per_model_" + dt_string + ".png"
	fig.savefig(results_plot_filename, bbox_inches='tight', dpi=150)


def test_all_models_classification(datasets_tester, test_model_func, partial_result_folder, results_folder):
	# Decision Tree
	pipe1 = Pipeline([('model', tree.DecisionTreeClassifier(random_state=0))])
	pipe2 = Pipeline([('arith', ArithmeticFeatures()), ('model', tree.DecisionTreeClassifier(random_state=0))])
	estimators_arr = [
		("DT", "Original Features", "", pipe1),
		("DT", "Arithmetic-based Features", "", pipe2)]
	test_model_func(datasets_tester, estimators_arr, partial_result_folder, results_folder, "Avg f1_macro")

	# RandomForest
	pipe1 = Pipeline([('model', RandomForestClassifier(random_state=0))])
	pipe2 = Pipeline([('arith', ArithmeticFeatures()), ('model', RandomForestClassifier(random_state=0))])
	estimators_arr = [
		("RF", "Original Features", "", pipe1),
		("RF", "Arithmetic-based Features", "", pipe2)]
	test_model_func(datasets_tester, estimators_arr, partial_result_folder, results_folder, "Avg f1_macro")

	# kNN
	pipe1 = Pipeline([('model', KNeighborsClassifier())])
	pipe2 = Pipeline([('arith', ArithmeticFeatures()), ('model', KNeighborsClassifier())])
	estimators_arr = [
		("kNN", "Original Features", "", pipe1),
		("kNN", "Arithmetic-based Features", "", pipe2)]
	test_model_func(datasets_tester, estimators_arr, partial_result_folder, results_folder, "Avg f1_macro")

	# Logistic Regression
	pipe1 = Pipeline([('model', LogisticRegression(penalty='l1', solver='liblinear', C=0.1))])
	pipe2 = Pipeline([('arith', ArithmeticFeatures()), ('model', LogisticRegression(penalty='l1', solver='liblinear', C=0.1))])
	estimators_arr = [
		("LR", "Original Features", "", pipe1),
		("LR", "Arithmetic-based Features", "", pipe2)]
	test_model_func(datasets_tester, estimators_arr, partial_result_folder, results_folder, "Avg f1_macro")

	# Gaussian Naive Bayes
	pipe1 = Pipeline([('model', GaussianNB())])
	pipe2 = Pipeline([('arith', ArithmeticFeatures()), ('model', GaussianNB())])
	estimators_arr = [
		("NB", "Original Features", "", pipe1),
		("NB", "Arithmetic-based Features", "", pipe2)]
	test_model_func(datasets_tester, estimators_arr, partial_result_folder, results_folder, "Avg f1_macro")

	# Extra Trees
	extra_tree1 = ExtraTreeClassifier(random_state=0)
	extra_tree2 = ExtraTreeClassifier(random_state=0)
	pipe1 = Pipeline([('model', BaggingClassifier(extra_tree1, random_state=0))])
	pipe2 = Pipeline([('arith', ArithmeticFeatures()), ('model', BaggingClassifier(extra_tree2, random_state=0))])
	estimators_arr = [
		("ET", "Original Features", "", pipe1),
		("ET", "Arithmetic-based Features", "", pipe2)]
	test_model_func(datasets_tester, estimators_arr, partial_result_folder, results_folder, "Avg f1_macro")

	# Ada Boost
	pipe1 = Pipeline([('model', AdaBoostClassifier())])
	pipe2 = Pipeline([('arith', ArithmeticFeatures()), ('model', AdaBoostClassifier())])
	estimators_arr = [
		("Ada", "Original Features", "", pipe1),
		("Ada", "Arithmetic-based Features", "", pipe2)]
	test_model_func(datasets_tester, estimators_arr, partial_result_folder, results_folder, "Avg f1_macro")

	# Gradient Boost
	pipe1 = Pipeline([('model', GradientBoostingClassifier(random_state=0))])
	pipe2 = Pipeline([('arith', ArithmeticFeatures()), ('model', GradientBoostingClassifier(random_state=0))])
	estimators_arr = [
		("GB", "Original Features", "", pipe1),
		("GB", "Arithmetic-based Features", "", pipe2)]
	test_model_func(datasets_tester, estimators_arr, partial_result_folder, results_folder, "Avg f1_macro")


def test_classification_default_parameters(datasets_tester, partial_result_folder, results_folder):
	if NUM_DATASETS_CLASSIFICATION_DEFAULT <= 0:
		return
	print_header("Classification with default parameters")
	test_all_models_classification(datasets_tester, test_model_default_params, partial_result_folder, results_folder)


def test_classification_parameter_search(datasets_tester, partial_result_folder, results_folder):
	if NUM_DATASETS_CLASSIFICATION_PARAMETER_SEARCH <= 0:
		return
	print_header("Classification with parameter search for best feature engineering parameters")
	test_all_models_classification(datasets_tester, test_model_param_search, partial_result_folder, results_folder)


def test_all_models_regression(datasets_tester, test_model_func, partial_result_folder, results_folder):
	# Decision Tree
	pipe1 = Pipeline([('model', tree.DecisionTreeRegressor(random_state=0))])
	pipe2 = Pipeline([('arith', ArithmeticFeatures()), ('model', tree.DecisionTreeRegressor(random_state=0))])
	estimators_arr = [
		("DT", "Original Features", "", pipe1),
		("DT", "Arithmetic-based Features", "", pipe2)]
	test_model_func(datasets_tester, estimators_arr, partial_result_folder, results_folder, "Avg NRMSE")

	# RandomForest
	pipe1 = Pipeline([('model', RandomForestRegressor(random_state=0))])
	pipe2 = Pipeline([('arith', ArithmeticFeatures()), ('model', RandomForestRegressor(random_state=0))])
	estimators_arr = [
		("RF", "Original Features", "", pipe1),
		("RF", "Arithmetic-based Features", "", pipe2)]
	test_model_func(datasets_tester, estimators_arr, partial_result_folder, results_folder, "Avg NRMSE")

	# kNN
	pipe1 = Pipeline([('model', KNeighborsRegressor())])
	pipe2 = Pipeline([('arith', ArithmeticFeatures()), ('model', KNeighborsRegressor())])
	estimators_arr = [
		("kNN", "Original Features", "", pipe1),
		("kNN", "Arithmetic-based Features", "", pipe2)]
	test_model_func(datasets_tester, estimators_arr, partial_result_folder, results_folder, "Avg NRMSE")

	# Linear Regression with Lasso Regularization
	pipe1 = Pipeline([('model', Lasso())])
	pipe2 = Pipeline([('arith', ArithmeticFeatures()), ('model', Lasso())])
	estimators_arr = [
		("LR", "Original Features", "", pipe1),
		("LR", "Arithmetic-based Features", "", pipe2)]
	test_model_func(datasets_tester, estimators_arr, partial_result_folder, results_folder, "Avg NRMSE")

	# Extra Trees
	extra_tree1 = ExtraTreeRegressor(random_state=0)
	extra_tree2 = ExtraTreeRegressor(random_state=0)
	pipe1 = Pipeline([('model', BaggingRegressor(extra_tree1, random_state=0))])
	pipe2 = Pipeline([('arith', ArithmeticFeatures()), ('model', BaggingRegressor(extra_tree2, random_state=0))])
	estimators_arr = [
		("ET", "Original Features", "", pipe1),
		("ET", "Arithmetic-based Features", "", pipe2)]
	test_model_func(datasets_tester, estimators_arr, partial_result_folder, results_folder, "Avg NRMSE")

	# Ada Boost
	pipe1 = Pipeline([('model', AdaBoostRegressor())])
	pipe2 = Pipeline([('arith', ArithmeticFeatures()), ('model', AdaBoostRegressor())])
	estimators_arr = [
		("Ada", "Original Features", "", pipe1),
		("Ada", "Arithmetic-based Features", "", pipe2)]
	test_model_func(datasets_tester, estimators_arr, partial_result_folder, results_folder, "Avg NRMSE")

	# Gradient Boost
	pipe1 = Pipeline([('model', GradientBoostingRegressor(random_state=0))])
	pipe2 = Pipeline([('arith', ArithmeticFeatures()), ('model', GradientBoostingRegressor(random_state=0))])
	estimators_arr = [
		("GB", "Original Features", "", pipe1),
		("GB", "Arithmetic-based Features", "", pipe2)]
	test_model_func(datasets_tester, estimators_arr, partial_result_folder, results_folder, "Avg NRMSE")


def test_regression_default_parameters(datasets_tester, partial_result_folder, results_folder):
	if NUM_DATASETS_REGRESSION_DEFAULT <= 0:
		return 

	print_header("Regression with default parameters")
	test_all_models_regression(datasets_tester, test_model_default_params, partial_result_folder, results_folder)


def test_regression_parameter_search(datasets_tester, partial_result_folder, results_folder):
	if NUM_DATASETS_REGRESSION_DEFAULT_PARAMETER_SEARCH <= 0:
		return

	print_header("Regression with parameter search for best feature engineering parameters")

	pipe1 = Pipeline([('dt', tree.DecisionTreeRegressor(random_state=0))])
	pipe2 = Pipeline([('arith', ArithmeticFeatures()), ('dt', tree.DecisionTreeRegressor(random_state=0))])

	orig_parameters = {}

	arith_parameters = {
		'arith__scale_data': (True, False),
		'arith__support_plus': (True, False),
		'arith__support_mult': (True, False),
		'arith__support_minus': (True, False),
		'arith__support_div': (True, False),
		'arith__support_min': (True, False),
		'arith__support_max': (True, False),
	}

	summary_df, saved_file_name = datasets_tester.run_tests_parameter_search(
		estimators_arr=[
			("DT", "Original Features", "Default", pipe1),
			("DT", "Arithmetic-based Features", "Default", pipe2)],
		parameters_arr=[orig_parameters, arith_parameters],
		search_method='grid',
		num_cv_folds=5,
		show_warnings=False,
		partial_result_folder=partial_result_folder,
		results_folder=results_folder,
		run_parallel=RUN_PARALLEL)  # If True, executes faster, but debugging is more difficult

	datasets_tester.summarize_results(summary_df, 'NRMSE', saved_file_name, results_folder)
	datasets_tester.plot_results(summary_df, 'NRMSE', saved_file_name, results_folder)


def main():
	cache_folder = "c:\\dataset_cache"
	partial_result_folder = "c:\\intermediate_results"
	results_folder = "c:\\results"

	# These are a bit slower, so excluded from some tests
	exclude_list = ["oil_spill", "fri_c4_1000_50", "fri_c3_1000_50", "fri_c1_1000_50", "fri_c2_1000_50",
					"waveform-5000", "mfeat-zernikemfeat-zernike", "auml_eml_1_b"]

	# Collect & test with the classification datasets
	datasets_tester = de.DatasetsTester(
		problem_type="classification",
		path_local_cache=cache_folder
	)
	datasets_tester.find_datasets(
		min_num_instances=500,
		max_num_instances=5_000,
		min_num_numeric_features=2,
		max_num_numeric_features=50)
	datasets_tester.collect_data(
		max_num_datasets_used=NUM_DATASETS_CLASSIFICATION_DEFAULT,
		use_automatic_exclude_list=True,
		exclude_list=exclude_list,
		save_local_cache=True,
		check_local_cache=True)

	test_classification_default_parameters(datasets_tester, partial_result_folder, results_folder)

	datasets_tester.collect_data(
		max_num_datasets_used=NUM_DATASETS_CLASSIFICATION_PARAMETER_SEARCH,
		use_automatic_exclude_list=True,
		exclude_list=exclude_list,
		save_local_cache=True,
		check_local_cache=True)

	test_classification_parameter_search(datasets_tester, partial_result_folder, results_folder)

	# Collect & test with the regression datasets
	datasets_tester = de.DatasetsTester(
		problem_type="regression",
		path_local_cache=cache_folder
	)
	datasets_tester.find_datasets(
		min_num_instances=500,
		max_num_instances=5_000,
		min_num_numeric_features=2,
		max_num_numeric_features=50,
	)
	datasets_tester.collect_data(
		max_num_datasets_used=NUM_DATASETS_REGRESSION_DEFAULT,
		use_automatic_exclude_list=True,
		save_local_cache=True,
		check_local_cache=True
	)

	test_regression_default_parameters(datasets_tester, partial_result_folder, results_folder)

	datasets_tester.collect_data(
		max_num_datasets_used=NUM_DATASETS_REGRESSION_DEFAULT_PARAMETER_SEARCH,
		use_automatic_exclude_list=True,
		save_local_cache=True,
		check_local_cache=True
	)

	test_regression_parameter_search(datasets_tester, partial_result_folder, results_folder)


if __name__ == "__main__":
	main()
