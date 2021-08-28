import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class ArithmeticFeatures:
	"""
	This tool generates new features based on arithmetic operations between existing numeric features. This may be
	useful for increasing the accuracy and/or interpretability of some models. As this tool is primarily to
	aid the interpretability of models, it uses strictly highly-interpretable operations.
	"""

	def __init__(
		self,
		determine_numeric_features=True,
		scale_data=False,
		support_plus=True,
		support_times=True,
		support_minus=True,
		support_div=True,
		support_min=False,
		support_max=False
	):
		"""
		Parameters
		----------
		determine_numeric_features: bool
			Features can only be generated for pairs of numeric features. If this is set True, the code
			will determine which are numeric. Otherwise it is assumed all features are numeric.

		scale_data: bool
			If True, each numeric column will be min-max scaled. This is False by default, as it does lower
			interpretability of the features. However, it can increase the accuracy for many downstream models.

		support_plus: bool
			If True, new features will be produced based on addition of each pair of numeric features

		support_times: bool
			If True, new features will be produced based on multiplication of each pair of numeric features

		support_minus: bool
			If True, new features will be produced based on subtraction of each pair of numeric features

		support_div: bool
			If True, new features will be produced based on the ratio of each pair of numeric features

		support_min: bool
			If True, new features will be produced based on the minimum of each pair of numeric features

		support_max: bool
			If True, new features will be produced based on the maximum of each pair of numeric features
		"""

		self.determine_numeric_features = determine_numeric_features
		self.scale_data = scale_data
		self.support_plus = support_plus
		self.support_times = support_times
		self.support_div = support_div
		self.support_minus = support_minus
		self.support_min = support_min
		self.support_max = support_max

		self.orig_X_df = None
		self.n_input_features_ = -1
		self.n_numeric_input_features_ = 0
		self.n_output_features_ = 0
		self.is_numeric_arr = []
		self.feature_names_ = []
		self.feature_sources_ = []
		self.scaler_ = None

	def fit(self, X):
		"""
		fit() simply determines the number of features that will be generated. As the new features are based
		on rotations, they do not depend on any specific data that must be fit to. 

		Parameters
		----------
		X: matrix

		Returns
		-------
		Returns self
		"""
		self.orig_X_df = pd.DataFrame(X)
		self.n_input_features_ = len(self.orig_X_df.columns)

		# Determine which features may be considered numeric
		if self.determine_numeric_features:
			self.is_numeric_arr = [
				1 if is_numeric_dtype(self.orig_X_df[self.orig_X_df.columns[c]]) and
					(self.orig_X_df[self.orig_X_df.columns[c]].nunique() > 2)
				else 0
				for c in range(self.n_input_features_)]
			self.n_numeric_input_features_ = self.is_numeric_arr.count(1)
		else:
			self.n_numeric_input_features_ = self.n_input_features_

		# Determine the number of features that will be created. We look at each pair of numeric features
		# (i.e., n(n-1)/2 pairs), for each operation.
		num_operators = self.support_plus + self.support_times + self.support_minus + self.support_div + \
						self.support_min + self.support_max
		self.n_output_features_ = self.n_numeric_input_features_ * (self.n_numeric_input_features_-1) * 0.5 * num_operators

		self.scaler_ = MinMaxScaler()
		self.scaler_.fit(X)

		return self

	def transform(self, X):
		"""
		Parameters
		----------
		X: matrix

		Returns
		-------
		Returns a new pandas dataframe containing the same rows and columns as the passed matrix X, as well 
		as the additional columns created. 
		"""

		orig_X_df = pd.DataFrame(X)
		assert len(orig_X_df.columns) == self.n_input_features_
		self.feature_sources_ = [()]*self.n_input_features_
		extended_X_np = orig_X_df.values.astype('float64')
		if self.scale_data:
			scaled_X_df = pd.DataFrame(self.scaler_.transform(orig_X_df), columns=orig_X_df.columns)
		else:
			scaled_X_df = orig_X_df
		scaled_X_np = scaled_X_df.values

		extended_X_np, ext_feature_sources, ext_feature_names = self.expand_matrix(extended_X_np, scaled_X_np)
		self.feature_names_ = list(orig_X_df.columns) + ext_feature_names
		self.feature_sources_.extend(ext_feature_sources)
		extended_X_df = pd.DataFrame(extended_X_np, columns=self.feature_names_)
		extended_X_df = extended_X_df.fillna(0.0)
		extended_X_df = extended_X_df.replace([np.inf, -np.inf], 0.0)
		extended_X_df.index = orig_X_df.index
		return extended_X_df

	def expand_matrix(self, extended_X_np, scaled_X_np):
		def add_column(extended_X_np, new_col, op_str):
			extended_X_np = np.hstack((extended_X_np, new_col.reshape(-1, 1)))
			feature_sources.append((i, j, op_str))
			feature_names.append(orig_cols[i] + " " + op_str + " " + orig_cols[j])
			return extended_X_np

		feature_sources = []
		feature_names = []
		orig_cols = self.orig_X_df.columns.astype(str)
		for i in range(self.n_input_features_-1):
			if self.is_numeric_arr[i] == 0:
				continue
			for j in range(i+1, self.n_input_features_):
				if self.is_numeric_arr[j] == 0:
					continue
				if self.support_plus:
					new_col = scaled_X_np[:, i] + scaled_X_np[:, j]
					extended_X_np = add_column(extended_X_np, new_col, "plus")
				if self.support_times:
					new_col = scaled_X_np[:, i] * scaled_X_np[:, j]
					extended_X_np = add_column(extended_X_np, new_col, "times")
				if self.support_minus:
					new_col = scaled_X_np[:, i] - scaled_X_np[:, j]
					extended_X_np = add_column(extended_X_np, new_col, "minus")
				if self.support_div:
					old_settings = np.seterr(all='ignore')
					new_col = scaled_X_np[:, i] / scaled_X_np[:, j]
					np.seterr(**old_settings)
					np.nan_to_num(new_col, copy=False, posinf=0.0, neginf=0.0)
					extended_X_np = add_column(extended_X_np, new_col, "divide")
				if self.support_min:
					new_col = np.minimum(scaled_X_np[:, i], scaled_X_np[:, j])
					extended_X_np = np.hstack((extended_X_np, new_col.reshape(-1, 1)))
					feature_sources.append((i, j, "min"))
					feature_names.append("min(" + orig_cols[i] + " and " + orig_cols[j] + ")")
				if self.support_max:
					new_col = np.maximum(scaled_X_np[:, i], scaled_X_np[:, j])
					extended_X_np = np.hstack((extended_X_np, new_col.reshape(-1, 1)))
					feature_sources.append((i, j, "max"))
					feature_names.append("max(" + orig_cols[i] + " and " + orig_cols[j] + ")")
		return extended_X_np, feature_sources, feature_names

	def fit_transform(self, X, y=None, **fit_params):
		"""
		Calls fit() and transform()

		Parameters
		----------
		X: matrix

		y: Unused

		fit_params: Unused

		Returns
		-------
		Returns a new pandas dataframe containing the same rows and columns as the passed matrix X, as well 
		as the additional columns created. 
		"""
		self.fit(X)
		return self.transform(X) 

	def get_feature_names(self):
		"""
		Returns the list of column names. This includes the original columns and the generated columns. The
		generated columns have names of the form: "R_" followed by a count. The generated columns have little
		meaning in themselves except as described as a rotation of two original features. 
		"""
		return self.feature_names_ 

	def get_feature_sources(self):
		"""
		Returns the list of column sources. This has an element for each column. For the original columns, this is empty and
		for generated columns, this lists the pair of original columns from which it was generated. 
		"""
		return self.feature_sources_

	def get_params(self, deep=True):		
		return \
			{
				"scale_data": self.scale_data,
				"support_plus": self.support_plus,
				"support_times": self.support_times,
				"support_minus": self.support_minus,
				"support_div": self.support_div,
				"support_min": self.support_min,
				"support_max": self.support_max
			}

	def set_params(self, **params):
		for key, value in params.items():
			setattr(self, key, value)
		return self
