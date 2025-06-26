import sklearn.metrics
import numpy as np

def aurocScore(inData, outData):
	old = False
	if old:
		# --- Old code (fragile) ---
		allData = np.concatenate((inData, outData))
		labels = np.concatenate((np.zeros(len(inData)), np.ones(len(outData))))
		fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, allData, pos_label = 0)  
		return fpr, tpr, sklearn.metrics.auc(fpr, tpr)

		# Problems:
		# - Fails with empty inputs or NaNs
		# - Breaks downstream code with None or nan
	else:
		# --- New code (safe return) ---
		inData = np.asarray(inData)
		outData = np.asarray(outData)

		if len(inData) == 0: # Case 1: No known detections — cannot compute AUROC meaningfully
			print("Warning: No known-class detections (inData is empty). AUROC undefined.")
			return np.array([]), np.array([]), 0.0
		if len(outData) == 0: # Case 2: No OOD detections — perfect filtering by detector
			print("Warning: No OOD-class detections (outData is empty). OOD samples perfectly filtered.")
			fpr = np.array([0.0, 1.0])
			tpr = np.array([0.0, 1.0]) #???
			return fpr, tpr, 1.0 # Simulate perfect AUROC: all knowns scored, no OODs

		allData = np.concatenate((inData, outData))
		labels = np.concatenate((np.zeros(len(inData)), np.ones(len(outData))))

		if len(np.unique(labels)) < 2:
			print("Warning: Only one class present in AUROC labels.")
			return np.array([]), np.array([]), 0.0
		if np.any(np.isnan(allData)):
			print("Warning: NaN values found in AUROC input scores.")
			return np.array([]), np.array([]), 0.0

		try:
			fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, allData, pos_label=0)
			auc_val = sklearn.metrics.auc(fpr, tpr)
		except ValueError as e:
			print(f"ROC computation failed: {e}")
			return np.array([]), np.array([]), 0.0

		#print(tpr)
		#print(fpr)
		return fpr, tpr, auc_val

def tprAtFpr(tpr, fpr, fprRate = 0.05):
	old = False
	if old:
		# --- Old code (wrong) ---
		fprAdjust = np.abs(np.array(fpr)-fprRate)     # Finds absolute difference from desired FPR
		fprIdx = np.argmin(fprAdjust)                 # Gets index of closest FPR value
		tpratfpr = tpr[fprIdx]                        # Returns TPR at that index
		return tpratfpr, fpr[fprIdx]

		# Problem with old code:
		# - It picks the FPR value *closest* to the target, which might be greater than fprRate.
		# - This violates the "FPR ≤ target" requirement (common in safety/uncertainty metrics).
		# - It may select a threshold with invalid FPR and misleading TPR.
	else:
		# --- New code (correct) ---
		fpr = np.array(fpr)
		tpr = np.array(tpr)
		valid = fpr <= fprRate  # Only consider points that satisfy FPR constraint

		if not np.any(valid):
			print("Warning: No valid fpr point")
			return 0.0, fprRate  # If no such point exists, return TPR=0 safely

		max_idx = np.argmax(tpr[valid])  # Choose the threshold that gives max TPR under constraint
		valid_fpr = fpr[valid]
		valid_tpr = tpr[valid]
		return valid_tpr[max_idx], valid_fpr[max_idx]

def minUE(inData, outData):
	allData = np.concatenate((inData, outData))

	thresholds = np.sort(allData)
	uncertainty_error = []
	#assuming that lower means more uncertain, less confident
	for t in thresholds:
		#number of in detections that are rejected incorrectly
		in_error = np.sum(inData < t)/len(inData)
		#number of out detections that are accepted incorrectly
		out_error = np.sum(outData >= t)/len(outData)

		ue = 0.5*in_error + 0.5*out_error
		uncertainty_error += [ue]
	
	return np.min(uncertainty_error), uncertainty_error
		
def summarise_performance(inData, outData, fprRates = [], printRes = True, methodName = ''):
	results = {}

	fpr, tpr, auroc = aurocScore(inData, outData)
	#print(fpr)
	#print(tpr)
	#print(auroc)
	results['auroc'] = auroc
	results['fpr'] = list(fpr)
	results['tpr'] = list(tpr)

	specPoints = []
	for fprRate in fprRates:
		tprRate = tprAtFpr(tpr, fpr, fprRate)
		specPoints += [tprRate]

		results[f'tpr at fprRate {fprRate}'] = tprRate
	
	if printRes:
		print(f'Results for Method: {methodName}')
		print(f'------ AUROC: {round(auroc, 3)}')
		for point in specPoints:
			fp = point[1]
			tp = point[0]
			print(f'------ TPR at {round((100.*fp), 1)}FPR: {round((100.*tp), 1)}')

	return results